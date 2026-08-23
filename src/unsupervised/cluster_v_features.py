"""Phase 4 — unsupervised representation of the anonymised ``V*`` block.

Motivation from Phase 2, not from taste: the 339 ``V*`` columns fall into 15
groups that share an exact missingness pattern, and within those groups they
are heavily collinear — 89 principal components retain 95% of within-group
variance. Two model families care about that (linear: unstable coefficients;
neural: wasted first-layer capacity), one does not (gradient boosting is
invariant to monotone rescaling and handles correlated splits). So the
compression is built as an *additional* representation, and Phase 9 reports
whether it helps each family, rather than assuming it helps all of them.

Leakage discipline
------------------
Everything here is fitted on the **training block only** — the scaler, the
imputer's medians, the PCA rotation and the K-means centroids. Fitting PCA on
the full frame is a textbook preprocessing leak: the rotation would encode the
covariance structure of the future. Nothing in this module ever sees
``isFraud``; the label is used only *afterwards*, to describe the clusters that
were already formed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src import config

log = logging.getLogger(__name__)


def v_missingness_groups(df: pd.DataFrame) -> list[list[str]]:
    """Recover the ``V*`` block structure from exact missingness patterns.

    Columns produced by the same upstream Vesta process go missing together,
    so hashing the null-mask of each column recovers the groups exactly — no
    threshold, no clustering hyperparameter. Phase 2 confirms this yields 15
    groups with near-contiguous index ranges.
    """
    vcols = [c for c in config.V_COLS if c in df.columns]
    buckets: dict[int, list[str]] = {}
    for c in vcols:
        buckets.setdefault(hash(df[c].isna().to_numpy().tobytes()), []).append(c)
    return sorted(buckets.values(), key=lambda g: int(g[0][1:]))


@dataclass
class VBlockCompressor:
    """Per-group PCA + a K-means clustering of the concatenated components.

    ``n_components_per_group`` is chosen adaptively: enough components to reach
    ``var_target`` of within-group variance, capped so that one large group
    cannot dominate the output width.

    ``n_clusters`` defaults to **6**, chosen from the stability sweep in
    ``scripts/04_cluster_v_block.py`` rather than from a round number. k=8 --
    the initial guess -- turned out to be the *least* reproducible setting on
    this data (mean pairwise ARI 0.56, worst pair 0.28) because it splits off
    two degenerate clusters of 81 and 2 rows whose membership flips with the
    seed. k=6 holds ARI 0.94 with the best silhouette in the sweep.
    """

    var_target: float = 0.95
    max_components_per_group: int = 12
    n_clusters: int = 6
    random_state: int = config.RANDOM_SEED

    groups_: list[list[str]] = field(default_factory=list)
    medians_: list[np.ndarray] = field(default_factory=list)
    means_: list[np.ndarray] = field(default_factory=list)
    scales_: list[np.ndarray] = field(default_factory=list)
    pcas_: list[PCA] = field(default_factory=list)
    kmeans_: KMeans | None = None
    component_names_: list[str] = field(default_factory=list)

    # -- internals ---------------------------------------------------------
    def _prepare(self, df: pd.DataFrame, gi: int, cols: list[str],
                 fit: bool) -> np.ndarray:
        """Impute with the training median, then standardise.

        Median rather than mean: several ``V*`` columns are counts with long
        right tails, where the mean sits outside the bulk of the data. The
        missingness *itself* is not thrown away — the `*_isnull` indicators and
        `has_identity_record` from Phase 3 carry it, and the group membership
        here is defined by it.
        """
        X = df[cols].to_numpy(dtype=np.float32)
        if fit:
            med = np.nanmedian(X, axis=0)
            med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
            self.medians_.append(med)
        med = self.medians_[gi]
        X = np.where(np.isnan(X), med, X)
        if fit:
            self.means_.append(X.mean(axis=0))
            s = X.std(axis=0)
            self.scales_.append(np.where(s > 1e-9, s, 1.0).astype(np.float32))
        return (X - self.means_[gi]) / self.scales_[gi]

    # -- fit ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> "VBlockCompressor":
        tr = df.loc[train_mask]
        self.groups_ = v_missingness_groups(df)
        log.info("V block: %d columns in %d missingness groups",
                 sum(len(g) for g in self.groups_), len(self.groups_))

        parts = []
        for gi, cols in enumerate(self.groups_):
            Xs = self._prepare(tr, gi, cols, fit=True)
            probe = PCA(n_components=min(len(cols), self.max_components_per_group),
                        random_state=self.random_state).fit(Xs)
            cum = np.cumsum(probe.explained_variance_ratio_)
            k = int(min(np.searchsorted(cum, self.var_target) + 1, len(probe.explained_variance_ratio_)))
            pca = PCA(n_components=k, random_state=self.random_state).fit(Xs)
            self.pcas_.append(pca)
            self.component_names_ += [f"vpca_g{gi:02d}_{j}" for j in range(k)]
            parts.append(pca.transform(Xs))
            log.info("  group %02d: %3d cols -> %2d PCs (%.1f%% var)",
                     gi, len(cols), k, 100 * cum[k - 1])

        Z = np.hstack(parts).astype(np.float32)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=10,
                              random_state=self.random_state).fit(Z)
        log.info("K-means k=%d fitted on %d x %d component matrix",
                 self.n_clusters, *Z.shape)
        return self

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [pca.transform(self._prepare(df, gi, cols, fit=False))
                 for gi, (cols, pca) in enumerate(zip(self.groups_, self.pcas_))]
        Z = np.hstack(parts).astype(np.float32)
        out = pd.DataFrame(Z, columns=self.component_names_, index=df.index)

        labels = self.kmeans_.predict(Z)
        out["v_cluster"] = labels.astype(np.int16)
        # Distance to the assigned centroid: "how typical is this transaction
        # of its own cluster". A large distance is an outlier flag that the
        # cluster ID alone cannot express.
        d = self.kmeans_.transform(Z)
        out["v_cluster_dist"] = d[np.arange(len(d)), labels].astype(np.float32)
        # Margin between the nearest and second-nearest centroid: low margin
        # means the assignment is ambiguous, so the model can discount it.
        part = np.partition(d, 1, axis=1)
        out["v_cluster_margin"] = (part[:, 1] - part[:, 0]).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# stability + label alignment (diagnostics, not part of the fitted model)
# ---------------------------------------------------------------------------
def cluster_stability(Z: np.ndarray, k_values: list[int], seeds: list[int],
                      subsample: int = 60_000,
                      random_state: int = config.RANDOM_SEED) -> pd.DataFrame:
    """How reproducible is the clustering across `k` and random seed?

    For each `k`, fit K-means under several seeds on a common subsample and
    report the mean pairwise adjusted Rand index between seed pairs, plus the
    silhouette score. A clustering that only appears at one seed is an artefact
    of initialisation, not structure in the data, and should not be fed
    downstream as a feature.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(Z), size=min(subsample, len(Z)), replace=False)
    Zs = Z[idx]

    rows = []
    for k in k_values:
        labelings = [KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Zs)
                     for s in seeds]
        aris = [adjusted_rand_score(labelings[i], labelings[j])
                for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
        sil = silhouette_score(Zs[:15_000], labelings[0][:15_000])
        inertia = KMeans(n_clusters=k, n_init=10,
                         random_state=seeds[0]).fit(Zs).inertia_
        rows.append({"k": k, "mean_ari": float(np.mean(aris)),
                     "min_ari": float(np.min(aris)), "silhouette": float(sil),
                     "inertia": float(inertia)})
    return pd.DataFrame(rows)


def cluster_label_alignment(labels: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Post-hoc: does an unsupervised partition happen to separate fraud?

    The label is used **only here**, after the clustering is already fixed.
    Using it to choose `k` or to select components would make the clustering
    supervised in disguise and would leak the target into a feature.
    """
    df = pd.DataFrame({"cluster": labels, "y": y})
    g = df.groupby("cluster")["y"].agg(n="size", fraud_rate="mean")
    g["lift_vs_base"] = g["fraud_rate"] / y.mean()
    return g.sort_values("fraud_rate", ascending=False)
