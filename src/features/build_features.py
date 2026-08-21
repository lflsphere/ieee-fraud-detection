"""Phase 3 — feature engineering.

Leakage doctrine
----------------
Every feature in this module falls into exactly one of three classes, and the
class is stated in the feature dictionary (`reports/final/03_feature_dictionary.md`)
for every single column:

**(R) Row-local.**  Computed from the current row only (``log1p(TransactionAmt)``,
the assumed hour, an is-null indicator).  Structurally incapable of leaking.

**(F) Fit-on-train.**  Needs a statistic estimated from a population
(frequency-encoding counts, quantile bin edges, category vocabularies).  These
are learned by ``FeatureBuilder.fit`` from the **training rows only** and then
applied unchanged to validation and test.  Fitting them on the full frame would
be *preprocessing leakage* — the split would no longer be a clean simulation of
"train now, score later".

**(P) Point-in-time.**  Aggregates over other rows.  These are the dangerous
ones.  Every such feature here is computed with a strict
``groupby(entity).shift(1)``-style construction over time-sorted data, so a row
can only ever see rows with a **strictly smaller ``TransactionDT``**.  The
invariant is machine-checked by ``src/leakage/audit.py``, not merely asserted
here.

**(P+L) Point-in-time with label lag.**  Target encodings additionally need the
*label* of prior rows, and labels do not exist at scoring time — a chargeback
arrives weeks after the transaction.  A backward-looking target encoding that
uses yesterday's labels is still leakage in deployment, just a subtler kind.
Those features apply a configurable ``label_lag_days`` (default 30) so a row
only sees labels of transactions that were already old enough to have been
adjudicated.  Phase 8 quantifies what happens if you skip this.

``TransactionDT`` itself is deliberately **never** exported as a model feature:
it increases monotonically across the file, so any model would learn
"later ⇒ riskier", which is a fact about this six-month window and not about
fraud.  It is used only for ordering and elapsed-time arithmetic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400

# Entity proxies.  No customer ID exists (Phase 1 §1), so we approximate the
# latent account at three granularities and let the model decide which is
# informative.  Coarser proxies are more stable but merge distinct people;
# finer ones fragment a single person across address changes.
UID_DEFINITIONS: dict[str, list[str]] = {
    # Issuer/BIN-level: coarse, ~13.5k levels, very stable.
    "uid_card1": ["card1"],
    # Card + billing region: the workhorse proxy.
    "uid_card_addr": ["card1", "card2", "card3", "card5", "addr1"],
}

# Columns whose missingness is *not* simply a restatement of the identity join
# (which `has_identity_record` already captures).  Chosen from the Phase 2
# is-null ROC-AUC scan: these carry signal of their own.
ISNULL_COLS = [
    "dist1", "dist2", "D7", "D12", "D13", "D14", "D6", "D8", "D9",
    "M1", "M4", "M5", "M6", "M7", "M8", "M9",
    "card2", "card5", "addr1", "P_emaildomain", "R_emaildomain",
]

# `D*` columns documented as "days since a previous event".  `day - D` therefore
# recovers the *day of that event*, which is close to constant for one account
# and so acts as a much better entity key than the raw countdown.  Purely
# row-local arithmetic (class R) — no other row is consulted.
D_NORMALISE_COLS = ["D1", "D2", "D4", "D10", "D11", "D15"]

# Categorical axes worth crossing, chosen from the Phase 2 sub-population
# contrasts (ProductCD 5.7x, DeviceType 4.8x, assumed hour 4.6x).
INTERACTIONS: list[tuple[str, ...]] = [
    ("ProductCD", "card4"),
    ("ProductCD", "card6"),
    ("card1", "addr1"),
    ("DeviceType", "P_emaildomain"),
    ("P_emaildomain", "R_emaildomain"),
]

# Frequency-encoded (class F).  High-cardinality identifiers where "how common
# is this level" is more useful to a model than the level itself.
FREQ_ENCODE_COLS = [
    "card1", "card2", "card3", "card5", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31", "id_30",
    "id_33", "id_19", "id_20",
]

# Target-encoded (class P+L).  Kept short on purpose: each one is a leakage
# risk that has to be individually justified, so we only spend the risk where
# the Phase 2 contrasts were largest.
TARGET_ENCODE_COLS = ["card1", "addr1", "P_emaildomain", "ProductCD_x_card4"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _concat_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Build a composite string key, treating NaN as its own level.

    NaN is *not* dropped: "unknown billing address" is itself a meaningful
    entity signature, and dropping it would silently merge those rows with
    whatever level happened to be first.
    """
    out = df[cols[0]].astype("string").fillna("NA")
    for c in cols[1:]:
        out = out + "_" + df[c].astype("string").fillna("NA")
    return out


def _prior_count_within_window(times: np.ndarray, groups: np.ndarray,
                               window_seconds: float) -> np.ndarray:
    """For each row: how many *earlier* rows share its group within `window`.

    Implemented with a per-group ``searchsorted`` on the sorted time array, so
    the current row is excluded by construction (we search the left edge of the
    current timestamp block and subtract nothing for self).  Ties on
    ``TransactionDT`` are treated as *not* prior — two transactions with the
    identical timestamp cannot see each other.  That is the conservative
    choice: it can only ever under-count, never leak.
    """
    out = np.zeros(len(times), dtype=np.float32)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    t_sorted = times[order]
    boundaries = np.flatnonzero(np.diff(g_sorted)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(g_sorted)]):
        t = t_sorted[start:stop]                      # already time-ordered
        lo = np.searchsorted(t, t - window_seconds, side="left")
        first_at_t = np.searchsorted(t, t, side="left")
        out[order[start:stop]] = (first_at_t - lo).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureBuilder:
    """Fit-on-train statistics + point-in-time feature construction.

    Usage::

        fb = FeatureBuilder().fit(df, train_mask)
        X  = fb.transform(df)

    ``df`` must contain **all** rows (train + later) so that point-in-time
    aggregations can see the genuine history a deployed model would have.
    ``train_mask`` marks the rows the *statistics* may be estimated from.
    """

    label_lag_days: int = 30
    te_smoothing: float = 50.0
    min_freq: int = 1

    freq_maps: dict[str, pd.Series] = field(default_factory=dict)
    amt_bin_edges: np.ndarray | None = None
    cat_vocab: dict[str, pd.Index] = field(default_factory=dict)
    prior_: float = 0.0
    feature_names_: list[str] = field(default_factory=list)
    _fitted: bool = False

    # -- fit ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame, train_mask: np.ndarray) -> "FeatureBuilder":
        tr = df.loc[train_mask]
        log.info("fitting feature statistics on %d training rows", len(tr))

        # Amount bin edges (class F).  Deciles of the *training* amount
        # distribution; the Phase 2 U-shape is why a linear model needs bins.
        self.amt_bin_edges = np.unique(
            np.quantile(tr["TransactionAmt"].to_numpy(), np.linspace(0, 1, 11))
        )

        work = self._add_interaction_keys(tr[[c for c in self._interaction_inputs()
                                              if c in tr.columns]].copy())

        for col in FREQ_ENCODE_COLS:
            if col in tr.columns:
                self.freq_maps[col] = tr[col].value_counts(dropna=False)
        for name in [self._inter_name(t) for t in INTERACTIONS]:
            if name in work.columns:
                self.freq_maps[name] = work[name].value_counts(dropna=False)

        # Category vocabularies (class F): levels unseen in training become a
        # single "unseen" code rather than a new integer the model never
        # trained on.
        for col in config.CATEGORICAL_COLS:
            if col in tr.columns:
                self.cat_vocab[col] = pd.Index(
                    pd.Series(tr[col].astype("string")).dropna().unique()
                )

        self.prior_ = float(tr[config.TARGET].mean())
        self._fitted = True
        return self

    # -- transform ---------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")

        n = len(df)
        t = df[config.TIME_COL].to_numpy(dtype=np.float64)
        if not np.all(np.diff(t) >= 0):
            raise ValueError(
                "transform() expects rows sorted by TransactionDT ascending; "
                "the point-in-time features are only correct on sorted input."
            )

        out: dict[str, np.ndarray | pd.Series] = {}
        out[config.ID_COL] = df[config.ID_COL].to_numpy()

        # ---------------- (R) row-local -----------------------------------
        amt = df["TransactionAmt"].to_numpy(dtype=np.float64)
        # Amounts are log-normal (Phase 2): logging makes the scale usable by
        # the linear and neural models, which are sensitive to a 31,937x range.
        out["amt_log"] = np.log1p(amt).astype(np.float32)
        # Decimal part: proxies currency conversion / tax; 51.6% of rows are
        # whole-dollar and fraud is slightly likelier to be (Phase 2).
        dec = amt - np.floor(amt)
        out["amt_decimal"] = dec.astype(np.float32)
        out["amt_is_round"] = (dec < 1e-6).astype(np.int8)
        # Decile bin: lets the *linear* model express the observed U-shape in
        # fraud-rate-vs-amount, which no monotone transform of amount can.
        out["amt_decile"] = np.clip(
            np.digitize(amt, self.amt_bin_edges[1:-1]), 0, 9).astype(np.int8)

        day = t / SECONDS_PER_DAY
        # ASSUMED calendar alignment: TransactionDT is a timedelta from an
        # unknown origin, so these are periodic positions, not real clock time.
        # Kept because the 24h periodicity is unambiguously present (16x volume
        # swing, 4.6x fraud-rate swing) whatever the offset happens to be.
        out["dt_hour_assumed"] = ((t // 3600) % 24).astype(np.int8)
        out["dt_dow_assumed"] = ((t // SECONDS_PER_DAY) % 7).astype(np.int8)

        out["has_identity_record"] = df["has_identity_record"].to_numpy(np.int8)

        # Informative missingness (Phase 2: 53% of testable columns are MNAR).
        # The indicator preserves the signal that imputation would destroy.
        for c in ISNULL_COLS:
            if c in df.columns:
                out[f"{c}_isnull"] = df[c].isna().to_numpy().astype(np.int8)

        # `day - D` recovers the day of the referenced prior event, which is
        # near-constant per account and so is both a stabler feature and a
        # better entity key than the raw countdown.
        day_i = np.floor(day)
        for c in D_NORMALISE_COLS:
            if c in df.columns:
                out[f"{c}_ref_day"] = (day_i - df[c].to_numpy(np.float64)).astype(np.float32)

        # ---------------- entity keys --------------------------------------
        uids: dict[str, np.ndarray] = {}
        for name, cols in UID_DEFINITIONS.items():
            cols = [c for c in cols if c in df.columns]
            uids[name] = pd.factorize(_concat_key(df, cols), use_na_sentinel=False)[0]
        # Refined proxy: the card/address signature *plus* the recovered
        # account-start day. Two different cards that share a BIN and region
        # rarely share a start day too, so this splits the worst collisions.
        if "D1_ref_day" in out:
            refined = (_concat_key(df, [c for c in UID_DEFINITIONS["uid_card_addr"]
                                        if c in df.columns])
                       + "_" + pd.Series(out["D1_ref_day"]).astype("string").fillna("NA").to_numpy())
            uids["uid_card_addr_d1"] = pd.factorize(refined, use_na_sentinel=False)[0]

        # ---------------- (P) point-in-time aggregations --------------------
        for name, gid in uids.items():
            out.update(self._pit_entity_features(name, gid, t, amt))

        # ---------------- (F) frequency encodings ---------------------------
        inter_df = self._add_interaction_keys(
            df[[c for c in self._interaction_inputs() if c in df.columns]].copy())
        for col, counts in self.freq_maps.items():
            src = df[col] if col in df.columns else inter_df.get(col)
            if src is None:
                continue
            mapped = src.map(counts).to_numpy(dtype=np.float64)
            # Unseen level -> 0, meaning "never observed in training", which is
            # itself the informative answer for a rarity feature.
            out[f"{col}_freq"] = np.nan_to_num(mapped, nan=0.0).astype(np.float32)

        # ---------------- (F) categorical codes -----------------------------
        for col in config.CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            vocab = self.cat_vocab.get(col, pd.Index([]))
            codes = vocab.get_indexer(pd.Series(df[col].astype("string")))
            # -1 covers both NaN and levels unseen during fit; the tree models
            # get it as a real category, the linear/NN models see it after
            # one-hot/embedding of the same code.
            out[f"{col}_code"] = codes.astype(np.int32)

        # ---------------- (P+L) lagged target encodings ---------------------
        if config.TARGET in df.columns:
            yv = df[config.TARGET].to_numpy(dtype=np.float64)
            for col in TARGET_ENCODE_COLS:
                src = df[col] if col in df.columns else inter_df.get(col)
                if src is None:
                    continue
                gid = pd.factorize(src.astype("string").fillna("NA"),
                                   use_na_sentinel=False)[0]
                out[f"{col}_te"] = self.lagged_target_encode(
                    gid, t, yv, self.label_lag_days, self.te_smoothing, self.prior_)

        # ---------------- raw numeric pass-through --------------------------
        passthrough = (
            [c for c in config.C_COLS if c in df.columns]
            + [c for c in config.D_COLS if c in df.columns]
            + [c for c in config.V_COLS if c in df.columns]
            + [c for c in config.ID_NUMERIC_COLS if c in df.columns]
            + [c for c in ("dist1", "dist2") if c in df.columns]
        )
        feat = pd.DataFrame(out, index=df.index)
        feat = pd.concat([feat, df[passthrough].astype(np.float32)], axis=1)

        self.feature_names_ = [c for c in feat.columns if c != config.ID_COL]
        log.info("built %d features for %d rows", len(self.feature_names_), n)
        return feat

    # -- point-in-time entity block ---------------------------------------
    @staticmethod
    def _pit_entity_features(name: str, gid: np.ndarray, t: np.ndarray,
                             amt: np.ndarray) -> dict[str, np.ndarray]:
        """Strictly-backward-looking aggregates for one entity proxy.

        All of these answer "what did this entity look like *before now*".
        Implemented with ``groupby().shift(1)`` / ``cumsum`` on time-sorted
        input, so the current row's own amount never enters its own mean.
        """
        g = pd.Series(gid)
        s_amt = pd.Series(amt)
        grp = s_amt.groupby(g)

        # Velocity: how many transactions has this entity already made?
        # Rationale: a compromised card is drained in a burst (Phase 2 -
        # 88.3% of fraud sits in repeat-fraud entities), so position within
        # the entity's own sequence is informative. Bias/variance: low
        # variance (an integer count), but biased toward 0 for entities whose
        # history predates the observation window.
        prior_count = grp.cumcount().to_numpy().astype(np.float32)

        # Prior mean/std of amount for this entity. shift(1) is what makes it
        # point-in-time: without it, the current amount would appear inside
        # its own reference statistic, which is textbook target-adjacent
        # leakage even though no label is involved.
        csum = grp.cumsum().to_numpy() - amt
        prior_mean = np.divide(csum, prior_count, out=np.full_like(csum, np.nan),
                               where=prior_count > 0)
        csum_sq = (s_amt.pow(2).groupby(g).cumsum().to_numpy() - amt ** 2)
        var = np.divide(csum_sq, prior_count, out=np.full_like(csum, np.nan),
                        where=prior_count > 0) - np.square(prior_mean)
        prior_std = np.sqrt(np.clip(var, 0, None))

        # Is this transaction unusual *for this entity*? A ratio is scale-free,
        # which matters because entities differ hugely in typical ticket size.
        ratio = np.divide(amt, prior_mean, out=np.full_like(amt, np.nan),
                          where=np.isfinite(prior_mean) & (prior_mean > 0))

        # Recency: seconds since this entity's previous transaction. Short gaps
        # are the signature of automated card-testing.
        prev_t = pd.Series(t).groupby(g).shift(1).to_numpy()
        secs_since = t - prev_t

        return {
            f"{name}_prior_count": prior_count,
            f"{name}_prior_amt_mean": prior_mean.astype(np.float32),
            f"{name}_prior_amt_std": prior_std.astype(np.float32),
            f"{name}_amt_to_prior_mean": ratio.astype(np.float32),
            f"{name}_secs_since_prev": secs_since.astype(np.float32),
            f"{name}_prior_count_24h": _prior_count_within_window(t, gid, SECONDS_PER_DAY),
            f"{name}_prior_count_7d": _prior_count_within_window(t, gid, 7 * SECONDS_PER_DAY),
        }

    # -- target encoding ---------------------------------------------------
    @staticmethod
    def lagged_target_encode(gid: np.ndarray, t: np.ndarray, y: np.ndarray,
                             lag_days: float, smoothing: float,
                             prior: float) -> np.ndarray:
        """Expanding target encoding restricted to *adjudicated* history.

        For row *i* in group *g*, the encoding is the smoothed mean label over
        rows *j* with ``g_j == g_i`` **and** ``t_j <= t_i - lag``.

        Two separate leakage controls are in play:

        * ``t_j < t_i`` — the obvious one; a row must not see the future.
        * ``t_j <= t_i - lag`` — the subtle one. A fraud label originates in a
          chargeback that arrives weeks later, so at genuine scoring time you
          do **not** know whether yesterday's transaction was fraudulent. A
          zero-lag expanding encoding is therefore still leakage in
          deployment, just invisible to a naive backtest. Phase 8 measures the
          gap. ``lag_days=0`` reproduces the naive version for that comparison.

        Smoothing shrinks small-sample groups toward the training prior:
        ``(sum + prior*k) / (count + k)``. Without it a card seen once with one
        fraud encodes as 1.0 and the model memorises noise.
        """
        lag = lag_days * SECONDS_PER_DAY
        out = np.full(len(gid), prior, dtype=np.float32)

        order = np.argsort(gid, kind="stable")
        g_sorted = gid[order]
        boundaries = np.flatnonzero(np.diff(g_sorted)) + 1
        for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(g_sorted)]):
            idx = order[start:stop]
            tt = t[idx]                                  # time-ordered already
            yy = y[idx]
            cum_y = np.concatenate([[0.0], np.cumsum(yy)])
            # Number of same-group rows old enough to be adjudicated.
            k = np.searchsorted(tt, tt - lag, side="right")
            s = cum_y[k]
            out[idx] = ((s + prior * smoothing) / (k + smoothing)).astype(np.float32)
        return out

    # -- interaction plumbing ---------------------------------------------
    @staticmethod
    def _inter_name(cols: tuple[str, ...]) -> str:
        return "_x_".join(cols)

    @staticmethod
    def _interaction_inputs() -> list[str]:
        return sorted({c for t in INTERACTIONS for c in t})

    def _add_interaction_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        for cols in INTERACTIONS:
            if all(c in df.columns for c in cols):
                df[self._inter_name(cols)] = _concat_key(df, list(cols))
        return df


def build_feature_matrix(df: pd.DataFrame, train_mask: np.ndarray,
                         **kwargs) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Convenience wrapper: sort by time, fit on train rows, transform all."""
    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    df_sorted = df.iloc[order].reset_index(drop=True)
    mask_sorted = train_mask[order]
    fb = FeatureBuilder(**kwargs).fit(df_sorted, mask_sorted)
    return fb.transform(df_sorted), fb
