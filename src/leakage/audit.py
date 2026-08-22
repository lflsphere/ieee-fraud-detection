"""Phase 8 — machine-checked leakage audits.

Kept in its own package, deliberately separate from ``src/features``: an audit
that lives inside the code it audits tends to drift into agreeing with it.
Everything here re-derives the expected value of a feature from first
principles, by brute force on a subsample, and compares it against what the
pipeline actually produced.

Three independent checks:

1. :func:`audit_point_in_time` — for a sample of rows, recompute each
   point-in-time feature using only rows with a strictly smaller
   ``TransactionDT`` and assert the pipeline agrees.
2. :func:`audit_future_invariance` — the decisive test. Perturb the *future*
   (shuffle the labels and multiply the amounts of every row after time *t*)
   and rebuild the features. Any feature whose value changes for a row before
   *t* has seen the future. This catches leaks that a re-derivation would miss
   because it shares the same wrong assumption.
3. :func:`audit_fit_on_train` — refit the fit-on-train statistics on the
   training rows only versus on the whole frame, and report which features
   move. Anything that moves is a preprocessing-leakage surface.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.features.build_features import (
    SECONDS_PER_DAY,
    FeatureBuilder,
    build_feature_matrix,
)

log = logging.getLogger(__name__)


def point_in_time_feature_names(columns) -> list[str]:
    """Features whose value depends on rows other than their own."""
    pit = [c for c in columns
           if c.startswith("uid_") or c.endswith("_te")]
    return pit


def audit_point_in_time(df_sorted: pd.DataFrame, X: pd.DataFrame,
                        entity_cols: list[str],
                        prefix: str,
                        n_sample: int = 400,
                        seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Brute-force re-derivation of one entity's PIT aggregates.

    For ``n_sample`` randomly chosen rows we scan the whole frame for rows in
    the same entity with a strictly smaller ``TransactionDT`` and recompute the
    prior count / prior mean amount from scratch. This is O(n) per sampled row
    and therefore only run on a sample, which is enough: a leak is systematic,
    not sporadic.
    """
    from src.features.build_features import _concat_key

    rng = np.random.default_rng(seed)
    key = _concat_key(df_sorted, entity_cols).to_numpy()
    t = df_sorted[config.TIME_COL].to_numpy(dtype=np.float64)
    amt = df_sorted["TransactionAmt"].to_numpy(dtype=np.float64)

    rows = rng.choice(len(df_sorted), size=min(n_sample, len(df_sorted)),
                      replace=False)
    recs = []
    for i in rows:
        prior = (key == key[i]) & (t < t[i])
        n_prior = int(prior.sum())
        exp_mean = float(amt[prior].mean()) if n_prior else np.nan
        got_count = float(X[f"{prefix}_prior_count"].iloc[i])
        got_mean = float(X[f"{prefix}_prior_amt_mean"].iloc[i])
        recs.append({
            "row": int(i),
            "expected_count": n_prior, "actual_count": got_count,
            "count_ok": n_prior == got_count,
            "expected_mean": exp_mean, "actual_mean": got_mean,
            "mean_ok": (np.isnan(exp_mean) and np.isnan(got_mean))
                       or np.isclose(exp_mean, got_mean, rtol=1e-4, atol=1e-3),
        })
    return pd.DataFrame(recs)


def audit_future_invariance(df: pd.DataFrame,
                            train_mask: np.ndarray,
                            cut_fraction: float = 0.6,
                            seed: int = config.RANDOM_SEED,
                            **fb_kwargs) -> pd.DataFrame:
    """Corrupt the future; anything in the past that moves has leaked.

    Returns one row per feature with the maximum absolute relative change
    observed among rows strictly before the cut. A correctly point-in-time
    feature must show exactly zero.

    Note the two perturbations are complementary: shuffling ``isFraud`` catches
    target leakage, scaling ``TransactionAmt`` catches value leakage. Both are
    applied only after the cut.

    The fitted statistics are deliberately **held fixed** across the two runs
    (the builder is fitted once, on the unperturbed frame, and used to
    transform both). Without that, the audit conflates two different questions:
    training rows that happen to fall after the cut would shift the encoders'
    fitted parameters, and every past row would appear to "change" because the
    *transformation* changed rather than because it saw the future. Dependence
    on the fit population is a separate concern, measured by
    :func:`audit_fit_on_train`.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    d0 = df.iloc[order].reset_index(drop=True)
    m0 = train_mask[order]

    cut = int(len(d0) * cut_fraction)
    fb = FeatureBuilder(**fb_kwargs).fit(d0, m0)
    X0 = fb.transform(d0)

    d1 = d0.copy()
    fut = slice(cut, len(d1))
    y_fut = d1[config.TARGET].to_numpy().copy()
    y_fut[cut:] = rng.permutation(y_fut[cut:])
    d1[config.TARGET] = y_fut
    a = d1["TransactionAmt"].to_numpy(dtype=np.float64).copy()
    a[fut] = a[fut] * 7.0 + 13.0
    d1["TransactionAmt"] = a
    X1 = fb.transform(d1)      # same fitted statistics; only the data differs

    recs = []
    for c in X0.columns:
        if c == config.ID_COL:
            continue
        a0 = X0[c].to_numpy(dtype=np.float64)[:cut]
        a1 = X1[c].to_numpy(dtype=np.float64)[:cut]
        both_nan = np.isnan(a0) & np.isnan(a1)
        diff = np.abs(a0 - a1)
        diff[both_nan] = 0.0
        nan_mismatch = int((np.isnan(a0) ^ np.isnan(a1)).sum())
        recs.append({"feature": c,
                     "max_abs_change": float(np.nanmax(diff)) if len(diff) else 0.0,
                     "n_changed": int(np.nansum(diff > 1e-9)) + nan_mismatch,
                     "leaks_future": bool(np.nansum(diff > 1e-9) + nan_mismatch)})
    return pd.DataFrame(recs).sort_values("n_changed", ascending=False)


def audit_fit_on_train(df: pd.DataFrame, train_mask: np.ndarray,
                       **fb_kwargs) -> pd.DataFrame:
    """Which features would change if statistics were fit on the full frame?

    Those are exactly the preprocessing-leakage surfaces. We do not expect the
    list to be empty — frequency encodings and bin edges legitimately depend on
    a fitted population. The point of the audit is that the list matches the
    features we *declared* as fit-on-train, with nothing unexpected on it.
    """
    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    d = df.iloc[order].reset_index(drop=True)
    m = train_mask[order]

    X_correct = FeatureBuilder(**fb_kwargs).fit(d, m).transform(d)
    X_leaky = FeatureBuilder(**fb_kwargs).fit(d, np.ones(len(d), bool)).transform(d)

    recs = []
    for c in X_correct.columns:
        if c == config.ID_COL:
            continue
        a0 = X_correct[c].to_numpy(dtype=np.float64)
        a1 = X_leaky[c].to_numpy(dtype=np.float64)
        d_ = np.abs(a0 - a1)
        d_[np.isnan(a0) & np.isnan(a1)] = 0.0
        changed = int(np.nansum(d_ > 1e-9)) + int((np.isnan(a0) ^ np.isnan(a1)).sum())
        if changed:
            recs.append({"feature": c, "n_rows_changed": changed,
                         "frac_rows_changed": changed / len(d)})
    return pd.DataFrame(recs).sort_values("n_rows_changed", ascending=False)


def leaky_target_encode_full_data(gid: np.ndarray, y: np.ndarray,
                                  smoothing: float = 50.0) -> np.ndarray:
    """The wrong way to target-encode, kept for the Phase 8 worked example.

    Uses the mean of the **entire** dataset per group, including the row's own
    label and every future row's label. This is the single most common way to
    destroy a fraud model's backtest, which is why it is written down
    explicitly here rather than merely described.
    """
    prior = float(y.mean())
    s = pd.Series(y).groupby(gid).transform("sum").to_numpy()
    n = pd.Series(y).groupby(gid).transform("size").to_numpy()
    return ((s + prior * smoothing) / (n + smoothing)).astype(np.float32)
