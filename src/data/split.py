"""Chronological (point-in-time) splitting and time-aware cross-validation.

Why not ``train_test_split(shuffle=True)`` or plain ``StratifiedKFold``?
----------------------------------------------------------------------
Fraud is an **adversarial, non-stationary** process (Phase 1): the mix of
attacks in month 6 is not the mix in month 1.  A shuffled split puts
transactions from the *same* compromised card on both sides of the split, so
the model is scored on its ability to recognise a fraud ring it has already
seen rather than on its ability to generalise to the next one.  That inflates
every metric and is precisely the temporal leakage Phase 8 audits.

Everything here orders rows by ``TransactionDT`` and only ever lets earlier
rows inform later ones.  ``TransactionDT`` is a *timedelta from an unknown
reference*, so we use it only for ordering and for elapsed-time arithmetic,
never as a calendar timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class Split:
    """Row-index arrays for one chronological three-way partition."""
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for name, idx in (("train", self.train_idx), ("valid", self.valid_idx),
                          ("test", self.test_idx)):
            sub = df.iloc[idx]
            rows.append({
                "part": name,
                "n": len(sub),
                "dt_min": float(sub[config.TIME_COL].min()),
                "dt_max": float(sub[config.TIME_COL].max()),
                "fraud_rate": float(sub[config.TARGET].mean())
                if config.TARGET in sub else np.nan,
            })
        return pd.DataFrame(rows)


def chronological_split(
    df: pd.DataFrame,
    valid_fraction: float = config.VALID_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
) -> Split:
    """Split *by time*: earliest -> train, then valid, latest -> test.

    Returns positional indices into ``df`` **as given** (the function sorts
    internally and maps back), so callers do not have to pre-sort.

    Note on "stratified": the plan asks for a stratified *and* time-based
    split.  Those two are in direct conflict — forcing an equal fraud rate in
    each time block would require moving transactions across time, which is
    the leak we are trying to prevent.  We resolve it the safe way: the split
    is strictly chronological, and we *report* the realised fraud rate per
    partition (``Split.describe``) so any class-balance drift is visible and
    accounted for rather than engineered away.  Stratification is applied only
    where it is harmless — in the class-weighting of the models themselves.
    """
    if not 0 < valid_fraction + test_fraction < 1:
        raise ValueError("valid_fraction + test_fraction must lie in (0, 1)")

    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    n = len(order)
    n_test = int(round(n * test_fraction))
    n_valid = int(round(n * valid_fraction))
    n_train = n - n_valid - n_test

    return Split(
        train_idx=order[:n_train],
        valid_idx=order[n_train:n_train + n_valid],
        test_idx=order[n_train + n_valid:],
    )


def expanding_window_folds(
    df: pd.DataFrame,
    n_folds: int = config.N_TIME_FOLDS,
    holdout_idx: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV folds over the time axis.

    Fold *k* trains on time blocks ``0..k`` and validates on block ``k+1``.
    An expanding (rather than sliding) window is used because fraud volume
    grows over the period and discarding early history would starve the early
    folds; the cost is that later folds see more data than earlier ones, which
    we account for by reporting per-fold metrics, not just the mean.

    ``holdout_idx`` — if given, those rows are removed entirely so the final
    chronological test block never appears in any fold.
    """
    mask = np.ones(len(df), dtype=bool)
    if holdout_idx is not None:
        mask[holdout_idx] = False

    pos = np.flatnonzero(mask)
    order = pos[np.argsort(df[config.TIME_COL].to_numpy()[pos], kind="stable")]

    blocks = np.array_split(order, n_folds + 1)
    folds = []
    for k in range(n_folds):
        tr = np.concatenate(blocks[: k + 1])
        va = blocks[k + 1]
        folds.append((tr, va))
    return folds


def fold_summary(df: pd.DataFrame,
                 folds: list[tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Per-fold sizes, time ranges and fraud rates — used in Phase 7's report."""
    rows = []
    for k, (tr, va) in enumerate(folds):
        rows.append({
            "fold": k,
            "n_train": len(tr),
            "n_valid": len(va),
            "train_dt_max": float(df[config.TIME_COL].to_numpy()[tr].max()),
            "valid_dt_min": float(df[config.TIME_COL].to_numpy()[va].min()),
            "train_fraud_rate": float(df[config.TARGET].to_numpy()[tr].mean()),
            "valid_fraud_rate": float(df[config.TARGET].to_numpy()[va].mean()),
        })
    return pd.DataFrame(rows)


def assert_no_time_overlap(df: pd.DataFrame,
                           folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Hard guard used by tests and by the Phase 8 audit.

    Every training row in a fold must be strictly at or before every
    validation row in that fold.  If this ever fails, the fold generator has
    introduced temporal leakage.
    """
    t = df[config.TIME_COL].to_numpy()
    for k, (tr, va) in enumerate(folds):
        if t[tr].max() > t[va].min():
            raise AssertionError(
                f"fold {k}: train reaches DT={t[tr].max()} but validation "
                f"starts at DT={t[va].min()} — temporal leakage."
            )
