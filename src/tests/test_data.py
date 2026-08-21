"""Smoke tests for Phase 0 data loading and Phase 7 split machinery.

These run against a tiny synthetic frame with the real schema's shape, so the
suite stays fast and does not require the (untracked) raw CSVs.  The one test
that needs real data skips itself when the parquet cache is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.load import downcast, normalise_identity_columns
from src.data.split import (
    assert_no_time_overlap,
    chronological_split,
    expanding_window_folds,
)


@pytest.fixture
def toy() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        config.ID_COL: np.arange(n),
        # Deliberately shuffled in row order: the split must sort by time
        # itself rather than assume the frame arrives sorted.
        config.TIME_COL: rng.permutation(np.arange(n) * 60.0),
        config.TARGET: (rng.random(n) < 0.035).astype(int),
        "TransactionAmt": rng.lognormal(4, 1, n),
        "ProductCD": rng.choice(list("WCHRS"), n),
    })


def test_normalise_identity_columns():
    df = pd.DataFrame({"TransactionID": [1], "id-01": [0.0], "DeviceType": ["m"]})
    out = normalise_identity_columns(df)
    assert "id_01" in out.columns and "id-01" not in out.columns


def test_downcast_shrinks_and_preserves_shape(toy):
    out = downcast(toy.copy())
    assert out.shape == toy.shape
    assert out["TransactionAmt"].dtype == np.float32
    assert out[config.ID_COL].dtype == np.int64  # ID stays exact
    assert str(out["ProductCD"].dtype) == "category"


def test_chronological_split_is_ordered_and_exhaustive(toy):
    sp = chronological_split(toy)
    t = toy[config.TIME_COL].to_numpy()
    assert t[sp.train_idx].max() < t[sp.valid_idx].min()
    assert t[sp.valid_idx].max() < t[sp.test_idx].min()
    allidx = np.concatenate([sp.train_idx, sp.valid_idx, sp.test_idx])
    assert len(np.unique(allidx)) == len(toy)


def test_folds_have_no_temporal_overlap_and_exclude_holdout(toy):
    sp = chronological_split(toy)
    folds = expanding_window_folds(toy, n_folds=4, holdout_idx=sp.test_idx)
    assert_no_time_overlap(toy, folds)
    holdout = set(sp.test_idx.tolist())
    for tr, va in folds:
        assert not (set(tr.tolist()) & holdout)
        assert not (set(va.tolist()) & holdout)


def test_expanding_window_actually_expands(toy):
    folds = expanding_window_folds(toy, n_folds=4)
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_assert_no_time_overlap_catches_a_leak(toy):
    """The guard must fail on a deliberately leaky fold, or it guards nothing."""
    idx = np.argsort(toy[config.TIME_COL].to_numpy())
    leaky = [(idx[500:900], idx[:400])]  # trains on the future, validates on past
    with pytest.raises(AssertionError):
        assert_no_time_overlap(toy, leaky)


@pytest.mark.skipif(
    not (config.INTERIM_DIR / "train_joined.parquet").exists(),
    reason="raw data / parquet cache not present in this environment",
)
def test_real_join_shape_and_rate():
    df = pd.read_parquet(config.INTERIM_DIR / "train_joined.parquet",
                         columns=[config.ID_COL, config.TARGET,
                                  "has_identity_record"])
    assert len(df) == 590_540
    assert df[config.ID_COL].is_unique
    # Documented in data/raw/README.md; a change here means different data.
    assert abs(df["has_identity_record"].mean() - 0.2442) < 1e-3
    assert abs(df[config.TARGET].mean() - 0.03499) < 1e-4
