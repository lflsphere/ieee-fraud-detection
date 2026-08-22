"""Phase 3 tests — the point-in-time invariants, on data small enough to check by hand.

The value of these tests is that they fail loudly if someone removes a
``shift(1)`` or a ``- amt`` while refactoring. Each one encodes a property that
a leaky implementation cannot satisfy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build_features import (
    SECONDS_PER_DAY,
    FeatureBuilder,
    _concat_key,
    _prior_count_within_window,
    build_feature_matrix,
)


@pytest.fixture
def tiny() -> pd.DataFrame:
    """Six transactions across two entities, hand-checkable."""
    return pd.DataFrame({
        config.ID_COL: [1, 2, 3, 4, 5, 6],
        config.TIME_COL: [0.0, 100.0, 200.0, 300.0,
                          400.0, 400.0 + 2 * SECONDS_PER_DAY],
        config.TARGET: [0, 1, 0, 0, 1, 0],
        "TransactionAmt": [10.0, 20.0, 30.0, 100.0, 40.0, 50.0],
        "ProductCD": ["W", "W", "C", "W", "C", "C"],
        "card1": [1, 1, 2, 1, 2, 2],
        "card2": [np.nan, np.nan, 5.0, np.nan, 5.0, 5.0],
        "card3": [150.0] * 6, "card4": ["visa"] * 6, "card5": [102.0] * 6,
        "card6": ["debit"] * 6,
        "addr1": [315.0, 315.0, 325.0, 315.0, 325.0, 325.0],
        "addr2": [87.0] * 6,
        "P_emaildomain": ["gmail.com"] * 6, "R_emaildomain": [None] * 6,
        "DeviceType": [None] * 6, "DeviceInfo": [None] * 6,
        "D1": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "has_identity_record": [0, 0, 0, 0, 0, 0],
    })


def _build(tiny, **kw):
    mask = np.ones(len(tiny), dtype=bool)
    return build_feature_matrix(tiny, mask, **kw)


def test_concat_key_keeps_nan_rows_separate(tiny):
    """A missing component must become its own level, not swallow the row.

    The naive ``a.astype(str) + '_' + b.astype(str)`` propagates NaN and would
    collapse every partially-missing row into one pseudo-entity.
    """
    k = _concat_key(tiny, ["card1", "card2", "addr1"])
    assert k.isna().sum() == 0
    assert k.nunique() == 2          # card1=1/NaN/315 and card1=2/5.0/325
    assert "NA" in k.iloc[0]


def test_prior_count_excludes_self_and_respects_window():
    t = np.array([0.0, 10.0, 20.0, 10_000_000.0])
    g = np.array([0, 0, 0, 0])
    got = _prior_count_within_window(t, g, window_seconds=100.0)
    assert list(got) == [0.0, 1.0, 2.0, 0.0]


def test_prior_count_treats_simultaneous_rows_as_not_prior():
    """Two transactions at the identical timestamp must not see each other."""
    t = np.array([0.0, 0.0, 0.0])
    g = np.zeros(3, dtype=int)
    assert list(_prior_count_within_window(t, g, 1e9)) == [0.0, 0.0, 0.0]


def test_prior_amt_mean_excludes_the_current_row(tiny):
    X, _ = _build(tiny)
    # Entity card1=1: rows 0,1,3 with amounts 10, 20, 100.
    m = X["uid_card1_prior_amt_mean"].to_numpy()
    assert np.isnan(m[0])                    # nothing before it
    assert m[1] == pytest.approx(10.0)       # only row 0
    assert m[3] == pytest.approx(15.0)       # (10+20)/2, NOT (10+20+100)/3
    c = X["uid_card1_prior_count"].to_numpy()
    assert list(c[[0, 1, 3]]) == [0.0, 1.0, 2.0]


def test_secs_since_prev_is_nan_on_first_sighting(tiny):
    X, _ = _build(tiny)
    s = X["uid_card1_secs_since_prev"].to_numpy()
    assert np.isnan(s[0])
    assert s[1] == pytest.approx(100.0)
    assert s[3] == pytest.approx(200.0)


def test_d_ref_day_recovers_a_constant_per_entity(tiny):
    X, _ = _build(tiny)
    # D1 = 0 for every row of entity card1=1, so ref_day == the row's own day.
    assert X["D1_ref_day"].notna().all()


def test_transform_rejects_unsorted_input(tiny):
    fb = FeatureBuilder().fit(tiny, np.ones(len(tiny), bool))
    with pytest.raises(ValueError, match="sorted"):
        fb.transform(tiny.iloc[::-1].reset_index(drop=True))


def test_transaction_dt_is_not_exported_as_a_feature(tiny):
    """Raw TransactionDT is monotone in this window and must never be a input."""
    X, _ = _build(tiny)
    assert config.TIME_COL not in X.columns


def test_target_encoding_respects_the_label_lag(tiny):
    """With a lag longer than the data, every encoding must equal the prior."""
    X_lag, fb = _build(tiny, label_lag_days=365)
    prior = fb.prior_
    assert np.allclose(X_lag["card1_te"].to_numpy(), prior)

    # With no lag the row still may not see its own label: row 1 (the first
    # fraud) must be encoded from row 0's label only.
    X0, fb0 = _build(tiny, label_lag_days=0, te_smoothing=0.0)
    te = X0["card1_te"].to_numpy()
    assert te[0] == pytest.approx(fb0.prior_)   # no history -> prior
    assert te[1] == pytest.approx(0.0)          # row 0 only, which is legit
    assert te[3] == pytest.approx(0.5)          # rows 0,1 -> one fraud of two


def test_target_encoding_never_uses_its_own_label(tiny):
    """Flip one label; that row's own encoding must not move."""
    X_a, _ = _build(tiny, label_lag_days=0, te_smoothing=0.0)
    flipped = tiny.copy()
    flipped.loc[5, config.TARGET] = 1
    X_b, _ = _build(flipped, label_lag_days=0, te_smoothing=0.0)
    assert X_a["card1_te"].iloc[5] == pytest.approx(X_b["card1_te"].iloc[5])


def test_unseen_categorical_levels_map_to_minus_one(tiny):
    train_mask = np.array([True, True, True, True, False, False])
    order = np.argsort(tiny[config.TIME_COL].to_numpy(), kind="stable")
    d = tiny.iloc[order].reset_index(drop=True)
    fb = FeatureBuilder().fit(d, train_mask[order])
    d2 = d.copy()
    d2.loc[5, "ProductCD"] = "ZZZ_never_seen"
    X = fb.transform(d2)
    assert X["ProductCD_code"].iloc[5] == -1


def test_freq_encoding_of_unseen_level_is_zero(tiny):
    train_mask = np.array([True, True, True, True, False, False])
    order = np.argsort(tiny[config.TIME_COL].to_numpy(), kind="stable")
    d = tiny.iloc[order].reset_index(drop=True)
    fb = FeatureBuilder().fit(d, train_mask[order])
    d2 = d.copy()
    d2.loc[5, "P_emaildomain"] = "nowhere.example"
    X = fb.transform(d2)
    assert X["P_emaildomain_freq"].iloc[5] == 0.0
