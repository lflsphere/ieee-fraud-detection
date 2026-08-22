"""Tests for Phase 8's audits — including that they catch a planted leak.

An audit that always passes is worthless. The decisive test here plants a
deliberately leaky feature and asserts the audit flags it; without that, a
green audit tells us nothing about whether the audit works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build_features import FeatureBuilder, build_feature_matrix
from src.leakage.audit import (
    audit_fit_on_train,
    audit_future_invariance,
    audit_point_in_time,
    leaky_target_encode_full_data,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 1500
    t = np.sort(rng.integers(0, 90 * 86_400, n).astype(float))
    card1 = rng.integers(1, 40, n)
    return pd.DataFrame({
        config.ID_COL: np.arange(n),
        config.TIME_COL: t,
        config.TARGET: (rng.random(n) < 0.06).astype(int),
        "TransactionAmt": rng.lognormal(4, 1, n),
        "ProductCD": rng.choice(list("WCHRS"), n),
        "card1": card1,
        "card2": rng.choice([100.0, 200.0, np.nan], n),
        "card3": 150.0, "card4": rng.choice(["visa", "mastercard"], n),
        "card5": 102.0, "card6": rng.choice(["debit", "credit"], n),
        "addr1": rng.choice([315.0, 325.0, np.nan], n), "addr2": 87.0,
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
        "R_emaildomain": None, "DeviceType": None, "DeviceInfo": None,
        "D1": rng.integers(0, 200, n).astype(float),
        "has_identity_record": rng.integers(0, 2, n).astype(np.int8),
    })


def _mask(frame):
    m = np.zeros(len(frame), bool)
    m[: int(len(frame) * 0.6)] = True
    return m


def test_pipeline_has_no_future_leakage(frame):
    fi = audit_future_invariance(frame, _mask(frame))
    leaking = fi.loc[fi["leaks_future"], "feature"].tolist()
    assert leaking == [], f"features saw the future: {leaking}"


def test_audit_catches_a_planted_leak(frame, monkeypatch):
    """Replace the PIT target encoder with a full-dataset one; audit must fail.

    This is the control for the test above. If the audit cannot detect a leak
    this blatant, a clean report from it means nothing.
    """
    def leaky(gid, t, y, lag_days, smoothing, prior):
        return leaky_target_encode_full_data(np.asarray(gid), np.asarray(y),
                                             smoothing)

    monkeypatch.setattr(FeatureBuilder, "lagged_target_encode",
                        staticmethod(leaky))
    fi = audit_future_invariance(frame, _mask(frame))
    leaking = set(fi.loc[fi["leaks_future"], "feature"])
    assert any(f.endswith("_te") for f in leaking), \
        "the audit failed to notice a full-dataset target encoding"


def test_pit_aggregates_match_brute_force(frame):
    m = _mask(frame)
    X, _ = build_feature_matrix(frame, m)
    d = frame.sort_values(config.TIME_COL).reset_index(drop=True)
    r = audit_point_in_time(d, X, ["card1", "card2", "card3", "card5", "addr1"],
                            "uid_card_addr", n_sample=120)
    assert r["count_ok"].all() and r["mean_ok"].all()


def test_fit_on_train_surface_contains_only_declared_features(frame):
    ft = audit_fit_on_train(frame, _mask(frame))
    unexpected = [f for f in ft["feature"]
                  if not (f.endswith("_freq") or f.endswith("_te")
                          or f.endswith("_code") or f == "amt_decile")]
    assert unexpected == [], f"undeclared fit-on-train dependency: {unexpected}"
    assert len(ft) > 0, "the audit found nothing at all — it is not running"
