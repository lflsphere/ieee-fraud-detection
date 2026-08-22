"""Smoke tests for Phases 4-7: the model interface, the views, and the metrics.

Fast by design — a 4,000-row synthetic frame with the real column naming
conventions. These check contracts (shapes, ranges, protocol invariants), not
predictive quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation.metrics import (
    budget_threshold,
    calibration_table,
    evaluate_scores,
    metrics_at_budget,
)
from src.models.base import view_columns
from src.models.gbm import GBMModel
from src.models.linear import LinearBaseline
from src.models.nn import NeuralNet


@pytest.fixture
def synth():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(size=n)
    X = pd.DataFrame({
        config.ID_COL: np.arange(n),
        "amt_log": rng.normal(4, 1, n),
        "amt_decile": rng.integers(0, 10, n),
        "dt_hour_assumed": rng.integers(0, 24, n),
        "dt_dow_assumed": rng.integers(0, 7, n),
        "ProductCD_code": rng.integers(0, 5, n),
        "card1_code": rng.integers(0, 300, n),
        "uid_card1_prior_count": rng.poisson(3, n).astype(float),
        "card1_te": rng.random(n) * 0.1,
        "V1": z, "V2": z * 0.99 + rng.normal(0, 0.01, n), "V3": rng.normal(size=n),
        "vpca_g00_0": rng.normal(size=n), "vpca_g00_1": rng.normal(size=n),
        "v_cluster": rng.integers(0, 6, n),
        "v_cluster_dist": rng.random(n), "v_cluster_margin": rng.random(n),
    })
    X.loc[X.sample(200, random_state=1).index, "V3"] = np.nan   # exercise NaN paths
    logit = -3.5 + 1.2 * z + 0.4 * (X["amt_decile"] > 7)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y.to_numpy()


def test_views_partition_as_documented(synth):
    X, _ = synth
    base = set(view_columns(X, "base"))
    unsup = set(view_columns(X, "unsup"))
    compact = set(view_columns(X, "compact"))
    assert "V1" in base and "vpca_g00_0" not in base and "v_cluster" not in base
    assert base < unsup and "vpca_g00_0" in unsup and "v_cluster" in unsup
    assert "V1" not in compact and "vpca_g00_0" in compact
    assert config.ID_COL not in base | unsup | compact


def test_view_rejects_unknown_name(synth):
    X, _ = synth
    with pytest.raises(ValueError):
        view_columns(X, "nope")


@pytest.mark.parametrize("factory", [
    lambda: LinearBaseline(max_iter=40),
    lambda: GBMModel(n_estimators=25, early_stopping_rounds=0),
    lambda: NeuralNet(max_epochs=2, patience=2),
])
def test_model_contract(synth, factory):
    X, y = synth
    cols = view_columns(X, "unsup")
    m = factory().timed_fit(X.iloc[:3000][cols], y[:3000],
                            X.iloc[3000:][cols], y[3000:])
    p = m.predict_proba(X.iloc[3000:][cols])
    assert p.shape == (1000,)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert m.fit_seconds_ > 0


def test_models_handle_unseen_categorical_codes(synth):
    """A level absent from the training fold must not crash prediction."""
    X, y = synth
    cols = view_columns(X, "base")
    Xtr, Xte = X.iloc[:3000].copy(), X.iloc[3000:].copy()
    Xte.loc[Xte.index[0], "card1_code"] = 99999
    Xte.loc[Xte.index[1], "card1_code"] = -1
    for factory in (lambda: LinearBaseline(max_iter=30),
                    lambda: GBMModel(n_estimators=20, early_stopping_rounds=0),
                    lambda: NeuralNet(max_epochs=1, patience=1)):
        m = factory().fit(Xtr[cols], y[:3000])
        assert np.isfinite(m.predict_proba(Xte[cols])).all()


def test_budget_threshold_flags_the_right_volume():
    scores = np.linspace(0, 1, 10_000)
    y = (scores > 0.98).astype(int)
    m = metrics_at_budget(y, scores, budget=0.01)
    assert abs(m["n_flagged"] - 100) <= 2


def test_pr_auc_lift_is_relative_to_the_base_rate():
    """A random ranker must score a lift of ~1 whatever the base rate is."""
    rng = np.random.default_rng(0)
    for base in (0.01, 0.05, 0.2):
        y = (rng.random(20_000) < base).astype(int)
        out = evaluate_scores(y, rng.random(20_000))
        assert 0.7 < out["pr_auc_lift"] < 1.4


def test_perfect_ranking_scores_one():
    y = np.r_[np.zeros(950), np.ones(50)].astype(int)
    out = evaluate_scores(y, y.astype(float))
    assert out["pr_auc"] == pytest.approx(1.0)
    assert out["roc_auc"] == pytest.approx(1.0)


def test_calibration_table_bins_are_monotone_in_prediction():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    t = calibration_table(y, p, n_bins=10)
    assert t["mean_pred"].is_monotonic_increasing
    assert len(t) == 10
