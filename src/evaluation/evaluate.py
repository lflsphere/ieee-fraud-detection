"""Phase 7 — the scripted evaluation harness.

One function, :func:`run_experiment`, drives every model through **identical**
folds and **identical** metrics. Nothing about the protocol lives in the model
classes, so a difference in the Phase 9 results table can only come from the
model or the feature view, never from the loop.

Protocol
--------
* The training file is split chronologically into ``train`` / ``valid`` /
  ``test`` (65 / 15 / 20 by row count, i.e. by time).
* The final ``test`` block is set aside and never touched by fold generation,
  hyperparameter choice or early stopping.
* Cross-validation over the remaining 80% uses **expanding-window** folds: fold
  *k* trains on time blocks 0..k and validates on block k+1. Every fold is
  checked by :func:`src.data.split.assert_no_time_overlap` before it is used.
* Within each fold, the model refits from scratch — including its own imputer,
  scaler, encoders and early stopping — so no statistic crosses the fold
  boundary.
* Metrics come from :mod:`src.evaluation.metrics`; PR-AUC is primary and is
  always reported next to that fold's base rate.

Deliberate omission: no hyperparameter search runs against the final test
block. Anything tuned is tuned on the CV folds. Selecting on the test block
would make the "generalisation to future transactions" claim false, which is
the one claim this phase exists to support.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.data.split import (
    assert_no_time_overlap,
    chronological_split,
    expanding_window_folds,
    fold_summary,
)
from src.evaluation.metrics import calibration_table, evaluate_scores
from src.models.base import FraudModel, view_columns

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    name: str
    per_fold: pd.DataFrame
    holdout: dict
    holdout_scores: np.ndarray
    fold_summary: pd.DataFrame
    calibration: pd.DataFrame
    importance: pd.Series | None
    history: pd.DataFrame | None


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase-3 features joined to Phase-4 components, plus the row metadata.

    Both parquet files were written in ``TransactionDT`` order by their
    respective drivers, so the concatenation is positional and no join key is
    needed — asserted below rather than assumed.
    """
    X = pd.read_parquet(config.INTERIM_DIR / "train_features.parquet")
    meta = pd.read_parquet(config.INTERIM_DIR / "train_meta.parquet")
    vpath = config.INTERIM_DIR / "train_vfeatures.parquet"
    if vpath.exists():
        V = pd.read_parquet(vpath)
        assert len(V) == len(X), "Phase-4 features have a different row count"
        X = pd.concat([X.reset_index(drop=True), V.reset_index(drop=True)], axis=1)
    assert (X[config.ID_COL].to_numpy() == meta[config.ID_COL].to_numpy()).all(), \
        "feature and metadata row order disagree"
    assert np.all(np.diff(meta[config.TIME_COL].to_numpy()) >= 0), \
        "metadata is not sorted by TransactionDT"
    return X, meta


def run_experiment(model_factory: Callable[[], FraudModel],
                   X: pd.DataFrame, meta: pd.DataFrame,
                   view: str = "base",
                   name: str | None = None,
                   n_folds: int = config.N_TIME_FOLDS) -> ExperimentResult:
    """Evaluate one (model, feature view) pair on the shared protocol."""
    cols = view_columns(X, view)
    y = meta[config.TARGET].to_numpy()
    name = name or f"{model_factory().name}+{view}"

    sp = chronological_split(meta)
    folds = expanding_window_folds(meta, n_folds=n_folds, holdout_idx=sp.test_idx)
    assert_no_time_overlap(meta, folds)

    rows = []
    for k, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        model = model_factory()
        # The validation block doubles as the early-stopping set. It is the
        # chronologically next block, so "stop when the next month stops
        # improving" is exactly the deployment question.
        model.fit(X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va])
        scores = model.predict_proba(X.iloc[va][cols])
        m = evaluate_scores(y[va], scores)
        m.update({"fold": k, "model": name, "fit_seconds": time.perf_counter() - t0,
                  "n_train": len(tr)})
        rows.append(m)
        log.info("%-24s fold %d  PR-AUC %.4f (base %.4f, lift %.1fx)  ROC %.4f  %.0fs",
                 name, k, m["pr_auc"], m["base_rate"], m["pr_auc_lift"],
                 m["roc_auc"], m["fit_seconds"])
        # Each fold's fitted model holds its own imputer, scaler and encoders,
        # and the fold slices are copies. Without an explicit drop they stay
        # alive until the next rebind, so peak memory is two folds rather than
        # one -- enough to OOM on the widest feature view.
        del model, scores
        gc.collect()

    per_fold = pd.DataFrame(rows)

    # ---- final holdout ---------------------------------------------------
    # The three-way chronological split exists for exactly this: fit on the
    # earliest 65%, early-stop on the next 15%, score the final 20% once.
    #
    # The tempting alternative -- fit on the whole 80% and early-stop on its
    # last block -- is wrong, because that block is then inside the training
    # set: the stopping criterion would be measured on rows the model has
    # already fitted, so it never triggers and the "best iteration" it reports
    # is meaningless. Giving up 15% of the training rows is the price of an
    # honest stopping rule, and it keeps the holdout untouched by every
    # decision taken anywhere in the project.
    final = model_factory()
    t0 = time.perf_counter()
    final.fit(X.iloc[sp.train_idx][cols], y[sp.train_idx],
              X.iloc[sp.valid_idx][cols], y[sp.valid_idx])
    fit_seconds = time.perf_counter() - t0

    hs = final.predict_proba(X.iloc[sp.test_idx][cols])
    holdout = evaluate_scores(y[sp.test_idx], hs)
    holdout.update({"model": name, "view": view, "fit_seconds": fit_seconds,
                    "n_features": len(cols)})
    log.info("%-24s HOLDOUT PR-AUC %.4f (base %.4f, lift %.1fx) ROC %.4f",
             name, holdout["pr_auc"], holdout["base_rate"],
             holdout["pr_auc_lift"], holdout["roc_auc"])

    hist = final.history_frame() if hasattr(final, "history_frame") else None
    return ExperimentResult(
        name=name,
        per_fold=per_fold,
        holdout=holdout,
        holdout_scores=hs,
        fold_summary=fold_summary(meta, folds),
        calibration=calibration_table(y[sp.test_idx], hs),
        importance=final.feature_importance(),
        history=hist,
    )


def results_table(results: list[ExperimentResult]) -> pd.DataFrame:
    """Holdout + cross-fold summary, one row per experiment."""
    rows = []
    for r in results:
        pf = r.per_fold
        rows.append({
            "model": r.name,
            "n_features": r.holdout["n_features"],
            "cv_pr_auc_mean": pf["pr_auc"].mean(),
            "cv_pr_auc_std": pf["pr_auc"].std(ddof=0),
            "cv_roc_auc_mean": pf["roc_auc"].mean(),
            "holdout_pr_auc": r.holdout["pr_auc"],
            "holdout_pr_auc_lift": r.holdout["pr_auc_lift"],
            "holdout_roc_auc": r.holdout["roc_auc"],
            "holdout_precision@1%": r.holdout["precision@1%"],
            "holdout_recall@1%": r.holdout["recall@1%"],
            "holdout_f1@1%": r.holdout["f1@1%"],
            "holdout_f1_oracle": r.holdout["f1_oracle"],
            "holdout_brier": r.holdout["brier"],
            "fit_seconds": r.holdout["fit_seconds"],
        })
    return pd.DataFrame(rows)


def save_results(results: list[ExperimentResult], tag: str) -> None:
    """Persist everything. Called after *each* experiment, not only at the end,
    so a crash late in a two-hour grid does not discard the runs that already
    finished."""
    table = results_table(results)
    table.to_csv(config.RESULTS_DIR / f"{tag}_summary.csv", index=False)
    pd.concat([r.per_fold for r in results]).to_csv(
        config.RESULTS_DIR / f"{tag}_per_fold.csv", index=False)
    for r in results:
        slug = r.name.replace("+", "_").replace(" ", "_")
        r.calibration.to_csv(config.RESULTS_DIR / f"{tag}_calibration_{slug}.csv")
        if r.importance is not None:
            r.importance.head(50).to_csv(
                config.RESULTS_DIR / f"{tag}_importance_{slug}.csv")
        if r.history is not None and len(r.history):
            r.history.to_csv(config.RESULTS_DIR / f"{tag}_history_{slug}.csv",
                             index=False)
    (config.RESULTS_DIR / f"{tag}_holdout.json").write_text(
        json.dumps([r.holdout for r in results], indent=2, default=float))
