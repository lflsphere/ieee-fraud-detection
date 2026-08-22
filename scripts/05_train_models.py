"""Phases 5 + 7 driver: train every (model, feature view) pair on shared folds.

    PYTHONPATH=. python scripts/05_train_models.py

All nine experiments run through `src.evaluation.evaluate.run_experiment`, so
they see byte-identical folds and identical metrics. Results land in
reports/results/05_* and the figures used by Phases 7 and 9 in reports/figures/.
"""
from __future__ import annotations

import gc
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluation.evaluate import (
    load_matrices,
    results_table,
    run_experiment,
    save_results,
)
from src.models.gbm import GBMModel
from src.models.linear import LinearBaseline
from src.models.nn import NeuralNet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

# (factory, view, display name). The three views answer three different
# questions about the Phase-4 representation - see reports/final/04_unsupervised.md.
EXPERIMENTS = [
    (lambda: LinearBaseline(), "base", "linear"),
    (lambda: LinearBaseline(), "unsup", "linear+unsup"),
    (lambda: LinearBaseline(), "compact", "linear+compact"),
    (lambda: GBMModel(), "base", "gbm"),
    (lambda: GBMModel(), "unsup", "gbm+unsup"),
    (lambda: GBMModel(), "compact", "gbm+compact"),
    (lambda: NeuralNet(), "base", "nn"),
    (lambda: NeuralNet(), "unsup", "nn+unsup"),
    (lambda: NeuralNet(), "compact", "nn+compact"),
]


def main() -> None:
    X, meta = load_matrices()
    logging.info("feature matrix %s", X.shape)

    results = []
    for factory, view, name in EXPERIMENTS:
        logging.info("=== %s (view=%s) ===", name, view)
        results.append(run_experiment(factory, X, meta, view=view, name=name))
        # Checkpoint after every experiment: the full grid is a ~2 hour run and
        # losing eight finished experiments to a failure in the ninth is not a
        # trade worth making.
        save_results(results, "05")
        gc.collect()

    save_results(results, "05")
    table = results_table(results)
    print("\n=== Phase 5/7 results ===")
    print(table.round(4).to_string(index=False))

    # --- figures ----------------------------------------------------------
    per_fold = pd.concat([r.per_fold for r in results])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for name, g in per_fold.groupby("model", sort=False):
        axes[0].plot(g["fold"], g["pr_auc"], marker="o", label=name)
    axes[0].set_xlabel("expanding-window fold"); axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("PR-AUC per time fold"); axes[0].legend(fontsize=7)

    base = per_fold.groupby("fold")["base_rate"].first()
    axes[1].bar(base.index, base.values, color="#999999")
    axes[1].set_xlabel("fold"); axes[1].set_ylabel("fraud rate in validation block")
    axes[1].set_title("Class prior drifts across folds")

    t = table.sort_values("holdout_pr_auc")
    axes[2].barh(t["model"], t["holdout_pr_auc"], color="#C44E52")
    axes[2].set_xlabel("holdout PR-AUC"); axes[2].set_title("Final chronological holdout")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_model_comparison.png", bbox_inches="tight")

    # calibration on the holdout
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for r in results:
        if r.name in ("linear", "gbm", "nn"):
            c = r.calibration
            ax.plot(c["mean_pred"], c["observed"], marker="o", label=r.name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", c="grey", label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraud rate")
    ax.set_title("Calibration on the chronological holdout"); ax.legend()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_calibration.png", bbox_inches="tight")

    # GBM gain importance
    gbm = next(r for r in results if r.name == "gbm")
    if gbm.importance is not None:
        top = gbm.importance.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top.index, top.values, color="#4C72B0")
        ax.set_xlabel("LightGBM gain"); ax.set_title("Top 25 features (GBM, base view)")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "07_gbm_importance.png", bbox_inches="tight")

    print("\n=== fold structure ===")
    print(results[0].fold_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
