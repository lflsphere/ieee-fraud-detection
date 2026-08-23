"""Phase 6 driver: neural-network training dynamics and sensitivity.

    PYTHONPATH=. python scripts/06_nn_training_dynamics.py

Runs on the largest expanding-window fold (train = blocks 0..4, validate =
block 5) so the curves reflect the amount of data the deployed model would
actually see. Early stopping is switched **off** for these runs (patience set
beyond max_epochs) — the point is to observe the divergence, not to avoid it.

Three questions:
  1. Where do train and validation curves diverge, and does validation *loss*
     diverge at the same point as validation *ranking* (PR-AUC)?
  2. What do dropout and weight decay actually change?
  3. How sensitive is the result to learning rate and hidden width?
"""
from __future__ import annotations

import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data.split import chronological_split, expanding_window_folds
from src.evaluation.evaluate import load_matrices
from src.models.base import view_columns
from src.models.nn import NeuralNet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

MAX_EPOCHS = 30

REGULARISATION = [
    ("no regularisation",   dict(dropout=0.0, weight_decay=0.0)),
    ("dropout 0.3 (used)",  dict(dropout=0.3, weight_decay=1e-5)),
    ("dropout 0.5 + wd 1e-3", dict(dropout=0.5, weight_decay=1e-3)),
]
LEARNING_RATES = [1e-4, 1e-3, 3e-3]
WIDTHS = [(64, 32), (256, 128), (512, 256)]


def run(tag: str, X_tr, y_tr, X_va, y_va, **kw) -> pd.DataFrame:
    model = NeuralNet(max_epochs=MAX_EPOCHS, patience=MAX_EPOCHS + 1, **kw)
    model.fit(X_tr, y_tr, X_va, y_va)
    h = model.history_frame()
    h["config"] = tag
    h["best_epoch"] = model.best_epoch_
    logging.info("%-26s best epoch %2d  best val PR-AUC %.4f  final %.4f",
                 tag, model.best_epoch_, h["valid_pr_auc"].max(),
                 h["valid_pr_auc"].iloc[-1])
    return h


def main() -> None:
    X, meta = load_matrices()
    y = meta[config.TARGET].to_numpy()
    sp = chronological_split(meta)
    folds = expanding_window_folds(meta, holdout_idx=sp.test_idx)
    tr, va = folds[-1]
    cols = view_columns(X, "base")
    X_tr, X_va = X.iloc[tr][cols], X.iloc[va][cols]
    y_tr, y_va = y[tr], y[va]
    logging.info("fold: train %d (%.2f%% fraud) / valid %d (%.2f%% fraud)",
                 len(tr), 100 * y_tr.mean(), len(va), 100 * y_va.mean())

    hists = []
    for tag, kw in REGULARISATION:
        hists.append(run(tag, X_tr, y_tr, X_va, y_va, **kw))
    for lr in LEARNING_RATES:
        hists.append(run(f"lr={lr:g}", X_tr, y_tr, X_va, y_va, lr=lr))
    for w in WIDTHS:
        hists.append(run(f"width={w}", X_tr, y_tr, X_va, y_va, hidden=w))

    hist = pd.concat(hists, ignore_index=True)
    hist.to_csv(config.RESULTS_DIR / "06_nn_history.csv", index=False)

    summary = (hist.groupby("config")
               .agg(best_epoch=("best_epoch", "first"),
                    best_valid_pr_auc=("valid_pr_auc", "max"),
                    final_valid_pr_auc=("valid_pr_auc", "last"),
                    final_train_loss=("train_loss", "last"),
                    final_valid_loss=("valid_loss", "last"),
                    min_valid_loss=("valid_loss", "min"))
               .reset_index())
    summary["pr_auc_decay_after_best"] = (
        summary["best_valid_pr_auc"] - summary["final_valid_pr_auc"])
    summary.to_csv(config.RESULTS_DIR / "06_nn_sensitivity.csv", index=False)
    print("\n=== NN sensitivity ===")
    print(summary.round(4).to_string(index=False))

    # --- figures ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    base = hist[hist["config"] == "dropout 0.3 (used)"]
    axes[0].plot(base["epoch"], base["train_loss"], marker="o", label="train loss")
    axes[0].plot(base["epoch"], base["valid_loss"], marker="s", label="valid loss")
    b = int(base["best_epoch"].iloc[0])
    axes[0].axvline(b, ls="--", c="k", label=f"best PR-AUC @ epoch {b}")
    axes[0].axvline(int(base["valid_loss"].idxmin() - base.index[0]), ls=":", c="grey",
                    label="min valid loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("class-weighted BCE")
    axes[0].set_title("Loss curves (default config)"); axes[0].legend(fontsize=8)

    for tag, _ in REGULARISATION:
        g = hist[hist["config"] == tag]
        axes[1].plot(g["epoch"], g["valid_pr_auc"], marker="o", label=tag)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation PR-AUC")
    axes[1].set_title("Effect of regularisation"); axes[1].legend(fontsize=8)

    for lr in LEARNING_RATES:
        g = hist[hist["config"] == f"lr={lr:g}"]
        axes[2].plot(g["epoch"], g["valid_pr_auc"], marker="o", label=f"lr={lr:g}")
    for w in WIDTHS:
        g = hist[hist["config"] == f"width={w}"]
        axes[2].plot(g["epoch"], g["valid_pr_auc"], ls="--", label=f"width={w}")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("validation PR-AUC")
    axes[2].set_title("Learning rate and width sensitivity"); axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "06_nn_training_dynamics.png", bbox_inches="tight")

    # train-vs-valid gap for the three regularisation settings
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for tag, _ in REGULARISATION:
        g = hist[hist["config"] == tag]
        ax.plot(g["epoch"], g["valid_loss"] - g["train_loss"], marker="o", label=tag)
    ax.axhline(0, c="grey", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("valid loss - train loss")
    ax.set_title("Generalisation gap"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "06_nn_generalisation_gap.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
