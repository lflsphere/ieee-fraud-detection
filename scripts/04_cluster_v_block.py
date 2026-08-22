"""Phase 4 driver: fit the V-block compressor, test stability, export features.

    PYTHONPATH=. python scripts/04_cluster_v_block.py

Writes:
  data/interim/train_vfeatures.parquet   (untracked)  PCA components + cluster
  reports/figures/04_cluster_stability.png
  reports/figures/04_cluster_fraud_alignment.png
  reports/results/04_cluster_stability.csv
  reports/results/04_cluster_alignment.csv
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data.split import chronological_split
from src.unsupervised.cluster_v_features import (
    VBlockCompressor,
    cluster_label_alignment,
    cluster_stability,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

K_GRID = [4, 6, 8, 10, 12, 16]
SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    df = pd.read_parquet(config.INTERIM_DIR / "train_joined.parquet")
    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)

    sp = chronological_split(df)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[np.arange(len(sp.train_idx))] = True   # df is already time-sorted
    y = df[config.TARGET].to_numpy()

    # k=6 is not a default: it is the outcome of the stability sweep printed
    # below. See the module docstring for why k=8 was rejected.
    comp = VBlockCompressor(n_clusters=6).fit(df, train_mask)
    V = comp.transform(df)
    n_pcs = len(comp.component_names_)
    logging.info("%d V columns -> %d PCA components + 3 cluster features",
                 sum(len(g) for g in comp.groups_), n_pcs)

    Z = V[comp.component_names_].to_numpy()

    # --- stability: fitted on TRAINING rows only ---------------------------
    stab = cluster_stability(Z[train_mask], K_GRID, SEEDS)
    stab.to_csv(config.RESULTS_DIR / "04_cluster_stability.csv", index=False)
    print("\n=== cluster stability (training block only) ===")
    print(stab.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].plot(stab["k"], stab["mean_ari"], marker="o", label="mean pairwise ARI")
    axes[0].plot(stab["k"], stab["min_ari"], marker="s", ls="--", label="worst pair")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("adjusted Rand index")
    axes[0].set_ylim(0, 1.02); axes[0].legend()
    axes[0].set_title("Seed-to-seed reproducibility")

    axes[1].plot(stab["k"], stab["silhouette"], marker="o", color="#55A868")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette")
    axes[1].set_title("Cluster separation")

    axes[2].plot(stab["k"], stab["inertia"], marker="o", color="#C44E52")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("inertia")
    axes[2].set_title("Elbow")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_cluster_stability.png", bbox_inches="tight")

    # --- post-hoc alignment with the label ---------------------------------
    align = cluster_label_alignment(V["v_cluster"].to_numpy(), y)
    align.to_csv(config.RESULTS_DIR / "04_cluster_alignment.csv")
    print("\n=== cluster vs fraud rate (post-hoc; label never used to fit) ===")
    print(align.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(align.index.astype(str), align["fraud_rate"], color="#C44E52")
    axes[0].axhline(y.mean(), ls="--", c="grey", label=f"base {y.mean():.2%}")
    axes[0].set_xlabel("v_cluster"); axes[0].set_ylabel("fraud rate")
    axes[0].set_title("Fraud rate by unsupervised cluster"); axes[0].legend()

    axes[1].bar(align.index.astype(str), align["n"], color="#4C72B0")
    axes[1].set_xlabel("v_cluster"); axes[1].set_ylabel("transactions")
    axes[1].set_title("Cluster sizes")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_cluster_fraud_alignment.png", bbox_inches="tight")

    V.to_parquet(config.INTERIM_DIR / "train_vfeatures.parquet", index=False)
    logging.info("wrote train_vfeatures.parquet %s", V.shape)


if __name__ == "__main__":
    main()
