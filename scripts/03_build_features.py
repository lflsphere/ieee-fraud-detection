"""Phase 3 driver: build and cache the training feature matrix.

    PYTHONPATH=. python scripts/03_build_features.py

Writes (all untracked, under data/interim/):
  train_features.parquet  — 508 model features, rows sorted by TransactionDT
  train_meta.parquet      — TransactionID / TransactionDT / isFraud in the same
                            row order, so no later phase has to re-derive it
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src import config
from src.data.split import chronological_split
from src.features.build_features import build_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    df = pd.read_parquet(config.INTERIM_DIR / "train_joined.parquet")
    sp = chronological_split(df)

    # Statistics are fitted on the earliest block only. Validation and the
    # final holdout are transformed with those frozen statistics, which is what
    # makes the split a faithful simulation of "fit now, score later".
    mask = np.zeros(len(df), dtype=bool)
    mask[sp.train_idx] = True

    t0 = time.time()
    X, fb = build_feature_matrix(df, mask)
    logging.info("built %s in %.1fs", X.shape, time.time() - t0)

    order = np.argsort(df[config.TIME_COL].to_numpy(), kind="stable")
    meta = (df.iloc[order][[config.ID_COL, config.TIME_COL, config.TARGET]]
              .reset_index(drop=True))

    X.to_parquet(config.INTERIM_DIR / "train_features.parquet", index=False)
    meta.to_parquet(config.INTERIM_DIR / "train_meta.parquet", index=False)
    logging.info("wrote train_features.parquet and train_meta.parquet")


if __name__ == "__main__":
    main()
