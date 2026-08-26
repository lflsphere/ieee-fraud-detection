"""Shared model interface and feature views.

One interface (`FraudModel`) so that Phase 7 can evaluate the linear baseline,
the GBM and the neural network through identical code on identical folds. If
each model brought its own evaluation loop, any difference in the results table
could be an artefact of the loop rather than of the model — which would make
the Phase 9 comparison worthless.

Feature *views* live here too, because "which columns does this model see" is
part of the experimental design, not an implementation detail of the model.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src import config

# Columns that are categorical *codes* rather than quantities. LightGBM is told
# about them explicitly; the dense models one-hot or embed them.
CATEGORICAL_FEATURES = ([f"{c}_code" for c in config.CATEGORICAL_COLS]
                        + ["amt_decile", "dt_hour_assumed", "dt_dow_assumed",
                           "v_cluster"])


def view_columns(X: pd.DataFrame, view: str) -> list[str]:
    """Column list for a named feature view.

    * ``base``    — the 508 Phase-3 features. The reference representation.
    * ``unsup``   — ``base`` plus the Phase-4 PCA components and the three
                    cluster features. Tests whether the unsupervised
                    representation *adds* anything.
    * ``compact`` — Phase-3 features with the 339 raw ``V*`` columns
                    **replaced** by their PCA components, plus the cluster
                    features. Tests whether the compression can *substitute*
                    for the raw block — the case that matters for the linear
                    and neural models, which are the ones that suffer from
                    collinearity.
    """
    cols = [c for c in X.columns if c != config.ID_COL]
    v_raw = [c for c in cols if c.startswith("V") and c[1:].isdigit()]
    v_pca = [c for c in cols if c.startswith("vpca_")]
    v_clu = [c for c in cols if c.startswith("v_cluster")]

    if view == "base":
        return [c for c in cols if c not in set(v_pca) | set(v_clu)]
    if view == "unsup":
        return cols
    if view == "compact":
        return [c for c in cols if c not in set(v_raw)]
    raise ValueError(f"unknown view {view!r}")


class FraudModel(ABC):
    """Minimal contract every Phase 5 model implements."""

    name: str = "model"
    #: Set by :meth:`fit`; seconds of wall-clock training time, reported in the
    #: Phase 9 cost column.
    fit_seconds_: float = 0.0

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_valid: pd.DataFrame | None = None,
            y_valid: np.ndarray | None = None) -> "FraudModel":
        """Fit on one fold.

        ``X_valid`` is offered for early stopping only. Implementations must
        not use it to select features or to fit any preprocessing statistic —
        that would leak the validation fold into the model.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) for each row, as a 1-D array."""

    def timed_fit(self, *args, **kwargs) -> "FraudModel":
        t0 = time.perf_counter()
        self.fit(*args, **kwargs)
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def feature_importance(self) -> pd.Series | None:
        """Optional; used by Phase 9's interpretability discussion."""
        return None
