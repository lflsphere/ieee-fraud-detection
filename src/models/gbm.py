"""Phase 5 — gradient-boosted trees (LightGBM).

Why LightGBM for this dataset specifically
------------------------------------------
* **Native categorical handling.** `card1_code` has 13,553 levels. LightGBM
  partitions levels directly by their gradient statistics instead of requiring
  a 13,553-column one-hot, which is the single biggest practical advantage over
  the linear and neural models here.
* **Native NaN handling.** A third of the feature matrix is missing, and the
  missingness is MNAR (Phase 2). LightGBM learns a default direction per split,
  so "value absent" becomes a decision rather than an imputation artefact.
* **Invariance to monotone rescaling.** The features span counts, seconds,
  dollars and anonymised scores. No scaler needed, so one fewer fitted
  statistic to leak.
* **Axis-aligned splits suit the U-shaped and non-monotonic responses** found
  in Phase 2 (amount deciles, assumed hour) without any manual binning.

Imbalance is handled with `scale_pos_weight` rather than resampling: resampling
would discard 96.5% of negatives or duplicate positives, and duplicating
positives inside a chronological fold effectively duplicates whole fraud bursts.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.models.base import CATEGORICAL_FEATURES, FraudModel


class GBMModel(FraudModel):
    name = "gbm"

    def __init__(self, n_estimators: int = 3000, learning_rate: float = 0.05,
                 num_leaves: int = 64, min_child_samples: int = 100,
                 subsample: float = 0.8, colsample_bytree: float = 0.6,
                 reg_lambda: float = 10.0, early_stopping_rounds: int = 100,
                 use_scale_pos_weight: bool = True,
                 seed: int = config.RANDOM_SEED):
        # n_estimators=3000 is a ceiling, not a target: early stopping on the
        # chronologically next block picks the actual count. At 2000 the cap
        # was binding (best iteration 1994), so the model was capacity-limited
        # rather than overfitting; 3000 gives it room and the realised
        # best_iteration is reported per fold so any remaining truncation is
        # visible.
        # num_leaves=64 with min_child_samples=100: deep enough to express
        # three- and four-way interactions among the C*/V* blocks, shallow
        # enough that a leaf still holds ~3 fraud cases at a 3.5% base rate.
        # colsample_bytree=0.6 is doing real work here — with 339 collinear V
        # columns, sampling features per tree decorrelates the ensemble.
        self.params = dict(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            subsample=subsample, subsample_freq=1,
            colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
            random_state=seed, n_jobs=-1, verbosity=-1,
            # Track ONLY average precision. Left at the default, LightGBM also
            # tracks binary_logloss, and lgb.early_stopping halts as soon as
            # *any* tracked metric stalls. Under scale_pos_weight the logloss
            # degrades from the first iterations (predictions are deliberately
            # pushed away from the base rate), so the default configuration
            # stopped training after 2 trees and cost ~0.2 PR-AUC.
            metric="average_precision")
        self.early_stopping_rounds = early_stopping_rounds
        self.use_scale_pos_weight = use_scale_pos_weight
        self.model_: lgb.LGBMClassifier | None = None
        self.cat_cols_: list[str] = []
        self.best_iteration_: int | None = None

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace([np.inf, -np.inf], np.nan).copy()
        for c in self.cat_cols_:
            X[c] = X[c].astype("category")
        return X

    def fit(self, X, y, X_valid=None, y_valid=None):
        self.cat_cols_ = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        params = dict(self.params)
        if self.use_scale_pos_weight:
            pos = float(np.sum(y == 1)); neg = float(np.sum(y == 0))
            params["scale_pos_weight"] = neg / max(pos, 1.0)

        self.model_ = lgb.LGBMClassifier(**params)
        Xf = self._prep(X)
        fit_kw = {"categorical_feature": self.cat_cols_}
        if X_valid is not None and y_valid is not None:
            # Early stopping on the *chronologically next* block, which is the
            # only honest way to pick an iteration count here: stopping on a
            # random subset of the training period would overstate the number
            # of rounds that generalise forward.
            fit_kw["eval_X"] = self._prep(X_valid)
            fit_kw["eval_y"] = y_valid
            fit_kw["eval_metric"] = "average_precision"
            fit_kw["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False,
                                   first_metric_only=True),
                lgb.log_evaluation(0)]
        self.model_.fit(Xf, y, **fit_kw)
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._prep(X))[:, 1]

    def feature_importance(self):
        if self.model_ is None:
            return None
        # "gain" rather than "split": split count rewards high-cardinality
        # columns for being splittable, gain rewards them for being useful.
        imp = self.model_.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.model_.feature_name_).sort_values(ascending=False)
