"""Phase 5 — logistic-regression baseline.

Why a linear model at all, when we expect it to lose
----------------------------------------------------
Three reasons, all of which survive it losing:

1. **It is the interpretability floor.** A coefficient per feature is something
   a fraud analyst or a model-risk reviewer can read. If the GBM only beats it
   by a small margin, the linear model may still be the right deployment.
2. **It is the leakage canary.** A linear model cannot memorise an entity the
   way a deep tree ensemble can. If a feature is leaking, the linear model's
   score jumps too — and a suspiciously good *linear* model on a 3.5%-fraud
   problem is a much louder alarm than a suspiciously good GBM.
3. **It localises where nonlinearity matters.** Phase 2 found the fraud-rate
   response to amount is U-shaped and the response to assumed hour is
   non-monotonic. Comparing the linear model with and without the binned
   versions of those features measures how much of the gap to the GBM is
   *nonlinearity* rather than *interactions*.

Preprocessing is inside the model's own pipeline on purpose: the imputer's
medians and the scaler's means must be fitted on the training fold only. A
scaler fitted before the split is the most common preprocessing leak there is,
and putting it in a `Pipeline` makes that structurally impossible here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src import config
from src.models.base import CATEGORICAL_FEATURES, FraudModel

# One-hot only the genuinely low-cardinality categoricals. `card1_code` has
# 13,553 levels; one-hot would add 13,553 mostly-empty columns and the
# coefficients would be pure variance. High-cardinality levels reach the linear
# model through their Phase-3 frequency and target encodings instead, which is
# the whole reason those encodings exist.
MAX_ONEHOT_CARDINALITY = 25


def _as_float32(X):
    """Cast to float32 between preprocessing steps.

    Not cosmetic: ``SimpleImputer`` and ``StandardScaler`` both emit float64,
    and on the widest feature view that is a 393,694 x 637 array at 8 bytes per
    cell -- 2.0 GB, several times over as the pipeline hands arrays between
    steps. The first full run peaked at 13.2 GB of the box's 16 GB and would
    have been OOM-killed on the largest fold. float32 keeps ~7 significant
    digits, far more than these standardised features carry.
    """
    A = np.asarray(X, dtype=np.float32)
    # Infinities become NaN here rather than via a DataFrame-level
    # ``.replace([inf, -inf], nan)``, which would copy the whole fold slice a
    # second time (~1 GB on the widest view) before the cast even starts.
    # This mutates the array the cast just produced, so it costs nothing.
    A[~np.isfinite(A)] = np.nan
    return A


class LinearBaseline(FraudModel):
    name = "linear"

    def __init__(self, C: float = 0.1, max_iter: int = 300,
                 seed: int = config.RANDOM_SEED):
        # C=0.1 (i.e. fairly strong L2) rather than the sklearn default of 1.0:
        # the feature matrix contains 339 heavily collinear V columns, and
        # without shrinkage the coefficients on near-duplicate columns explode
        # in opposite directions. This is the bias/variance trade made
        # explicitly rather than inherited from a default.
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.pipe_: Pipeline | None = None
        self.numeric_: list[str] = []
        self.onehot_: list[str] = []

    def _split_columns(self, X: pd.DataFrame) -> None:
        cat_present = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        self.onehot_ = [c for c in cat_present
                        if X[c].nunique(dropna=False) <= MAX_ONEHOT_CARDINALITY]
        self.numeric_ = [c for c in X.columns if c not in set(self.onehot_)]

    def fit(self, X, y, X_valid=None, y_valid=None):
        self._split_columns(X)
        pre = ColumnTransformer([
            # Median, not mean: many C*/V* columns are counts with long tails.
            # The float32 cast comes FIRST, not last. SimpleImputer and
            # StandardScaler each emit a full-size array and promote float32
            # input only if it arrives as float64, so casting afterwards means
            # both float64 intermediates have already been allocated -- the
            # measured peak was 11.2 GB even with a trailing cast. Casting
            # first keeps the whole chain in float32.
            ("num", Pipeline([("f32", FunctionTransformer(_as_float32,
                                                          feature_names_out="one-to-one")),
                              ("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), self.numeric_),
            ("cat", Pipeline([("onehot", OneHotEncoder(
                handle_unknown="ignore", min_frequency=50,
                sparse_output=False, dtype=np.float32))]), self.onehot_),
        ], sparse_threshold=0.0)

        clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, solver="lbfgs",
            # Re-weighting rather than resampling: it keeps every negative in
            # the fit (resampling throws away 96.5% of the data) and it leaves
            # the row ordering untouched, which matters because our folds are
            # chronological.
            class_weight="balanced", random_state=self.seed)

        self.pipe_ = Pipeline([("pre", pre), ("clf", clf)])
        self.pipe_.fit(X, y)          # inf -> NaN happens inside _as_float32
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(X)[:, 1]

    def feature_importance(self):
        """|coefficient| on standardised features — comparable across columns."""
        if self.pipe_ is None:
            return None
        names = self.pipe_.named_steps["pre"].get_feature_names_out()
        coefs = self.pipe_.named_steps["clf"].coef_.ravel()
        return pd.Series(np.abs(coefs), index=names).sort_values(ascending=False)
