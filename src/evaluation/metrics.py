"""Phase 7 — metrics, chosen for a rare-event ranking problem.

Why this set, and why not the obvious ones
------------------------------------------
* **PR-AUC is primary.** With a 3.5% base rate, ROC-AUC is dominated by the
  97% of the curve that concerns ranking negatives against each other. Two
  models can differ by 0.01 ROC-AUC and by 0.10 PR-AUC. PR-AUC is also defined
  *relative to the prevailing base rate*, so when the fraud rate drifts from
  1.85% to 5.06% across folds (Phase 2), the metric moves with it — which is a
  feature, not a bug, because it stops us averaging away real drift. Every
  PR-AUC below is reported next to its fold's base rate and as a lift over it.
* **ROC-AUC is reported anyway**, because it is the competition metric and
  because it is base-rate invariant, which makes it the fairer number for
  comparing across folds with different priors. The two together say more than
  either alone.
* **F1 at a fixed review budget, not at 0.5.** A probability threshold of 0.5
  is meaningless for a class-weighted model on a 3.5% problem. What a fraud
  team actually has is a bounded review capacity, so we score the top
  ``REVIEW_BUDGET_FRACTION`` (1%) of transactions by predicted risk and report
  precision, recall and F1 there. That is a decision the business can take.
* **Brier score and a calibration curve**, reported but interpreted with care:
  the labels are the output of a prior detection policy with selection bias
  (Phase 1 §4), so "well calibrated" means calibrated to the historical
  labelling process, not to true fraud probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from src import config


def budget_threshold(scores: np.ndarray,
                     budget: float = config.REVIEW_BUDGET_FRACTION) -> float:
    """Score cut-off that flags exactly the top ``budget`` share of rows."""
    return float(np.quantile(scores, 1.0 - budget))


def metrics_at_budget(y: np.ndarray, scores: np.ndarray,
                      budget: float = config.REVIEW_BUDGET_FRACTION) -> dict:
    thr = budget_threshold(scores, budget)
    flagged = scores >= thr
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        f"precision@{budget:.0%}": precision,
        f"recall@{budget:.0%}": recall,
        f"f1@{budget:.0%}": f1,
        "n_flagged": int(flagged.sum()),
        "threshold": thr,
    }


def best_f1(y: np.ndarray, scores: np.ndarray) -> dict:
    """Best achievable F1 over all thresholds — an upper bound, not an operating point.

    Reported so that the budget-constrained F1 can be read against what the
    ranking could deliver if capacity were unlimited. Choosing this threshold
    on the evaluation set would itself be a mild form of selection on the test
    data, which is why it is labelled as an oracle bound and never used to set
    a deployed threshold.
    """
    p, r, t = precision_recall_curve(y, scores)
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(p), where=denom > 0)
    i = int(np.argmax(f1))
    return {"f1_oracle": float(f1[i]),
            "f1_oracle_threshold": float(t[i]) if i < len(t) else 1.0,
            "f1_oracle_precision": float(p[i]), "f1_oracle_recall": float(r[i])}


def evaluate_scores(y: np.ndarray, scores: np.ndarray,
                    budget: float = config.REVIEW_BUDGET_FRACTION) -> dict:
    """The full metric bundle for one (labels, scores) pair."""
    base = float(np.mean(y))
    pr_auc = float(average_precision_score(y, scores))
    out = {
        "n": int(len(y)),
        "base_rate": base,
        "pr_auc": pr_auc,
        # A random ranker scores PR-AUC == base rate, so the ratio is the
        # honest "how much better than nothing" number across drifting folds.
        "pr_auc_lift": pr_auc / base if base > 0 else np.nan,
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, np.clip(scores, 0, 1))),
    }
    out.update(metrics_at_budget(y, scores, budget))
    out.update(best_f1(y, scores))
    return out


def calibration_table(y: np.ndarray, scores: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed fraud rate in equal-count score bins.

    Equal-count (quantile) bins rather than equal-width: with a 3.5% base rate
    almost every row lands in the lowest equal-width bin and the table says
    nothing.
    """
    q = pd.qcut(pd.Series(scores), n_bins, duplicates="drop", labels=False)
    df = pd.DataFrame({"bin": q, "y": y, "p": scores})
    g = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"),
                              observed=("y", "mean"))
    g["gap"] = g["mean_pred"] - g["observed"]
    return g


def summarise_folds(per_fold: pd.DataFrame) -> pd.Series:
    """Mean and spread across folds.

    The spread matters as much as the mean: a model whose PR-AUC swings by
    0.10 across time folds is not usable at a fixed threshold even if its
    average looks good.
    """
    num = per_fold.select_dtypes(include=[np.number])
    out = {}
    for c in num.columns:
        out[f"{c}_mean"] = float(num[c].mean())
        out[f"{c}_std"] = float(num[c].std(ddof=0))
    return pd.Series(out)
