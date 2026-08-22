"""Render result CSVs as markdown tables for the Phase 7/9 write-ups.

Reports quote a lot of numbers. Transcribing them by hand is how a write-up
ends up disagreeing with the code that produced it, so every table in
reports/final/ that carries measured values is generated from the CSVs in
reports/results/ through this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}",
                index: bool = False) -> str:
    """Minimal markdown table renderer (no tabulate dependency)."""
    d = df.reset_index() if index else df.copy()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "—"
            return floatfmt.format(v)
        return str(v)

    cols = list(d.columns)
    rows = [[fmt(v) for v in rec] for rec in d.itertuples(index=False)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) if rows else len(str(c))
              for i, c in enumerate(cols)]
    # pandas 3 gives text columns the ``str`` dtype rather than ``object``,
    # so test for numeric instead of testing for object.
    align = ["---:" if pd.api.types.is_numeric_dtype(d[c]) else "---"
             for c in cols]

    out = ["| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |",
           "|" + "|".join(a for a in align) + "|"]
    out += ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join(out)


def model_comparison(summary: pd.DataFrame) -> str:
    """The Phase 9 headline table."""
    d = summary.copy()
    d = d[["model", "n_features", "cv_pr_auc_mean", "cv_pr_auc_std",
           "holdout_pr_auc", "holdout_pr_auc_lift", "holdout_roc_auc",
           "holdout_precision@1%", "holdout_recall@1%", "holdout_f1@1%",
           "fit_seconds"]]
    d.columns = ["model", "features", "CV PR-AUC", "CV sd", "holdout PR-AUC",
                 "lift", "holdout ROC-AUC", "prec@1%", "recall@1%", "F1@1%",
                 "fit s"]
    return to_markdown(d.sort_values("holdout PR-AUC", ascending=False))


def per_fold_pivot(per_fold: pd.DataFrame, metric: str = "pr_auc") -> str:
    p = per_fold.pivot(index="model", columns="fold", values=metric)
    p.columns = [f"fold {c}" for c in p.columns]
    p["mean"] = p.mean(axis=1)
    p["sd"] = p.std(axis=1, ddof=0)
    return to_markdown(p.reset_index())
