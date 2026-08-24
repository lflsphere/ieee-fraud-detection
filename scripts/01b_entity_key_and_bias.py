"""Two Phase-1 claims that were asserted and are now measured.

    PYTHONPATH=. python scripts/01b_entity_key_and_bias.py

A. WHY THE ENTITY KEY EXCLUDES card4 / card6 / addr2.
   Reports each column's cardinality and, decisively, how many *additional*
   proxy entities it creates when appended to the base key, plus how often it
   is functionally determined by that key. A column that is determined by the
   key carries no entity-resolution information, and appending a near-constant
   column can only split entities - never merge them - so any split it does
   create is as likely to be a recording inconsistency fragmenting one real
   account as a genuine distinction.

B. HOW SELECTION AND FEEDBACK BIAS LAND ON THE REVIEW BUDGET.
   Uses the saved holdout scores to show where the 1% review queue actually
   falls relative to `has_identity_record`, separately for true positives,
   false positives and missed fraud. Requires data/interim/holdout_scores.parquet
   (written by the GBM holdout refit); skips section B if absent.

Output: reports/results/01_entity_key.csv, reports/results/01_review_bias.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.build_features import _concat_key

BASE_KEY = ["card1", "card2", "card3", "card5", "addr1"]
EXCLUDED = ["card4", "card6", "addr2"]


def section_a() -> pd.DataFrame:
    df = pd.read_parquet(config.INTERIM_DIR / "train_joined.parquet",
                         columns=BASE_KEY + EXCLUDED)
    base = _concat_key(df, BASE_KEY)
    n0 = base.nunique()
    print(f"base key {'|'.join(BASE_KEY)} -> {n0:,} proxy entities\n")

    rows = []
    for c in BASE_KEY + EXCLUDED:
        in_key = c in BASE_KEY
        if in_key:
            extra = np.nan
            single = np.nan
        else:
            extra = _concat_key(df, BASE_KEY + [c]).nunique() - n0
            # How often is this column constant within a base-key entity? If
            # always, it is a deterministic function of the key.
            nu = pd.DataFrame({"k": base, "v": df[c].astype("string").fillna("NA")}) \
                   .groupby("k")["v"].nunique()
            single = float((nu <= 1).mean())
        rows.append({
            "column": c, "in_entity_key": in_key,
            "n_levels": int(df[c].nunique()),
            "missing_rate": float(df[c].isna().mean()),
            "largest_level_share": float(df[c].value_counts(normalize=True).iloc[0]),
            "extra_entities_if_added": extra,
            "pct_gain_if_added": (extra / n0) if extra == extra else np.nan,
            "frac_entities_single_valued": single,
        })
    t = pd.DataFrame(rows)
    t.to_csv(config.RESULTS_DIR / "01_entity_key.csv", index=False)
    print(t.round(4).to_string(index=False))
    return t


def section_b() -> pd.DataFrame | None:
    path = config.INTERIM_DIR / "holdout_scores.parquet"
    if not path.exists():
        print("\n[section B skipped: holdout_scores.parquet not present]")
        return None
    d = pd.read_parquet(path)
    thr = np.quantile(d["score"], 1.0 - config.REVIEW_BUDGET_FRACTION)
    q = d[d["score"] >= thr]
    tp, fp = q[q.y == 1], q[q.y == 0]
    fn = d[(d.y == 1) & (d.score < thr)]

    rows = [
        {"population": "holdout (all)", "n": len(d),
         "share_with_identity": d.has_identity_record.mean()},
        {"population": "1% review queue", "n": len(q),
         "share_with_identity": q.has_identity_record.mean()},
        {"population": "queue: true positives", "n": len(tp),
         "share_with_identity": tp.has_identity_record.mean()},
        {"population": "queue: FALSE positives", "n": len(fp),
         "share_with_identity": fp.has_identity_record.mean()},
        {"population": "missed fraud (false negatives)", "n": len(fn),
         "share_with_identity": fn.has_identity_record.mean()},
    ]
    for seg, lab in [(1, "segment: identity present"), (0, "segment: identity absent")]:
        s = d[d.has_identity_record == seg]
        sq = s[s.score >= thr]
        rows.append({"population": lab, "n": len(s),
                     "share_with_identity": float(seg),
                     "fraud_rate": s.y.mean(),
                     "share_of_queue": len(sq) / len(q),
                     "recall_within_segment": sq.y.sum() / max(s.y.sum(), 1)})

    t = pd.DataFrame(rows)
    t.to_csv(config.RESULTS_DIR / "01_review_bias.csv", index=False)
    print(f"\nprecision@1% {q.y.mean():.4f}   recall@1% {q.y.sum()/d.y.sum():.4f}\n")
    print(t.round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    section_a()
    section_b()
