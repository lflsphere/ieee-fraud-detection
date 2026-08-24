"""Is a column's missingness explained by the identity join, or independent of it?

    PYTHONPATH=. python scripts/03b_missingness_vs_join.py

Phase 3 keeps 21 `*_isnull` indicators rather than all 171 informative ones,
on the grounds that most of the 171 merely restate `has_identity_record`. That
grounds needs testing rather than asserting, and this script is the test.

For each candidate column it reports the missing rate conditional on whether
the transaction matched an identity row, and the phi coefficient between the
is-null indicator and "has no identity record":

  phi ~ +1  missing exactly when the join fails  -> redundant with
            has_identity_record, should not be kept
  phi ~  0  missingness independent of the join  -> genuinely new signal
  phi ~ -1  missing exactly when the join *succeeds* -> also independent, and
            evidence of a second, disjoint collection process

Output: reports/results/03_missingness_vs_join.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.load import RAW_FILES
from src.features.build_features import ISNULL_COLS

# A few identity-block columns are included as positive controls: they *must*
# come out near phi = 1, or the diagnostic is not measuring what it claims to.
CONTROLS = ["id_12", "id_02", "id_31", "DeviceType"]


def main() -> None:
    cols = list(dict.fromkeys(ISNULL_COLS + CONTROLS))
    df = pd.read_parquet(
        config.INTERIM_DIR / "train_joined.parquet",
        columns=cols + ["has_identity_record", config.TARGET])
    has_id = df["has_identity_record"].to_numpy().astype(bool)
    no_id = (~has_id).astype(float)

    txn_cols = set(pd.read_csv(config.RAW_DIR / RAW_FILES["train_transaction"],
                               nrows=1).columns)

    rows = []
    for c in cols:
        m = df[c].isna().to_numpy()
        rows.append({
            "col": c,
            "source_file": "transaction" if c in txn_cols else "identity",
            "miss_overall": m.mean(),
            "miss_given_no_identity": m[~has_id].mean(),
            "miss_given_identity": m[has_id].mean(),
            "phi_vs_no_identity": float(np.corrcoef(m.astype(float), no_id)[0, 1]),
        })

    t = pd.DataFrame(rows)
    # Bucket on SIGNED phi, not |phi|. A strongly negative phi means the column
    # is missing precisely when the join *succeeds* - that is independent
    # signal from a second collection process, the opposite of redundant.
    # Bucketing on |phi| would file it alongside the id_* block, which is
    # exactly backwards.
    t["verdict"] = pd.cut(
        t["phi_vs_no_identity"], [-1.01, -0.2, 0.2, 0.45, 0.75, 1.01],
        labels=["inverse (missing when join succeeds)", "independent",
                "weakly related", "partly explained by join",
                "largely redundant with join"])
    t = t.sort_values("phi_vs_no_identity", ascending=False)
    t.to_csv(config.RESULTS_DIR / "03_missingness_vs_join.csv", index=False)
    print(t.round(4).to_string(index=False))

    redundant = t[(t.phi_vs_no_identity > 0.75) & (t.source_file == "transaction")]
    if len(redundant):
        print("\nTransaction-file columns whose missingness is still largely a "
              "restatement of the join (should NOT be kept as separate "
              "indicators):")
        print("  " + ", ".join(redundant["col"]))


if __name__ == "__main__":
    main()
