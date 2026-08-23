"""Phase 0 schema verification: does the data on disk match the plan?

Run:  python -m src.data.schema_check

The plan (IMPLEMENTATION_PLAN.md) states a schema up front.  This script
verifies it against the actual files rather than trusting it, and prints any
discrepancy so it can be recorded instead of silently worked around.
"""

from __future__ import annotations

import json

import pandas as pd

from src import config
from src.data.load import RAW_FILES, file_checksums, load_raw


def check() -> dict:
    report: dict = {"files": {}, "discrepancies": []}

    headers = {}
    for name in RAW_FILES:
        df = pd.read_csv(config.RAW_DIR / RAW_FILES[name], nrows=5)
        raw_cols = list(df.columns)
        headers[name] = raw_cols
        n_rows = sum(1 for _ in open(config.RAW_DIR / RAW_FILES[name], "rb")) - 1
        report["files"][name] = {"n_cols": len(raw_cols), "n_rows": n_rows}

    # --- discrepancy 1: hyphenated identity columns in the test file --------
    hyphen_train = [c for c in headers["train_identity"] if c.startswith("id-")]
    hyphen_test = [c for c in headers["test_identity"] if c.startswith("id-")]
    if hyphen_test and not hyphen_train:
        report["discrepancies"].append(
            "test_identity.csv uses hyphenated column names (id-01 ... id-38) "
            "while train_identity.csv uses underscores (id_01 ... id_38). "
            "Normalised in src.data.load.normalise_identity_columns."
        )

    # --- discrepancy 2: V-block width ---------------------------------------
    v_train = [c for c in headers["train_transaction"] if c.startswith("V")]
    if len(v_train) != len(config.V_COLS):
        report["discrepancies"].append(
            f"V-block has {len(v_train)} columns on disk, config declares "
            f"{len(config.V_COLS)}."
        )

    # --- declared categoricals all present ----------------------------------
    txn_cols = set(headers["train_transaction"])
    idt_cols = {c.replace("-", "_") for c in headers["train_identity"]}
    missing = [c for c in config.CATEGORICAL_COLS
               if c not in txn_cols and c not in idt_cols]
    if missing:
        report["discrepancies"].append(
            f"Declared categorical columns absent from the data: {missing}"
        )

    # --- target present only in train ---------------------------------------
    report["target_in_train"] = config.TARGET in txn_cols
    report["target_in_test"] = config.TARGET in set(headers["test_transaction"])

    # --- identity match rate (the join is a *left* join for a reason) -------
    tr_txn_ids = pd.read_csv(config.RAW_DIR / "train_transaction.csv",
                             usecols=[config.ID_COL])[config.ID_COL]
    tr_idt_ids = pd.read_csv(config.RAW_DIR / "train_identity.csv",
                             usecols=[config.ID_COL])[config.ID_COL]
    report["train_identity_match_rate"] = float(
        tr_txn_ids.isin(set(tr_idt_ids)).mean()
    )

    report["checksums_md5"] = file_checksums()
    return report


if __name__ == "__main__":
    rep = check()
    print(json.dumps(rep, indent=2))
    out = config.RESULTS_DIR / "00_schema_check.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwritten -> {out}")
