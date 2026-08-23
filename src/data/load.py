"""Raw-CSV loading, transaction<->identity join, and memory downcasting.

Design notes
------------
* The join is a **left join from ``transaction``**: only ~24% of training
  transactions have an identity row, so an inner join would silently discard
  three quarters of the data (and, worse, discard it *non-randomly* — see
  ``has_identity_record`` in ``src/features/build_features.py``).
* ``test_identity.csv`` ships with **hyphenated** column names (``id-01``)
  while ``train_identity.csv`` uses underscores (``id_01``).  This is a real
  quirk of the released files, not a typo on our side; we normalise to
  underscores at load time so downstream code never has to know.
* Downcasting float64->float32 / int64->int32 roughly halves the ~1.9 GB
  in-memory footprint of the joined training frame.  float32 keeps ~7 decimal
  digits, which is far more precision than these features carry.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)

RAW_FILES = {
    "train_transaction": "train_transaction.csv",
    "train_identity": "train_identity.csv",
    "test_transaction": "test_transaction.csv",
    "test_identity": "test_identity.csv",
    "sample_submission": "sample_submission.csv",
}


def normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``id-NN`` -> ``id_NN`` (test_identity.csv uses hyphens)."""
    renames = {c: c.replace("-", "_") for c in df.columns if c.startswith("id-")}
    return df.rename(columns=renames) if renames else df


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes in place-ish; object columns become ``category``.

    Why: the joined train frame is 590,540 x 433.  At float64 that is ~2 GB
    before a single feature is built, which makes every subsequent groupby a
    memory event.  Precision loss is irrelevant for these features.
    """
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            # Keep the ID and target comfortably exact.
            df[col] = s.astype(np.int32) if col != config.ID_COL else s.astype(np.int64)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            # pandas >= 3 gives string columns the ``str`` dtype rather than
            # ``object``; both become ``category`` so that the tree models can
            # use native categorical splits and the encoders see a fixed
            # vocabulary.
            df[col] = s.astype("category")
    return df


def load_raw(name: str, raw_dir: Path | None = None, **kwargs) -> pd.DataFrame:
    """Read one raw CSV by logical name (see ``RAW_FILES``)."""
    raw_dir = raw_dir or config.RAW_DIR
    path = raw_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/raw/README.md for provenance and how "
            "to obtain the five raw CSVs; they are deliberately untracked."
        )
    df = pd.read_csv(path, **kwargs)
    return normalise_identity_columns(df)


def load_joined(split: str = "train", raw_dir: Path | None = None,
                nrows: int | None = None) -> pd.DataFrame:
    """Load ``{split}_transaction`` left-joined to ``{split}_identity``.

    Adds ``has_identity_record`` here (not in the feature module) because it
    is a property of the *join itself* and would be unrecoverable once the
    frames are merged and nulls filled.
    """
    txn = load_raw(f"{split}_transaction", raw_dir, nrows=nrows)
    idt = load_raw(f"{split}_identity", raw_dir)

    matched = txn[config.ID_COL].isin(idt[config.ID_COL])
    df = txn.merge(idt, on=config.ID_COL, how="left", validate="one_to_one")
    # 1/0 rather than bool so it survives downcasting and model matrices alike.
    df = df.copy()  # de-fragment before the insert below (pandas PerformanceWarning)
    df["has_identity_record"] = matched.astype(np.int8).to_numpy()

    log.info("%s: %d rows, %d cols, identity match rate %.4f",
             split, len(df), df.shape[1], matched.mean())
    return downcast(df)


def file_checksums(raw_dir: Path | None = None) -> dict[str, str]:
    """md5 of each raw file, for the provenance table in data/raw/README.md."""
    raw_dir = raw_dir or config.RAW_DIR
    out = {}
    for name, fname in RAW_FILES.items():
        path = raw_dir / fname
        if not path.exists():
            continue
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[fname] = h.hexdigest()
    return out


def to_parquet_cache(split: str = "train") -> Path:
    """Materialise the joined frame as parquet in data/interim/ (untracked).

    Reading the 683 MB CSV takes ~40 s; the parquet round-trip takes ~4 s.
    Every later phase loads from here.
    """
    out = config.INTERIM_DIR / f"{split}_joined.parquet"
    if out.exists():
        return out
    df = load_joined(split)
    df.to_parquet(out, index=False)
    return out


def load_cached(split: str = "train") -> pd.DataFrame:
    """Load the joined frame, building the parquet cache on first call."""
    return pd.read_parquet(to_parquet_cache(split))
