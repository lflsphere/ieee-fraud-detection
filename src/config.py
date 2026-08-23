"""Central configuration: paths and the dataset's declared schema.

Kept in one place so that every phase (EDA, features, models, evaluation,
leakage audit) reads the *same* notion of "which columns are categorical",
rather than each script re-deriving it from dtypes.  Re-deriving is exactly
how a numeric-looking categorical (``card1``, ``addr1``, ``id_13``) silently
gets fed to a model as an ordered quantity.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FINAL_DIR = REPORTS_DIR / "final"
RESULTS_DIR = REPORTS_DIR / "results"

for _d in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, FINAL_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
# Declared categorical by the competition organisers.  Several of these hold
# integer codes (``card1``, ``addr1``, ``id_13``...); treating them as numeric
# would impose a false ordering ("card1=13926 > card1=2755"), so they are
# named explicitly here rather than inferred from dtype.
TRANSACTION_CATEGORICALS = [
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

IDENTITY_CATEGORICALS = (
    ["DeviceType", "DeviceInfo"] + [f"id_{i:02d}" for i in range(12, 39)]
)

CATEGORICAL_COLS = TRANSACTION_CATEGORICALS + IDENTITY_CATEGORICALS

# The anonymised numeric blocks.  ``C*`` are counts, ``D*`` are timedeltas,
# ``V*`` are Vesta-engineered richness/relationship features, ``id_01``-``id_11``
# are numeric identity measurements.
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
V_COLS = [f"V{i}" for i in range(1, 340)]
ID_NUMERIC_COLS = [f"id_{i:02d}" for i in range(1, 12)]

# --------------------------------------------------------------------------
# Chronological split points (fractions of the ordered-by-TransactionDT train
# set).  Fixed here so every phase uses byte-identical folds; see
# ``src/data/split.py`` for the rationale behind chronological (not random)
# partitioning.
# --------------------------------------------------------------------------
VALID_FRACTION = 0.15   # last 15% before the holdout -> model selection
TEST_FRACTION = 0.20    # final 20% of time -> untouched generalisation test
N_TIME_FOLDS = 5        # expanding-window folds for time-aware CV

# Business-relevant operating point used for F1/precision/recall reporting.
# Rationale: a fraud team can only manually review a bounded volume; reviewing
# the top 1% riskiest transactions is a realistic daily caseload for a stream
# of this size.  See reports/final/07_evaluation.md.
REVIEW_BUDGET_FRACTION = 0.01

RANDOM_SEED = 42
