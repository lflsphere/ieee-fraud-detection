# IEEE-CIS Fraud Detection — Data Science Case Study

A full pipeline on the IEEE-CIS Fraud Detection dataset (590,540 labelled
transactions over 183 days, 3.50% fraud), from data-generating-process
reasoning through leakage-audited, chronologically-validated model comparison.

**Read the report first:** [`reports/final/REPORT.md`](reports/final/REPORT.md).

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | Paths and the declared schema — one source of truth for which columns are categorical. |
| `src/data/` | Loading, the transaction⟕identity left join, chronological splitting and time-aware CV. |
| `src/features/` | Feature engineering, with every feature tagged by leakage class. |
| `src/unsupervised/` | `V*`-block PCA compression and stability-selected clustering. |
| `src/models/` | Logistic-regression baseline, LightGBM, PyTorch MLP — behind one interface. |
| `src/evaluation/` | Metrics and the shared evaluation harness all models run through. |
| `src/leakage/` | Machine-checked leakage audits, kept separate from the code they audit. |
| `src/tests/` | `pytest` suite covering the point-in-time invariants and model contracts. |
| `scripts/` | One driver per phase; each writes to `reports/`. |
| `notebooks/02_eda.ipynb` | Executed EDA with interpretation under every figure. |
| `reports/final/` | Phase write-ups and the stitched `REPORT.md`. |
| `reports/figures/`, `reports/results/` | Generated figures and result tables. |

## Reproducing

```bash
pip install -r requirements.txt

# 1. Place the five raw CSVs in data/raw/ — see data/raw/README.md for
#    provenance and the checksum table. They are never committed.
python -m src.data.schema_check              # verify the files match the schema

# 2. Run the phases in order (PYTHONPATH=. so `src` resolves)
PYTHONPATH=. python scripts/01_dgp_evidence.py
PYTHONPATH=. python scripts/02_build_eda_notebook.py
PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 notebooks/02_eda.ipynb
PYTHONPATH=. python scripts/03_build_features.py
PYTHONPATH=. python scripts/04_cluster_v_block.py
PYTHONPATH=. python scripts/05_train_models.py
PYTHONPATH=. python scripts/06_nn_training_dynamics.py
PYTHONPATH=. python scripts/08_leakage_experiments.py

pytest src/tests -q
```

`scripts/03` and `scripts/04` write parquet caches to `data/interim/`
(untracked); later phases read from those rather than re-parsing the CSVs.

## Ground rules this project holds itself to

1. **Chronological validation, never random.** 91.7% of fraudulent
   transactions sit in proxy entities that carry more than one fraud, so a
   shuffled split scores memorisation of a fraud burst rather than detection.
   Phase 8 measures the inflation.
2. **Every aggregate is strictly backward-looking**, and it is machine-checked:
   corrupting the future must not change a single feature value in the past.
3. **Target encodings respect a 30-day label lag**, because a chargeback that
   has not arrived yet is not information you have at scoring time.
4. **Every feature carries a written rationale** —
   `reports/final/03_feature_dictionary.md`.
5. **No Transformer.** Out of scope for this dataset option; the time went to
   the leakage audit and the write-up instead.
