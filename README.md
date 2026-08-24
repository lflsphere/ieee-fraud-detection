# IEEE-CIS Fraud Detection — Data Science Case Study

A full pipeline on the IEEE-CIS Fraud Detection dataset (590,540 labelled
transactions over 183 days, 3.50% fraud), from data-generating-process
reasoning through leakage-audited, chronologically-validated model comparison.

**Read the report first:** [`reports/final/REPORT.md`](reports/final/REPORT.md).

## Results, traced through the EDA

Every design decision below was forced by a Phase 2 measurement, and each one
has a measured consequence.

| EDA finding (Phase 2) | Decision it forced | Measured impact |
|---|---|---|
| Fraud rate drifts **1.85% → 5.06%** weekly; volume 13.5× | PR-AUC primary, per-fold reporting, rank-based review budget instead of a probability threshold | **Vindicated.** ROC-AUC ranks the models wrong: NN 0.8442 vs linear 0.8356 (a 1% gap) hides **2.3× PR-AUC and 6.9× precision** |
| **91.7%** of fraud sits in repeat-fraud proxy entities | Chronological split, enforced by a hard assertion | **Largest single effect in the project.** A random split inflates PR-AUC **+44%** (0.594 → 0.856) and precision@1% to 99.6% |
| Amount univariate ROC-AUC **0.4975**, but fraud rate **U-shaped** across deciles | `log1p` *and* decile bins, so a linear model can express non-monotonicity | Minor: the whole amount/time/join family is **4.1%** of GBM gain. The U-shape is real; it just isn't where the signal lives |
| Clean 24-h cycle despite the unknown time origin; fraud peaks **4.6×** in the volume trough | `dt_hour_assumed` (labelled an assumption); raw `TransactionDT` **excluded** as monotone | `dt_hour_assumed` is a **top-8 feature at 1.8%** of gain — the assumption earned its place |
| **171 of 324** columns have informative missingness (MNAR) | 21 indicators, not 171; `has_identity_record` as the canonical one | **Overrated.** The whole is-null family is **0.07%** of GBM gain, and one (`R_emaildomain`, φ = 0.910) should not have been kept at all |
| 339 `V*` → 15 exact missingness groups → **89 PCs** for 95% of within-group variance | Phase 4 PCA + clustering, tested as an A/B against the raw block | **Negative result.** Helps no model family. Consolation: `gbm+compact` ties the full GBM on **41% fewer features** — a deployment argument, not an accuracy one |

### Bottom line

Gradient boosting wins: holdout **PR-AUC 0.5342**, **87.8% precision** and
**25.5% recall** at a 1%-of-volume review queue — 88% of the 29.0% recall a
*perfect* ranker could reach at that capacity. The neural net is a real second
(82% precision, half the training cost, most drift-robust of the three). The
linear baseline is not deployable at 12% precision, but earned its place as a
leakage canary and as the measurement of what nonlinearity is worth here
(0.34 PR-AUC).

| model | holdout PR-AUC | ROC-AUC | precision@1% | recall@1% | Brier | fit s |
|---|---:|---:|---:|---:|---:|---:|
| **gbm** | **0.5342** | **0.8913** | **0.878** | **0.255** | **0.023** | 338 |
| nn | 0.4417 | 0.8442 | 0.821 | 0.239 | 0.087 | 179 |
| linear | 0.1941 | 0.8356 | 0.119 | 0.035 | 0.200 | 78 |

### What the EDA got right, and wrong

The two **structural** findings — entity clustering and temporal drift — drove
the decisions that mattered most, and both were confirmed quantitatively. The
two **"informative pattern"** findings — MNAR missingness and `V*`
compressibility — looked impressive in EDA and delivered almost nothing
downstream. Compressibility and informativeness are not the same thing as
*incremental* predictive value, once a model already handles redundancy and
missingness natively.

### The caveat EDA did not surface

`card1_code` alone carries **33%** of GBM gain, and **98.6%** of holdout
transactions use a card seen during training — so the headline is largely a
statement about *known cards*. The review queue also concentrates **86.8%** of
its slots in the 20.1% of transactions carrying an identity record, where
recall is 40.1% against 8.3% elsewhere. That is the main risk in the
recommendation, and it only became visible after modelling. See
[`reports/final/01_dgp.md`](reports/final/01_dgp.md) §4 and
[`09_comparative_analysis.md`](reports/final/09_comparative_analysis.md) §4.

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
