# IEEE-CIS Fraud Detection — Data Science Case Study

Graduate case study: full pipeline from DGP reasoning through leakage-safe
evaluation on the IEEE-CIS Fraud Detection dataset.

See `IMPLEMENTATION_PLAN.md` for the phased plan and `claude.md` for
the brief used to hand this project off to an autonomous coding agent.

## Structure
- `src/data`        — loading, joins, PIT-safe train/test splitting
- `src/features`     — feature engineering (interactions, aggregations, leakage-checked)
- `src/unsupervised`  — clustering / representation learning
- `src/models`        — linear baseline, classical ML, neural net
- `src/evaluation`    — CV strategy, metrics
- `src/leakage`       — explicit leakage audits (kept separate from features on purpose)
- `notebooks/`        — EDA and exploratory work only; production logic lives in `src/`
- `reports/`          — figures + final write-up
