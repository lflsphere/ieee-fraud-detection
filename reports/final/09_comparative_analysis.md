# Phase 9 — Comparative Analysis and Recommendation

**Inputs:** `reports/results/05_summary.csv`, `05_per_fold.csv`,
`05_importance_gbm.csv`, `06_nn_sensitivity.csv`, `08_*.csv`
**Figures:** `reports/figures/07_model_comparison.png`, `07_calibration.png`,
`07_gbm_importance.png`, `08_leakage_impact.png`

---

## 1. The comparison

All nine configurations, one harness, byte-identical folds, one chronological
holdout scored once. Sorted by holdout PR-AUC.

| model          | feats | CV PR-AUC | CV sd  | holdout PR-AUC | lift    | ROC-AUC | prec@1% | recall@1% | F1@1%  | Brier  | fit s    |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gbm+unsup      | 637   | 0.6386    | 0.0352 | 0.5354         | 15.5607 | 0.8899  | 0.8858  | 0.2576    | 0.3992 | 0.0224 | 423.6364 |
| gbm+compact    | 298   | 0.6374    | 0.0347 | 0.5350         | 15.5472 | 0.8876  | 0.8849  | 0.2574    | 0.3988 | 0.0224 | 240.5635 |
| gbm            | 508   | 0.6341    | 0.0345 | 0.5342         | 15.5247 | 0.8913  | 0.8782  | 0.2554    | 0.3957 | 0.0226 | 337.7094 |
| nn             | 508   | 0.5093    | 0.0481 | 0.4417         | 12.8366 | 0.8442  | 0.8206  | 0.2387    | 0.3698 | 0.0870 | 179.2808 |
| nn+compact     | 298   | 0.5096    | 0.0464 | 0.4292         | 12.4738 | 0.8610  | 0.7893  | 0.2296    | 0.3557 | 0.0991 | 101.8957 |
| nn+unsup       | 637   | 0.5107    | 0.0435 | 0.4165         | 12.1051 | 0.8489  | 0.7699  | 0.2239    | 0.3469 | 0.0769 | 190.3002 |
| linear         | 508   | 0.4152    | 0.0463 | 0.1941         | 5.6406  | 0.8356  | 0.1191  | 0.0347    | 0.0537 | 0.1999 | 77.5486  |
| linear+unsup   | 637   | 0.4137    | 0.0488 | 0.1907         | 5.5413  | 0.8339  | 0.1210  | 0.0352    | 0.0545 | 0.1981 | 101.4841 |
| linear+compact | 298   | 0.4161    | 0.0360 | 0.1841         | 5.3501  | 0.8387  | 0.0535  | 0.0167    | 0.0255 | 0.2060 | 54.1752  |

`feats` = input columns. `lift` = PR-AUC ÷ the holdout base rate of 3.44%
(a random ranker scores 1.0). `prec@1%` / `recall@1%` = performance on the
top 1% of scores — a 1,181-transaction review queue against 4,062 actual
frauds. `fit s` = wall-clock seconds to fit the final model on 4 CPU cores.

### 1.1 Three tiers, and the gaps between them are not subtle

**GBM (0.534–0.535).** All three views land within 0.0012 of each other.

**Neural network (0.417–0.442).** 17–22% behind the GBM on PR-AUC but much
closer at the operating point — 82.1% precision against 87.8%.

**Linear (0.184–0.194).** Not in the same conversation. At the review budget it
is right 12% of the time.

---

## 2. Recommendation: deploy the GBM, on the `compact` feature view

`gbm+unsup` is nominally top of the table at 0.5354, but the three GBM arms are
separated by 0.0012 — noise. Choosing between them on that number would be
selecting on the holdout, which is exactly what Phase 7 refuses to do
everywhere else. The choice should be made on everything *except* the fourth
decimal:

| | `gbm` | `gbm+unsup` | `gbm+compact` |
|---|---:|---:|---:|
| holdout PR-AUC | 0.5342 | 0.5354 | 0.5350 |
| input columns | 508 | 637 | **298** |
| fit time | 338 s | 424 s | **241 s** |
| pipeline stages at scoring time | Phase 3 | Phase 3 **+ Phase 4** | Phase 3 + Phase 4 |

`gbm+compact` matches the others on accuracy with **41% fewer input columns**
and **29% less training time**. Against that, it adds a dependency: the PCA
rotations and K-means centroids from Phase 4 must be fitted, versioned and
applied at scoring time, whereas the plain `gbm` view needs only the Phase 3
pipeline.

**Recommendation: `gbm` on the base view for a first deployment; `gbm+compact`
if and when the scoring-time cost of materialising 339 `V*` columns becomes a
real constraint.** The accuracy case between them is a tie, so the tiebreak is
operational simplicity, and one fewer fitted artefact to version is worth more
on day one than 97 fewer columns.

This is the outcome the plan anticipated — "likely GBM" — but the reasons are
worth stating in the project's own terms rather than by appeal to convention:

1. **Native categorical handling is decisive here.** `card1` has 13,553 levels.
   LightGBM partitions levels directly by gradient statistics; the linear and
   neural models must one-hot or embed. Categorical codes account for **66.8%
   of total GBM gain** (§4), so this is not a marginal convenience — it is most
   of the model.
2. **Native NaN handling.** A third of the feature matrix is missing and the
   missingness is MNAR (Phase 2 §4). LightGBM learns a default direction per
   split; the other two must impute, and the indicator features exist only to
   repair what imputation destroys.
3. **It is the only model that produces a usable probability** (Brier 0.023;
   top decile predicts 24.7%, observes 23.5%). The other two are overconfident
   by 4–5×.
4. **It is the most stable across time** (per-fold sd 0.032 against 0.044 for
   the NN and 0.046 for the linear model) — and for a model sitting at a fixed
   production threshold, consistency is worth as much as the mean.

---

## 3. The tradeoffs, stated honestly

### 3.1 Interpretability versus performance

The usual framing — the linear model is interpretable, the GBM is a black box,
choose your point on the curve — **does not survive contact with these
numbers**. The linear model is not a worse-but-explainable option; at 12%
precision on the review queue it is not an option. There is no tradeoff to
make, because one arm of it does not function.

What *is* true is that the GBM's interpretability is different in kind, not
absent. Gain-based importance is available and legible (§4), per-prediction
SHAP attributions are computable, and monotonic constraints could be imposed on
features where domain knowledge justifies them. What is lost is the ability to
hand a regulator a table of coefficients. For fraud detection — where the
decision is "route to human review," not "deny credit" — that is a much smaller
loss than in lending, and the human reviewer supplies the explanation the model
cannot.

The linear baseline still earned its place, for two reasons that have nothing
to do with deployment. It is the **leakage canary**: a linear model cannot
memorise an entity the way a deep ensemble can, so an implausibly good linear
score is a loud alarm. And it **localises the gap** — the 0.34 PR-AUC between
linear and GBM is the measured value of nonlinearity and interaction on this
problem, which is a quantity worth knowing.

### 3.2 Cost versus marginal gain

| | linear | nn | gbm |
|---|---:|---:|---:|
| fit time (4 CPU cores) | 78 s | 179 s | 338 s |
| holdout PR-AUC | 0.194 | 0.442 | 0.534 |
| PR-AUC per second of fit | 0.0025 | 0.0025 | 0.0016 |

The GBM costs 4.3× the linear model's training time for 2.8× the PR-AUC and 7×
the precision. On any fraud book of meaningful size that is not a close call:
338 seconds of CPU is free relative to a single prevented chargeback, and
retraining is weekly at most.

Inference cost tells the same story. All three are milliseconds per
transaction; the GBM's ~2,000 trees are more expensive than a dot product but
remain far below any realistic authorisation latency budget. **Compute cost is
not a live constraint on this decision**, and it would be dishonest to present
it as a meaningful counterweight.

### 3.3 Where the neural network actually stands

The plan predicted the NN would lose on tabular data at this scale, and it does
— but the honest reading is more interesting than "prediction confirmed":

- It reaches **82% precision at the review budget against the GBM's 88%**. Both
  are deployable; the linear model is not. Framing this as "the NN failed"
  would be wrong.
- It is the **most robust to temporal drift of the three** (−13% CV-to-holdout
  against the GBM's −16%). I expected the opposite, given ~20,663 positives
  against ~250k parameters, and said so before running it. Phase 6 suggests
  why: early stopping on validation PR-AUC halts it at epoch 24 of 30, well
  before the memorisation the unregularised arm shows.
- It is **half the training cost of the GBM** (179 s vs 338 s).
- Its weakness is concentrated where it matters: **calibration is 3.8× worse**
  (Brier 0.087 vs 0.023), a direct and documented consequence of the Phase 6
  stopping rule.

Where an NN would become the right answer: if the feature set gained genuinely
sequential structure (per-card transaction sequences rather than pre-aggregated
counts), or high-cardinality entities where learned embeddings could be shared
across a multi-task objective. Neither is present here, which is precisely why
it loses.

### 3.4 The unsupervised work: a negative result, reported as one

The Phase 4 representation helps **no** model family, and the write-up
(`04_unsupervised.md` §5) has been amended to say so rather than left standing
on a falsified prediction:

| family | `base` | `unsup` | `compact` |
|---|---:|---:|---:|
| linear | **0.1941** | 0.1907 | 0.1841 |
| GBM | 0.5342 | **0.5354** | 0.5350 |
| NN | **0.4417** | 0.4165 | 0.4292 |

Phase 4 predicted the collinearity-sensitive families would benefit. They do
not; the NN — the family with the strongest theoretical case — is hurt most
(−5.7%). Three explanations the data supports: PCA ranks directions by variance
and never sees the label, so the discarded 5% of within-group variance was not
uniformly uninformative; each model already had a cheaper answer to
collinearity (L2, feature subsampling, embeddings + BatchNorm); and the damage
concentrates in the head of the score distribution, where the review budget
lives.

What survives is real: the `V*` block genuinely decomposes into 15 exact
missingness groups, genuinely compresses to 126 components, and a **label-free
partition genuinely isolates 4,853 transactions at 40.8% fraud — an 11.7×
lift**. Compressibility simply does not imply a better input representation
for models that were already handling the redundancy.

The one arm with practical value is `gbm+compact`: same accuracy, 41% fewer
columns. That is a deployment argument, not an accuracy one, and §2 treats it
as such.

---

## 4. What the deployed model is actually using — and the risk in it

Share of total LightGBM gain by feature family:

| family | share of gain |
|---|---:|
| categorical codes | **60.7%** |
| raw `V*` | 12.8% |
| raw `C*` (counts) | 9.0% |
| point-in-time entity aggregates (Phase 3) | 4.6% |
| amount / assumed-time / join | 4.1% |
| `D*` reference-day (Phase 3) | 3.2% |
| raw `D*` | 3.2% |
| frequency encodings | 1.8% |
| target encodings | 0.7% |
| is-null indicators | 0.07% |

Top individual features: `card1_code` (**33.0%**), `id_19_code` (8.4%),
`card2_code` (6.0%), `addr1_code` (5.0%), `V294` (3.9%), `C1` (3.0%),
`C13` (2.7%), `dt_hour_assumed` (1.8%).

*(These are shares of the **full** 508-feature gain vector. An earlier version
of this table was computed from an importance file that `save_results`
truncated to the top 50 rows, which inflates every share by silently changing
the denominator — categorical codes read 66.8% instead of 60.7%. The
truncation is fixed and the vector is now persisted in full.)*

**Three things follow, and the first is a genuine concern.**

**(a) The model is substantially a card-risk lookup.** `card1_code` alone
carries 33% of gain, and categorical codes 61%. That is legitimate — the card identifier is available at
authorisation time — but it means much of what the model knows is *which cards
have been trouble*, not *what fraud looks like*. Two consequences: performance
on genuinely new cards will be materially worse than the headline, and the
model inherits the incumbent policy's blind spots (Phase 1 §4).

We can bound the first: **98.6% of holdout transactions use a `card1` value
seen during training**, so only 1.4% of the holdout tests the
never-seen-before case (and their fraud rate, 3.53%, is close to the 3.44%
overall, so they are not a distinct population). **The headline number is
therefore almost entirely a statement about known cards.** Any deployment
should monitor new-card performance separately, and that is a stated
limitation rather than a resolved issue.

**(b) The Phase 3 engineered features earn a modest but real 7.8%** (4.6%
point-in-time entity aggregates + 3.2% reference-day) — more than the
frequency and target encodings combined. The `day − D` normalisation and the
strictly-backward-looking velocity features were worth building.

**(c) Target encodings contribute 0.7% of gain — and that is the point.** Phase
8, measuring the same share on its lighter 600-tree configuration, showed it
rising to 4.65% at zero label lag and 7.17% with full-dataset encoding. The shipped model barely relies on them precisely
because the leakage controls stripped out the part that was free information.
A future pipeline change that pushes this number up is a leakage alarm, not an
improvement.

**(d) The one feature-selection error had no measurable cost — in this model.**
`R_emaildomain_isnull` should not have been retained (its missingness is 91%
collinear with the identity join; see `03_feature_dictionary.md` §3). In the
GBM it carries **0.0000% of gain** — LightGBM simply never split on it, and the
entire is-null family accounts for 0.07%. So the error changed nothing here.
It is not free everywhere, though: in the linear model the same indicator ranks
**24th of the top 50 coefficients** (|coef| 0.657 against a top coefficient of
3.516), so a model without native categorical handling does lean on a feature
that is largely a restatement of `has_identity_record`. That is one more reason
the linear arm underperforms, and a reminder that feature hygiene matters most
for the models least able to ignore a bad feature.

`dt_hour_assumed` at 1.8% is worth flagging as an *assumption* carrying real
weight: `TransactionDT` has no known origin, so this is a periodic position,
not a clock reading. It works because the 24-hour structure is unambiguous
(Phase 2 §3), but if the true offset were ever recovered the feature should be
rebuilt.

---

## 5. Deployment recommendation in operational terms

**Deploy the GBM at a top-1%-of-volume review threshold, set on rank rather
than on an absolute probability.**

Expected performance, from the holdout: of 1,181 flagged transactions per
118,108 processed, **~1,037 are fraudulent (87.8% precision)**, recovering
**25.5% of all fraud** in the period.

That recall deserves context rather than apology. At 1% capacity against a 3.5%
base rate, a **perfect** ranker caps at 29.0% recall — there are 4,062 frauds
and only 1,181 slots. The GBM achieves **88% of the arithmetic maximum**.
Raising recall requires buying review capacity, not a better model; that is a
budget decision, and the PR curve in `07_model_comparison.png` is the input to
it.

Operational requirements:

1. **Retrain at least monthly.** Fraud rate moved 1.85% → 5.06% across the
   observed 26 weeks (Phase 2 §1), and the CV-to-holdout drop of 16% is what
   ~40 days of staleness costs.
2. **Threshold on rank, not probability**, even though the GBM is the
   best-calibrated of the three. Rank is invariant to the base-rate drift that
   would otherwise silently change the queue's volume week to week.
3. **Monitor the `*_te` gain share** (currently 0.3%). A rise indicates a
   leakage regression.
4. **Monitor new-card performance separately** from the aggregate, per §4(a).
5. **Track precision decay as an early-warning signal.** A drop at fixed volume
   means either drift or an adversary adapting — and under the Phase 1 DGP,
   adaptation is the expected steady state, not an anomaly.

---

## 6. What we would do next, given more time

1. **Recover the calendar origin of `TransactionDT`.** Several features are
   labelled "assumed" and one carries 2% of gain; a known offset would upgrade
   them from assumption to measurement.
2. **A per-entity sequence model.** The one architecture with a real case for
   beating the GBM here, since it would use structure the current pre-aggregated
   features destroy. Explicitly *not* a Transformer — out of scope for this
   assignment by instruction, and unjustified at this data scale regardless.
3. **Empirically estimate the chargeback lag** rather than assuming 30 days.
   Phase 8 §7 lists this as the top residual risk.
4. **Cost-sensitive thresholding** using transaction amount, replacing the
   flat 1% budget with an expected-loss ranking. This needs calibrated
   probabilities, which is the one place the GBM's Brier advantage would
   translate into money.
5. **Semi-supervised use of the unlabelled competition test set** — 506,691
   rows that Phase 4's clustering could have used and did not.
