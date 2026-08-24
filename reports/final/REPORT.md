# IEEE-CIS Fraud Detection — Final Report

A predictive model for card-transaction fraud, built on the IEEE-CIS dataset
(590,540 labelled transactions over 183 days, 3.50% fraud) and validated
chronologically with a machine-checked leakage audit.

**Recommendation: deploy gradient-boosted trees at a top-1%-of-volume review
threshold. Expected performance on held-out future transactions: 87.8%
precision, recovering 25.5% of all fraud — 88% of what a perfect ranker could
achieve at that capacity.**

| | |
|---|---|
| Phase write-ups | [DGP](01_dgp.md) · [EDA notebook](../../notebooks/02_eda.ipynb) · [Feature dictionary](03_feature_dictionary.md) · [Unsupervised](04_unsupervised.md) · [NN training](06_nn_training.md) · [Evaluation](07_evaluation.md) · [Leakage audit](08_leakage_audit.md) · [Comparison](09_comparative_analysis.md) |
| Code | `src/` (data, features, unsupervised, models, evaluation, leakage) |
| Reproduce | see [README](../../README.md) |

---

## 1. The problem, as the data actually poses it

Three properties of this dataset determine every design decision that follows.
They were measured before any model was trained, and each one closed off an
approach that would otherwise have looked reasonable.

**Fraud is adversarial and non-stationary.** Weekly fraud rate moves from
1.85% to 5.06% across the observed period — a 2.7× swing that is a trend, not
noise around a constant. Transaction volume moves 13.5× independently. The
i.i.d. assumption underlying random cross-validation is false here, and any
model has a shelf life.

**Fraud is bursty and clusters within entities.** There is no customer ID, so
`card1|card2|card3|card5|addr1` serves as a proxy — 42,946 of them, 40.3% seen
only once. Only 8.3% ever carry a fraud label, but **91.7% of all fraudulent
transactions sit in entities carrying more than one fraud**. A compromised card
is drained, not defrauded once.

**The label is a policy output, not ground truth.** `isFraud` records
chargebacks and confirmed investigations, so undetected fraud is labelled
legitimate. A visible symptom: transactions with an identity record are 7.85%
fraudulent against 2.09% without — 3.7×. The likely mechanism is not that
fingerprinted devices are riskier, but that **the incumbent system collected
extra identity signal on transactions it already suspected**. We are modelling
the residual of a detection policy we cannot observe.

Full argument and assumptions: [`01_dgp.md`](01_dgp.md).

## 2. What the data showed

[`notebooks/02_eda.ipynb`](../../notebooks/02_eda.ipynb) — interpretation under
every figure. Six findings that changed the plan:

| Finding | Consequence |
|---|---|
| Fraud rate drifts 1.85% → 5.06% weekly | PR-AUC primary, per-fold reporting, review-budget operating point instead of a fixed probability threshold |
| Amount is near-useless alone (ROC-AUC **0.4975**) but fraud rate is **U-shaped** across deciles — elevated at both the cheapest (card-testing) and dearest (cash-out) ends | `log1p` *and* decile bins, so a linear model can express non-monotonicity |
| A clean 24-hour cycle exists despite the unknown time origin — volume swings 16×, fraud peaks **4.6×** in the volume trough | An explicitly *assumed* hour feature; raw `TransactionDT` excluded as monotone and non-transferable |
| **171 of 324** testable columns have informative missingness (MNAR) | Explicit null indicators; NaN passed natively to LightGBM |
| The 339 `V*` columns form **15 exact missingness groups** needing only **89 PCs** for 95% of within-group variance | Motivated the Phase 4 compression |
| 91.7% of fraud sits in repeat-fraud entities | Chronological split enforced by assertion, quantified in Phase 8 |

## 3. Features

508 features, each tagged with one of four leakage classes and each carrying a
written rationale — [`03_feature_dictionary.md`](03_feature_dictionary.md).

The load-bearing constructions:

- **21 point-in-time entity aggregates** over three proxy UIDs — velocity,
  prior amount mean/std, amount-to-prior-mean ratio, recency, 24h/7d window
  counts. All strictly backward-looking: running sums with the current row
  subtracted, `shift(1)` recency, and window counts that treat simultaneous
  transactions as *not* prior.
- **`D*` reference-day normalisation.** The `D*` columns are "days since a
  prior event"; `day − D` recovers *when that event happened*, which is
  near-constant per account and a far better entity signature than a
  countdown.
- **4 target encodings with a 30-day label lag** — restricted to labels old
  enough to have been adjudicated, because a chargeback that has not arrived
  is not information you have.
- **21 missingness indicators**, not 171: the identity-block indicators are
  near-duplicates of one another and collapse to `has_identity_record`.

## 4. Unsupervised representation — a negative result

The `V*` block compresses from 339 columns to 126 components across its 15
missingness groups. `k = 6` was chosen by a stability sweep (5 seeds × 6
values, mean pairwise adjusted Rand index), which **rejected the initial k = 8
as the least reproducible setting in the grid** (ARI 0.56, worst pair 0.28 —
it spins off degenerate 81-row and 2-row clusters whose membership flips with
the seed).

The clustering finds real structure: a partition built **with no access to the
label** isolates 4,853 transactions at **40.8% fraud — an 11.7× lift**.

**It does not help any model.** Phase 4 predicted the collinearity-sensitive
families would benefit; they do not, and the neural network — the strongest
theoretical case — is hurt most. PCA ranks directions by variance and never
sees the label; each model already had a cheaper answer to collinearity. The
write-up records this as a falsified prediction rather than dropping it:
[`04_unsupervised.md`](04_unsupervised.md) §5.

## 5. Models and results

Three families × three feature views, one harness, byte-identical folds, one
holdout scored once.

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

Protocol: chronological 65/15/20 split; five expanding-window CV folds guarded
by a hard `assert_no_time_overlap`; every preprocessing statistic refitted
inside each fold; the final 42-day holdout untouched by any other decision in
the project. [`07_evaluation.md`](07_evaluation.md).

### Three findings that matter more than the ranking

**ROC-AUC would have ranked these models wrong.** The neural network's 0.8442
sits 1% above the linear model's 0.8356 — a rounding error — while delivering
**2.3× the PR-AUC and 6.9× the precision** at the review budget.
`linear+compact` has the *best* ROC-AUC of the linear arms and the *worst*
precision@1%. The two metrics disagree in sign. At 3.5% prevalence, ROC-AUC
integrates mostly over regions no fraud team operates in.

**Cross-validation overstates forward performance, unequally.** Linear −53%,
GBM −16%, NN −13%. Part is protocol (the holdout forecasts further ahead), but
if it were only that, all three would lose similarly. **Most of the linear
model's collapse is the model.** This also corrects an expectation I held and
stated: I predicted the NN would be the most drift-fragile given ~20,663
positives against ~250k parameters. It is the least.

**Only the GBM produces a usable probability.** Its top decile predicts 24.7%
and observes 23.5% (Brier 0.023). The NN predicts 82.5% where 20.8% occurs; the
class-weighted linear model 92.6% against 18.5%. Both are overconfident by 4–5×
*by construction* — class weighting inflates probabilities, and the NN's
PR-AUC-based early stopping (worth 8.2%, [`06_nn_training.md`](06_nn_training.md))
explicitly trades calibration for ranking. Every threshold in this project is
therefore rank-based.

## 6. Leakage audit

[`08_leakage_audit.md`](08_leakage_audit.md). The pipeline is proven clean by
machine check, and each kind of leakage is quantified by building the leaky
version and measuring it.

**Clean:** 300/300 point-in-time aggregates match a brute-force re-derivation
exactly; corrupting the future changes **0 of 508 features** in the past; the
44 features depending on the fit population are **exactly** the 44 declared.
The audit is itself audited — a control test plants a full-dataset target
encoder and confirms the checker catches it.

**What leakage would have bought:**

| Type | Control | Effect if omitted |
|---|---|---:|
| Temporal | Chronological split | **+44% PR-AUC** (0.5945 → 0.8557), precision@1% → 99.6% |
| Feature (blatant) | Strictly-prior aggregation | +4.4% |
| Feature (subtle) | 30-day label lag | +1.0% — *and it passes the future-invariance audit* |
| Preprocessing | Fit inside each fold | **−22%: it made results worse** |

Two results deserve emphasis. The **zero-lag target encoding is backward-looking
and still wrong** — it consumes chargeback adjudications that would not exist at
scoring time, invisibly to a naive backtest. And the share of GBM gain taken by
target-encoded features rises **0.51% → 4.65% → 7.17%** across the three arms:
the leakier a feature is, the more the model leans on it, which makes that
ratio a standing leakage detector.

The preprocessing result is reported as measured, against expectation. The
lesson is stronger than "leakage inflates": leakage makes results
**unpredictable**, so "it would only flatter us" is not a defence.

## 7. Recommendation

**Deploy the GBM at a top-1%-of-volume review threshold, ranked rather than
thresholded on probability.** Of 1,181 flagged per 118,108 processed, ~1,037
are fraudulent; 25.5% of all fraud is recovered.

At 1% capacity against a 3.5% base rate a perfect ranker caps at 29.0% recall.
**The GBM reaches 88% of that arithmetic maximum.** More recall means buying
review capacity, not a better model.

The interpretability-versus-performance tradeoff does not arise: at 12%
precision the linear model is not a worse-but-explainable option, it is not an
option. The linear baseline earned its place as a **leakage canary** and as the
measurement of how much nonlinearity is worth here (0.34 PR-AUC).

The neural net is a genuine second, not a failure — 82% precision, half the
GBM's training cost, and the most drift-robust of the three. It loses on
calibration and on categorical handling, both traceable to specific documented
choices.

**The main risk in the recommended model:** `card1_code` alone carries **33% of
total gain**, and categorical codes 61%. The model is substantially a
card-risk lookup. **98.6% of holdout transactions use a card seen in
training**, so the headline number is almost entirely a statement about known
cards, and new-card performance must be monitored separately. It also means the
model partly inherits the incumbent policy's blind spots — which is what §1
predicted, now visible in the importances.

## 8. Limitations

1. **One holdout, one 42-day period.** Per-fold sd (0.032 for the GBM) is the
   best available guide to how much it might differ elsewhere.
2. **The holdout abuts the training period; the real task starts 30 days
   later.** These numbers are optimistic relative to the intended deployment.
3. **Label selection bias is unmeasurable here.** Precision figures are lower
   bounds — some "false positives" are fraud the incumbent system missed.
4. **The 30-day chargeback lag is an assumption**, not a measurement.
5. **Calendar features are assumed alignments** against an unknown time origin;
   one carries 2% of gain.
6. **No repeated-seed variance estimate.** The training loop is deterministic
   under a fixed seed, so fold spread captures data variation but not
   initialisation variation.
7. **The competition test set (506,691 rows) is unused** — unlabelled, and the
   semi-supervised opportunity was not taken.

## 9. Deviations from the plan

1. **"Stratified *and* time-based" split is self-contradictory.** Forcing equal
   fraud rates across time blocks means moving rows across time — the leak the
   split exists to prevent. Resolved by splitting strictly chronologically and
   *reporting* the realised drift. [`07_evaluation.md`](07_evaluation.md) §1.2.
2. **Phase 4's stated prediction was falsified** and the write-up amended
   rather than quietly reframed.
3. **No Transformer**, per instruction. Time went to Phases 8 and 9.
4. **Data provenance:** the Drive folder was located via the connector and the
   five files confirmed byte-identical, but owner-only sharing and a
   base64-only transport made it unusable for a 683 MB file. The files were
   fetched from a public Kaggle mirror of the same release and validated
   against the Drive originals on byte size, row counts, column counts and
   join rate. This is the real competition data, not a synthetic stand-in.
   [`data/raw/README.md`](../../data/raw/README.md).
