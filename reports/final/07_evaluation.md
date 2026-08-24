# Phase 7 — Model Evaluation and Validation

**Code:** `src/evaluation/evaluate.py` (harness), `src/evaluation/metrics.py`
(metrics), driver `scripts/05_train_models.py`
**Results:** `reports/results/05_summary.csv`, `05_per_fold.csv`,
`05_calibration_*.csv`, `05_holdout.json`
**Figures:** `reports/figures/07_model_comparison.png`,
`07_calibration.png`, `07_gbm_importance.png`

---

## 1. The protocol, and why each part of it is there

One function, `run_experiment`, drives all nine (model × feature view) pairs
through **byte-identical folds and identical metrics**. Nothing about the
protocol lives inside a model class. That matters for the Phase 9 comparison:
if each model brought its own evaluation loop, any difference in the results
table could be an artefact of the loop rather than of the model.

### 1.1 A chronological three-way split

`train_transaction.csv` is ordered by `TransactionDT` and cut 65 / 15 / 20 **by
time**, not at random:

| partition | rows | `TransactionDT` range (days) | fraud rate |
|---|---:|---|---:|
| train | 383,851 | 1.0 – 111.3 | 3.42% |
| valid | 88,581 | 111.3 – 141.1 | **3.92%** |
| test (holdout) | 118,108 | 141.1 – 183.0 | 3.44% |

The holdout is the **last 42 days** of the labelled period. It is scored once,
at the end, and is touched by no other decision in the project — not fold
generation, not early stopping, not hyperparameter choice, not feature
selection.

### 1.2 On "stratified *and* time-based"

The plan asks for a stratified split and a time-based split. **Those two
requirements are in direct conflict**, and the conflict is worth naming rather
than papering over. Forcing an equal fraud rate into each time block would
require moving transactions across time — which is precisely the leak the
chronological split exists to prevent.

We resolve it the safe way: the split is strictly chronological, and the
realised class balance is *reported* rather than engineered. The table above
shows the drift that survives — the validation block sits at 3.92% against
3.42% in training — and every metric below is quoted next to the base rate it
was computed against. Stratification is applied only where it is harmless: in
the class weighting inside each model (`class_weight="balanced"`,
`scale_pos_weight`, `pos_weight`).

### 1.3 Expanding-window cross-validation

Five folds over the 80% that is not the holdout. Fold *k* trains on time blocks
0…*k* and validates on block *k*+1:

| fold | n train | n valid | train ends (day) | valid starts (day) | train fraud | valid fraud |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78,739 | 78,739 | 19.8 | 19.8 | 2.67% | 2.73% |
| 1 | 157,478 | 78,739 | 37.9 | 37.9 | 2.70% | **4.20%** |
| 2 | 236,217 | 78,739 | 64.7 | 64.7 | 3.20% | 3.94% |
| 3 | 314,956 | 78,738 | 90.8 | 90.8 | 3.38% | 3.71% |
| 4 | 393,694 | 78,738 | 114.6 | 114.6 | 3.45% | 3.83% |

Expanding rather than sliding, because fraud volume grows across the period and
discarding early history would starve the early folds. The cost is that later
folds train on five times the data of fold 0, so **per-fold results are always
reported, never only their mean**.

`src/data/split.assert_no_time_overlap` is called on every fold set before use
and raises if any training row post-dates any validation row in the same fold.
It is a hard assertion, not a convention, and `src/tests/test_data.py` includes
a test that it *fails* on a deliberately inverted fold — an assertion nobody has
watched fail is not evidence of anything.

### 1.4 What refits inside each fold

Everything. Each fold constructs a fresh model, which fits its own imputer,
scaler, one-hot encoder, category vocabulary and early-stopping iteration
count from that fold's training rows alone. Preprocessing lives inside each
model's `Pipeline` specifically so that fitting it before the split is
structurally impossible.

### 1.5 The final holdout fit

Fit on the earliest 65%, early-stop on the next 15%, score the last 20% once.

An earlier version of the harness fitted the final model on the whole 80% and
early-stopped on the last block *of that same 80%* — so the stopping criterion
was measured on rows the model had already fitted, could never trigger, and
reported a meaningless "best iteration". That was a real defect, caught and
fixed (commit `c55dab1`). Giving up 15% of the training rows is the price of an
honest stopping rule.

**No hyperparameter search ever ran against the holdout.** Everything tuned was
tuned on the CV folds or, for the neural network, on the Phase 6 ablations.

---

## 2. Metrics, and why not the obvious ones

**PR-AUC is primary.** At a 3.5% base rate, ROC-AUC is dominated by the 96.5%
of the curve concerned with ranking negatives against each other. §4 shows this
is not a theoretical concern on this data: ROC-AUC would have ranked these
models wrong.

**PR-AUC lift over the base rate is reported alongside it.** A random ranker
scores PR-AUC equal to the prevailing base rate, so when that rate moves from
2.67% to 4.20% across folds, the raw PR-AUC moves with it for reasons that have
nothing to do with the model. The lift ratio is the comparable number.
`src/tests/test_models_and_eval.py` asserts a random ranker scores a lift near
1 at base rates of 1%, 5% and 20%.

**F1 at a fixed review budget, not at threshold 0.5.** A 0.5 cut is meaningless
for a class-weighted model on a 3.5% problem. What a fraud team actually has is
bounded review capacity, so we score the **top 1%** of transactions by
predicted risk (`REVIEW_BUDGET_FRACTION` in `src/config.py`) and report
precision, recall and F1 there. On the 118,108-row holdout that is a queue of
1,181 transactions — a plausible daily caseload.

**An oracle F1 is reported as a bound, never as an operating point.** Choosing
a threshold by maximising F1 on the evaluation set is selection on the test
data; it appears in the table labelled `f1_oracle` purely so the
budget-constrained number can be read against what the ranking could deliver
with unlimited capacity.

**Brier score, interpreted with care.** The labels are the output of a prior
detection policy with selection bias (Phase 1 §4), so "well calibrated" means
calibrated to the historical labelling process, not to true fraud probability.

---

## 3. Results

### 3.1 Headline table (sorted by holdout PR-AUC)

| model          | features | CV PR-AUC | CV sd  | holdout PR-AUC | lift    | holdout ROC-AUC | prec@1% | recall@1% | F1@1%  | fit s    |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gbm+unsup      | 637      | 0.6386    | 0.0352 | 0.5354         | 15.5607 | 0.8899          | 0.8858  | 0.2576    | 0.3992 | 423.6364 |
| gbm+compact    | 298      | 0.6374    | 0.0347 | 0.5350         | 15.5472 | 0.8876          | 0.8849  | 0.2574    | 0.3988 | 240.5635 |
| gbm            | 508      | 0.6341    | 0.0345 | 0.5342         | 15.5247 | 0.8913          | 0.8782  | 0.2554    | 0.3957 | 337.7094 |
| nn             | 508      | 0.5093    | 0.0481 | 0.4417         | 12.8366 | 0.8442          | 0.8206  | 0.2387    | 0.3698 | 179.2808 |
| nn+compact     | 298      | 0.5096    | 0.0464 | 0.4292         | 12.4738 | 0.8610          | 0.7893  | 0.2296    | 0.3557 | 101.8957 |
| nn+unsup       | 637      | 0.5107    | 0.0435 | 0.4165         | 12.1051 | 0.8489          | 0.7699  | 0.2239    | 0.3469 | 190.3002 |
| linear         | 508      | 0.4152    | 0.0463 | 0.1941         | 5.6406  | 0.8356          | 0.1191  | 0.0347    | 0.0537 | 77.5486  |
| linear+unsup   | 637      | 0.4137    | 0.0488 | 0.1907         | 5.5413  | 0.8339          | 0.1210  | 0.0352    | 0.0545 | 101.4841 |
| linear+compact | 298      | 0.4161    | 0.0360 | 0.1841         | 5.3501  | 0.8387          | 0.0535  | 0.0167    | 0.0255 | 54.1752  |

### 3.2 PR-AUC per time fold

| model          | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | mean   | sd     |
|---|---:|---:|---:|---:|---:|---:|---:|
| gbm            | 0.5899 | 0.6276 | 0.6638 | 0.6069 | 0.6825 | 0.6341 | 0.0315 |
| gbm+compact    | 0.5982 | 0.6266 | 0.6678 | 0.6069 | 0.6873 | 0.6374 | 0.0316 |
| gbm+unsup      | 0.5928 | 0.6320 | 0.6685 | 0.6116 | 0.6879 | 0.6386 | 0.0321 |
| linear         | 0.3363 | 0.4383 | 0.4336 | 0.3958 | 0.4722 | 0.4152 | 0.0423 |
| linear+compact | 0.3688 | 0.4505 | 0.4290 | 0.3781 | 0.4543 | 0.4161 | 0.0329 |
| linear+unsup   | 0.3357 | 0.4400 | 0.4344 | 0.3831 | 0.4753 | 0.4137 | 0.0446 |
| nn             | 0.4298 | 0.5288 | 0.5402 | 0.4822 | 0.5656 | 0.5093 | 0.0439 |
| nn+compact     | 0.4443 | 0.5320 | 0.5418 | 0.4654 | 0.5645 | 0.5096 | 0.0424 |
| nn+unsup       | 0.4441 | 0.5141 | 0.5409 | 0.4851 | 0.5693 | 0.5107 | 0.0397 |

---

## 4. Reading the results

### 4.1 ROC-AUC would have ranked the models wrong

This is the single most important validation finding in the project.

| model | holdout ROC-AUC | holdout PR-AUC | precision@1% |
|---|---:|---:|---:|
| linear | 0.8356 | 0.1941 | 0.119 |
| nn | 0.8442 | 0.4417 | 0.821 |
| gbm | 0.8913 | 0.5342 | 0.878 |

By ROC-AUC the neural network (0.8442) looks like a rounding error above the
logistic regression (0.8356) — a 1% improvement that few would consider worth
the engineering. By the metrics that describe the actual deployment, it finds
**2.3× the PR-AUC and 6.9× the precision** at the review budget.

`linear+compact` makes the point sharper still: it records the **best ROC-AUC
of the three linear arms** (0.8387) and by far the **worst precision@1%**
(0.054, less than half the base view's 0.119). ROC-AUC and the review-budget
metrics disagree in *sign* on that comparison.

The mechanism is that ROC-AUC integrates over all thresholds with equal weight,
and at 3.5% prevalence almost all of that integral concerns regions no fraud
team will ever operate in. A review budget lives entirely in the extreme head
of the score distribution, and that is exactly where the models differ most.

### 4.2 Cross-validation systematically overstates forward performance

Every model scores materially lower on the chronological holdout than on the
CV folds — but by very different amounts:

| model | CV PR-AUC (mean) | holdout PR-AUC | relative drop |
|---|---:|---:|---:|
| linear | 0.4152 | 0.1941 | **−53%** |
| nn | 0.5093 | 0.4417 | **−13%** |
| gbm | 0.6341 | 0.5342 | **−16%** |

Two distinct effects are mixed here, and separating them matters.

*The part that is protocol.* The holdout is the last 42 days and the model saw
only the first 111; the CV folds validate one block ahead of a training set
that always abuts them. So even a drift-free process would show some gap,
because the holdout forecast horizon is longer.

*The part that is the model.* If the gap were purely the horizon, all three
models would lose a similar fraction. They do not. **The linear model loses
53% where the GBM loses 16% and the neural network 13%.** Most of the linear
model's collapse is therefore the model, not the drift. A plausible reading:
the linear model leans on a handful of strong marginal effects
(`reports/results/05_importance_linear.csv` shows the `C*` counting block and
`D1`/`D1_ref_day` dominating), and those marginals shift most as the attack mix
turns over, whereas the tree ensemble and the network both spread their weight
over many interactions and degrade more gracefully.

This also corrects an intuition I held before running it: I expected the
neural network to be the most drift-fragile of the three, given ~20,663
positives against ~250k parameters. It is in fact the **least** — a 13% drop,
slightly better than the GBM's 16%. The Phase 6 ablations point at why: early
stopping on validation PR-AUC halts the network at epoch 24 out of 30, well
before it memorises the training fold (the unregularised arm reaches a training
loss of 0.05 and decays ten times faster after its peak).

### 4.3 Fold-to-fold variation is real and orderly

Fold 0 is the weakest for every single model (GBM 0.590 vs 0.683 at fold 4;
linear 0.336 vs 0.472). Two causes compound: fold 0 trains on only 78,739 rows,
and it sits in the lowest-fraud stretch of the period (2.67%). The steady
improvement to fold 4 is mostly the expanding window doing its job.

The standard deviations are informative in their own right. The GBM arms are
the most stable (sd ≈ 0.032) and the linear and NN arms the least (0.040–0.045)
despite lower means — so the GBM is not merely better on average, it is better
*and* more consistent across time, which is what matters for a model that will
sit at a fixed threshold in production.

### 4.4 Calibration: the models are not interchangeable as probabilities

Highest-scoring decile on the holdout, predicted versus observed fraud rate:

| model | mean predicted | observed | Brier |
|---|---:|---:|---:|
| gbm | 0.2465 | **0.2352** | **0.0226** |
| nn | 0.8254 | 0.2084 | 0.0870 |
| linear | 0.9257 | 0.1847 | 0.1999 |

**Only the GBM produces anything resembling a probability.** Its top decile
predicts 24.7% and observes 23.5% — close enough to act on. The neural network
predicts 82.5% where 20.8% is observed, and the class-weighted logistic
regression predicts 92.6% against 18.5%: both are wildly overconfident, by a
factor of four to five.

This is *expected* and partly deliberate, and it is important not to present it
as a defect discovered by accident:

- `class_weight="balanced"` re-weights the logistic regression's loss as though
  the classes were equal, which inflates every predicted probability. The
  resulting scores are a valid *ranking* and an invalid *probability*.
- The network is early-stopped on PR-AUC rather than loss (Phase 6 §1), which
  explicitly trades calibration for ranking — worth 8.2% relative PR-AUC and
  costing exactly what the Brier score here shows.
- The GBM's `scale_pos_weight` has the same distorting pressure, but LightGBM's
  leaf values are fitted to residuals in log-odds space and the effect is far
  milder.

**Consequence.** All threshold decisions in this project are made on *rank*
(the top-1% review budget), never on an absolute probability, and that choice
is what makes the comparison valid across three models with wildly different
calibration. If a downstream use required real probabilities — expected-loss
routing, for instance, where you multiply P(fraud) by transaction value — the
GBM is the only one usable as-is, and the other two would need Platt or
isotonic recalibration fitted on a held-out block.

### 4.5 What the review budget buys

At the top 1% of holdout scores (1,181 of 118,108 transactions, against 4,067
actual frauds):

| model | precision@1% | recall@1% | F1@1% | frauds caught |
|---|---:|---:|---:|---:|
| gbm | **0.878** | 0.255 | 0.396 | ~1,037 |
| nn | 0.821 | 0.239 | 0.370 | ~971 |
| linear | 0.119 | 0.035 | 0.054 | ~141 |

The GBM's queue is right **88% of the time** and recovers a quarter of all
fraud in the period from 1% of the volume. The linear baseline's queue is right
12% of the time — seven wasted reviews in eight. That is the difference between
a deployable process and one an operations team would abandon in a week, and it
is far wider than the 2.8× PR-AUC ratio suggests.

**The queue is not evenly spread across the population**, and the headline
hides it: 86.8% of the 1,182 flagged transactions carry an identity record,
against a 20.1% base rate, and recall is 40.1% inside that segment versus 8.3%
outside it. Phase 1 §4 measures this and works through what selection and
feedback bias each do to these numbers — in short, precision is a lower bound
while the *incremental* value over the incumbent system is overstated. Any
deployment should report these metrics segmented rather than pooled.

Recall of 25.5% is the honest ceiling of this operating point, not a
disappointment: at 1% capacity against a 3.5% base rate, catching everything is
arithmetically impossible — even a perfect ranker would cap at ~29% recall
(1,181 slots for 4,067 frauds). **The GBM achieves 88% of that theoretical
maximum.** Raising recall means buying review capacity, not a better model, and
that is the trade-off Phase 9 puts in front of the business.

---

## 5. Limitations of this evaluation

1. **One holdout, one period.** The final number rests on a single 42-day
   window. The per-fold spread (sd ≈ 0.032 for the GBM) is the best available
   guide to how much it might have differed on a different 42 days.
2. **The competition test set is unusable here.** It carries no labels, so all
   evaluation happens inside the labelled period. The real deployment gap is
   also larger than anything measured: the competition test period begins 30
   days after training data ends, whereas our holdout abuts the training period
   directly. **Our holdout numbers are therefore optimistic** relative to the
   intended task.
3. **Label selection bias is unmeasurable from this data** (Phase 1 §4). Some
   "false positives" in the precision figures are real fraud the incumbent
   system missed, so precision@1% is a lower bound.
4. **No repeated-seed variance estimate.** Each configuration ran once per
   fold. The Phase 6 re-runs confirm the training loop is deterministic under a
   fixed seed, so the fold-to-fold spread captures data variation but not
   initialisation variation.
