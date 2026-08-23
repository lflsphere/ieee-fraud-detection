# Phase 8 — Data Leakage Audit

**Code:** `src/leakage/audit.py` (the machine checks), driver
`scripts/08_leakage_experiments.py`
**Tests:** `src/tests/test_leakage_audit.py`
**Results:** `reports/results/08_static_audits.json`, `08_temporal.csv`,
`08_target_encoding.csv`, `08_preprocessing.csv`
**Figure:** `reports/figures/08_leakage_impact.png`

The brief flags data leakage as the highest-risk failure mode for this
assignment. This phase does two things: it **proves** the shipped pipeline is
clean by machine check, and it **quantifies** what each kind of leakage would
have bought, by building the leaky version and measuring it.

Every experiment below is a matched pair or triple differing in exactly one
decision, scored on the same chronological holdout.

---

## 1. The audit is itself audited

An audit that always passes is worthless. Before any of the results below can
mean anything, the checker has to be shown capable of failing.

`src/tests/test_leakage_audit.py::test_audit_catches_a_planted_leak`
monkeypatches the point-in-time target encoder with a full-dataset one and
asserts the future-invariance audit flags it. It does. There is also a test
that `assert_no_time_overlap` raises on a deliberately inverted fold. Both are
controls for the clean results in §2 — without them, "0 features leak" would be
indistinguishable from "the audit does nothing."

The audit code lives in `src/leakage/`, deliberately separate from
`src/features/`. An audit that lives inside the code it audits tends to drift
into agreeing with it.

---

## 2. Static audits: is the shipped pipeline clean?

Three independent checks, run over a contiguous 120,000-row time slice
(`reports/results/08_static_audits.json`):

| Check | Method | Result |
|---|---|---|
| **Point-in-time re-derivation** | For 300 randomly sampled rows, recompute `uid_card_addr_prior_count` and `uid_card_addr_prior_amt_mean` by brute-force scanning the whole frame for same-entity rows with strictly smaller `TransactionDT`. | **300/300 exact**, both features |
| **Future invariance** | Shuffle every label and multiply every amount after a cut point, rebuild all features with the *same fitted statistics*, and compare rows before the cut. | **0 of 508 features changed** |
| **Fit-population dependence** | Refit the fit-on-train statistics on the full frame instead of the training block, and list which features move. | **44 features move — exactly the 44 declared class F/P+L. Zero unexpected.** |

The future-invariance check is the decisive one, and it is stronger than
re-derivation: re-derivation could agree with the pipeline because both share
the same wrong assumption, whereas corrupting the future and observing that
nothing in the past moves cannot be fooled that way.

**A note on how that audit was itself fixed.** Its first version *reported* five
leaking features, all of them false positives. It refitted the encoders on the
perturbed frame, so training rows that happened to fall after the cut shifted
the fitted parameters and every past row appeared to "change" — because the
*transformation* changed, not because it saw the future. Holding the fitted
statistics constant across the two runs isolates the question the audit is
actually asking. Fit-population dependence is a different concern, measured
separately in the third row above.

---

## 3. Temporal leakage — the single largest effect

Same model, same features, same 20% test size. **The only difference is which
rows are held out.**

| split | PR-AUC | lift | ROC-AUC | precision@1% | recall@1% |
|---|---:|---:|---:|---:|---:|
| chronological (correct) | **0.5945** | 17.3× | 0.9111 | 0.9188 | 0.2672 |
| random shuffled (leaky) | **0.8557** | 24.1× | 0.9744 | 0.9958 | 0.2808 |

**A random split inflates PR-AUC by 44%** (0.5945 → 0.8557) and drives
precision@1% to 99.6% — a queue that is essentially never wrong. Anyone
reporting the shuffled number would be claiming a near-perfect fraud detector.

The mechanism was established in Phase 2 §7 before any model was trained:
**91.7% of fraudulent transactions sit in proxy entities that carry more than
one fraud.** A compromised credential is not defrauded once, it is drained.
Under a shuffled split, transaction #4 of a nine-fraud burst lands in training
and transaction #7 lands in test — and `card1`, `addr1` and the entity
aggregates are all right there in the feature set. The model scores well by
recognising *the entity*, not by detecting fraud. It is a memorisation score.

Note that ROC-AUC moves far less than PR-AUC (0.911 → 0.974, +7%) — another
instance of the Phase 7 finding that ROC-AUC is the less sensitive instrument
on this problem. A team monitoring only ROC-AUC could ship a 44%-inflated
model and see a 7% blip.

**This is why the chronological split is a leakage-prevention mechanism first
and an evaluation choice second.** `assert_no_time_overlap` runs on every fold
set before use, and it is an assertion rather than a convention precisely
because the failure it prevents is worth 44% of the headline metric.

---

## 4. Feature leakage — the worked before/after example

Target encoding replaces a high-cardinality category with the historical fraud
rate of its level. It is the most useful encoding available for `card1`
(13,553 levels) and the most dangerous feature type in the project. Three
arms, identical in every other column:

| arm | PR-AUC | ROC-AUC | precision@1% | share of GBM gain from `*_te` |
|---|---:|---:|---:|---:|
| **expanding, 30-day label lag (shipped)** | **0.5945** | 0.9111 | 0.9188 | **0.51%** |
| expanding, zero lag (subtly leaky) | 0.6004 | 0.9182 | 0.9196 | 4.65% |
| full-dataset encoding (blatantly leaky) | 0.6205 | 0.9263 | 0.9281 | 7.17% |

### 4.1 The blatant version

`leaky_target_encode_full_data` computes each level's mean label over the
**entire** dataset — including the row's own label and every future row's
label. It is kept in `src/leakage/audit.py` rather than merely described,
because it is the version that gets written by accident.

It buys **+4.4% PR-AUC** (0.5945 → 0.6205) that would evaporate in production,
because at scoring time no future labels exist. A backtest reporting 0.6205
would be over-promising by that margin.

### 4.2 The subtle version, which is the more instructive one

The zero-lag arm is **already backward-looking**. Every row sees only rows with
a strictly smaller `TransactionDT`. It passes the future-invariance audit. By
the usual definition of temporal leakage it is clean.

It is still wrong. **A fraud label originates in a chargeback that arrives
weeks after the transaction.** At real scoring time you do not know whether
yesterday's transaction turned out to be fraudulent — that adjudication has not
happened yet. A zero-lag expanding encoding therefore consumes information
that will not exist when the model runs, and it is invisible to a naive
backtest because the row ordering looks impeccable.

The shipped encoder restricts each row to labels older than
`label_lag_days = 30`, approximating a chargeback settlement window. The cost
of that honesty is **1.0% PR-AUC** (0.6004 → 0.5945) — a real number, given up
deliberately.

### 4.3 The diagnostic that generalises

The rightmost column is the most useful thing in this table. It is the share of
total LightGBM gain attributable to the four `*_te` features:

- 0.51% with the lag — the model barely uses them
- 4.65% at zero lag — **9× more**
- 7.17% with full-dataset encoding — **14× more**

**The leakier a feature is, the more the model leans on it.** That is a
detectable signature. A target-encoded feature that dominates importance is
evidence of leakage, not evidence of a good feature, and this ratio is a cheap
standing check on any future pipeline change.

---

## 5. Preprocessing leakage — where the result was not what we expected

Imputer medians and scaler statistics fitted on the full frame versus on the
training block only, plain logistic regression so the effect is not absorbed by
a scale-invariant model:

| arm | PR-AUC | ROC-AUC |
|---|---:|---:|
| fitted on train only (correct) | **0.2451** | 0.8460 |
| fitted on train + test (leaky) | **0.1921** | 0.8449 |

**The leaky version is 22% *worse*.** This was not the expected direction and
it is worth reporting precisely because it complicates the tidy story.

The mechanism is the interaction between standardisation and regularisation.
Both arms rescale train and test consistently, so an *unregularised* linear
model would be nearly invariant to the choice. But this model carries L2
shrinkage at `C = 0.1`, and the penalty applies to coefficients on the
*standardised* scale. Changing the scaler changes which features get shrunk
and by how much. Because the test period's covariate distribution has drifted
away from the training period's (Phase 1 §2), folding it into the fitted
statistics moves the standardisation away from the distribution the
coefficients are actually being fitted against — and the regularisation then
penalises the wrong things.

**The methodological point is stronger than "leakage inflates results."** It is
that leakage makes results *unpredictable*. Here it happened to hurt; on a
different feature set or a different regulariser it would help. Either way the
number no longer measures what it claims to measure, and it will not reproduce.
"It would only make us look better" is not a defence for leaving it in, because
it is not reliably true.

This is why every preprocessing step in this project lives inside the model's
own `Pipeline` (`src/models/linear.py`, `src/models/nn.py`) and is refitted
inside every fold. Structurally, there is nowhere to put a prematurely-fitted
scaler.

---

## 6. Summary: what each control costs and buys

| Leakage type | Control applied | What it costs us | What it would have bought |
|---|---|---:|---:|
| **Temporal** | Chronological split + `assert_no_time_overlap` | — | **+44% PR-AUC** (0.5945 → 0.8557) |
| **Feature (blatant)** | Expanding, strictly-prior aggregation | — | +4.4% PR-AUC (0.5945 → 0.6205) |
| **Feature (subtle)** | 30-day label-availability lag | **−1.0% PR-AUC** | +1.0% PR-AUC (0.5945 → 0.6004) |
| **Preprocessing** | Fit inside each fold's `Pipeline` | — | −22% PR-AUC (it *hurt*) |

Total avoided overstatement, compounding the temporal and feature effects:
a leaky pipeline would report roughly **0.86 PR-AUC where the honest number is
0.59** — a 44% overstatement of a headline metric, on a model that would then
under-deliver from its first day in production.

## 7. Residual risks we could not eliminate

1. **The 30-day lag is an assumption, not a measurement.** The true chargeback
   distribution is unobserved. Thirty days is a plausible settlement window;
   the correct value could be 60 or 90, in which case the shipped encoding is
   still mildly optimistic.
2. **Entity overlap across the split is reduced, not removed.** An entity
   active in month 1 may still be active in month 5, so the holdout contains
   entities the model has seen. This is *correct* — at deployment you genuinely
   do know a card's history up to now — but it means the holdout is easier than
   scoring an entirely new population.
3. **The holdout abuts the training period.** The real task starts 30 days
   after training data ends (Phase 1 §2). Our numbers are optimistic relative
   to that gap.
4. **Label selection bias is unmeasurable from this data** (Phase 1 §4).
   Undetected fraud is labelled legitimate, so precision figures are lower
   bounds and no control in this pipeline can fix that.
5. **`has_identity_record` is legitimate but uncomfortable.** It is available
   at scoring time, so it is not leakage. But its 3.7× fraud-rate contrast
   likely reflects the incumbent system's own triage rather than the
   phenomenon, so a model leaning on it is partly imitating the prior policy.
   Flagged in the feature dictionary and tracked in Phase 9.
