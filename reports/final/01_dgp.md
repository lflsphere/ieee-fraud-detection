# Phase 1 — The Data Generating Process

*All figures in this document are computed by `scripts/01_dgp_evidence.py` and
persisted to `reports/results/01_dgp_evidence.json`. Numbers refer to the
labelled training file (`train_transaction.csv` ⟕ `train_identity.csv`),
590,540 rows spanning `TransactionDT` days 1–182.*

---

## 1. Unit of analysis

**One row = one attempted card transaction**, keyed by `TransactionID`.

The unit we would *like* to model is the account or the cardholder, because
fraud is a property of a compromised credential, not of an isolated payment.
The data gives us no such key. What it gives us instead is a set of **noisy
proxies for a latent entity**: `card1`–`card6`, `addr1`/`addr2`, and — for the
24.4% of rows that have an identity record — `DeviceInfo` and the `id_*` block.

How noisy? Concatenating `card1|card2|card3|card5|addr1` (treating a missing component as
its own `NA` level — 75,885 rows have at least one) yields **42,946 distinct
proxy entities** across 590,540 transactions:

| Statistic | Value |
|---|---:|
| Distinct proxy entities | 42,946 |
| Median transactions per entity | 2 |
| Mean transactions per entity | 13.8 |
| 95th percentile | 42 |
| Maximum | 9,900 |
| Entities seen exactly once | 40.3% |

That single entity with 9,900 transactions is almost certainly a collision —
several real cards sharing an issuer/BIN/region signature — not one very busy
cardholder. So the proxy is simultaneously **too coarse** (it merges distinct
people) and **too fine** (a cardholder who moves house gets a new `addr1` and
thus a new identity). Every entity-level feature built in Phase 3 inherits
this error, and the feature dictionary says so explicitly for each one.

**Why this matters beyond feature design:** fraud is *concentrated* within
entities. Only 8.3% of proxy entities ever carry a fraud label, but **91.7% of
all fraudulent transactions sit in entities that carry more than one fraud**.
Fraud arrives in bursts on a compromised credential. A randomly shuffled
train/test split would therefore place transactions from the *same burst* on
both sides of the split, and a model could score well by memorising the
entity rather than by recognising fraud. This is the single strongest
empirical argument for the chronological split (Phases 7–8), and it is a
property of the DGP, not of the evaluation protocol.

---

## 2. The generating process is adversarial and non-stationary

This must be argued rather than assumed, so here is the argument and the
evidence for it.

**The claim.** Legitimate transactions are produced by a large population of
consumers whose behaviour is *approximately* stationary over a six-month
window. Fraudulent transactions are produced by a small population of
attackers who are **optimising against the very detection system that
generates our labels**. When a tactic starts getting declined, attackers stop
using it; when a gap opens, they pour through it. The fraud-generating
distribution is therefore a *response function* to the defender's policy, not
a fixed distribution being sampled.

**The evidence.** Weekly fraud rate over the 26 weeks of training data ranges
from **1.85% to 5.06%** — a 2.7× swing — and it is not noise around a
constant:

| Period | Fraud rate |
|---|---:|
| Weeks 1–4 | 2.81%, 2.61%, 2.55%, 1.85% |
| Weeks 23–26 | 3.25%, 3.67%, 4.16%, 4.32% |

Overall fraud rate is 3.499% (20,663 of 590,540). Transaction volume is also
strongly non-stationary — weekly counts range from 2,754 to 37,251, a 13.5×
swing. Volume drift on its own would be unremarkable (seasonality, portfolio
growth), but volume and fraud rate drifting *together and in different
directions across sub-populations* is what makes a static model decay.

**Three consequences we accept as design constraints:**

1. **The i.i.d. assumption underlying random cross-validation is false here.**
   Rows are neither independent (they cluster by entity and by attack burst)
   nor identically distributed (the class prior moves by a factor of 2.7).
2. **Model performance measured on the past overstates performance on the
   future.** Phase 7 measures generalisation to *later* transactions
   specifically, and reports per-fold metrics rather than only a mean, so
   drift is visible instead of averaged away.
3. **Any deployed model has a shelf life.** The competition's own test period
   begins at day 213 — **a 30-day gap after training data ends at day 182** —
   and runs to day 395. The intended task is explicitly "predict a month
   ahead, then keep predicting for six more months," which is a forecasting
   problem dressed as a classification problem.

---

## 3. Latent variables

The observed columns are shadows of variables we never see. Naming them
matters because it tells us which features are proxies (and therefore
unstable) rather than measurements.

| Latent variable | Observed proxies | Why it is only a proxy |
|---|---|---|
| **True cardholder / account** | `card1`–`card6`, `addr1`, `addr2` | Collides across people (9,900-transaction "entity"); fragments when address changes. `card1` has 13,553 levels, `addr1` only 332 — these are issuer/BIN/region codes, not identities. |
| **Physical device / session** | `DeviceType`, `DeviceInfo` (1,786 levels), `id_30`–`id_33` | Present for only 24.4% of rows; browser/OS strings are shared by millions of devices. |
| **Merchant and merchant risk** | `ProductCD`, `R_emaildomain`, parts of the `V*` block | No merchant ID exists at all. `ProductCD` has 5 levels for what is certainly a heterogeneous merchant population. |
| **Account tenure / relationship age** | `D1`–`D15` (timedeltas, `D1` ranges 0–640 days) | Vesta describes these as "days since previous event of type *k*." Which event is undisclosed, so tenure is inferred, not measured. |
| **Vesta's own internal risk features** | `V1`–`V339` | 339 anonymised, heavily correlated engineered features — a *model's* view of the transaction, already one transformation away from raw behaviour. |
| **The incumbent detection policy** | *nothing* | Not observed at all. See §4 — this is the most consequential omission. |

---

## 4. The label is a policy output, not ground truth

`isFraud` is defined by the organisers as a reported chargeback or a
confirmed-fraud investigation outcome, with the associated user account and
card then retro-labelled. Two biases follow directly, and neither is fixable
with the data at hand:

**(a) Selection bias — undetected fraud is labelled legitimate.** Fraud that
was never charged back, never disputed, or disputed after the labelling window
appears in the data as `isFraud = 0`. The negative class is therefore
"legitimate *or* undetected fraud." The measured 3.499% base rate is a lower
bound on true fraud incidence, and every precision figure in Phase 7 is
correspondingly a lower bound too: some "false positives" are real fraud the
incumbent system missed.

**(b) Feedback bias — the labels encode the incumbent system's blind spots.**
Transactions the incumbent system blocked outright may never appear in this
file at all. What we are learning is not "what fraud looks like" but **"what
fraud looked like *conditional on getting past the filters that were live in
2017–2018 and then being caught afterwards.*"** A model trained on this is,
strictly, an *imitator of the existing detection policy's residual*, and
deploying it creates a feedback loop: it declines what the old system declined,
so those transactions never generate new labels, so the blind spot is never
observed.

**Practical implication we act on:** we do not treat a high false-positive
count as automatically bad in Phase 7. We report performance at a fixed
*review-budget* operating point (top 1% of scores, `REVIEW_BUDGET_FRACTION` in
`src/config.py`) because that is a decision the business can actually take,
and because ranking quality is robust to label noise in a way that a
calibrated absolute probability is not.

**A visible symptom of (b).** Fraud rate by whether an identity record exists
at all:

| `has_identity_record` | n | Fraud rate |
|---|---:|---:|
| 0 (no identity row) | 446,307 | **2.09%** |
| 1 (identity row present) | 144,233 | **7.85%** |

A 3.7× difference. The plausible reading is not that having a device
fingerprint makes you a fraudster; it is that **the incumbent system collected
extra identity signals precisely on the transactions it already found
suspicious.** The presence of the identity record is partly an artefact of the
prior policy's own triage. This makes `has_identity_record` genuinely
predictive *and* a textbook example of a feature whose predictive power comes
from the labelling process rather than from the phenomenon. We keep it (it is
available at scoring time, so it is not leakage) but we flag it in the feature
dictionary and treat any large reliance on it as a warning sign.

---

## 5. This is a predictive model, not a causal one

We make **no causal claim**. Nothing here supports a statement of the form
"issuing on `ProductCD = C` *causes* fraud," even though the contrast is
dramatic:

| `ProductCD` | n | Fraud rate |
|---|---:|---:|
| C | 68,519 | **11.69%** |
| S | 11,628 | 5.90% |
| H | 33,024 | 4.77% |
| R | 37,699 | 3.78% |
| W | 439,670 | **2.04%** |

`ProductCD = C` is 5.7× riskier than `W`, but the product code is confounded
with everything about how that product is sold, who buys it, what the ticket
size is, and how attackers monetise it. An intervention ("stop selling C")
would not move fraud by the amount this table suggests; attackers would
substitute. The same holds for the device contrast (mobile 10.17%, desktop
6.52%, no device record 2.10%).

The target of inference is: **given the features observable at authorisation
time, how much does this transaction resemble transactions that were
historically labelled fraudulent?** That is a ranking problem, and it is
sufficient for the deployment decision (route to review or not).

---

## 6. What the DGP implies for the rest of the pipeline

| DGP property | Where it constrains the build |
|---|---|
| Fraud clusters within proxy entities (91.7% of fraud in repeat-fraud entities) | Entity aggregation features must be **strictly backward-looking** (Phase 3); random CV is disallowed (Phase 7). |
| Class prior drifts 1.85% → 5.06% | Report per-fold metrics, not just the mean; prefer PR-AUC, which is defined against the prevailing base rate, and interpret it *relative to that base rate* per fold (Phase 7). |
| Labels are a policy output with selection bias | Evaluate at a fixed review budget; do not over-interpret calibration; do not claim to measure "true fraud rate" (Phase 7, Phase 9). |
| `TransactionDT` is a timedelta from an unknown origin | Use it for ordering and elapsed time only. Any hour-of-day/day-of-week feature is an **assumption** and is labelled as one (Phase 3). |
| 75.6% of rows have no identity record, non-randomly | Left join, never inner; `has_identity_record` as an explicit feature; missingness treated as MNAR and encoded, not imputed away (Phases 2–3). |
| Only 339 anonymised `V*` features, heavily redundant | Unsupervised compression is a legitimate representation step, not just a curiosity (Phase 4). |

---

## 7. Assumptions, stated explicitly

1. **`TransactionDT` is monotone in real time** and its units are seconds.
   The observed range (1–182 days at 86,400 units/day, with volume patterns
   showing a clean 7-cycle) supports this. We rely on the *ordering* only.
2. **No true calendar alignment is recoverable.** We do derive
   `DT_hour = (TransactionDT / 3600) mod 24` and `DT_dow`, but they are
   *assumed* alignments relative to an unknown origin, labelled as such
   everywhere they appear. They are still useful: whatever the offset, the
   modular structure is real and periodic.
3. **`card1|card2|card3|card5|addr1` approximates a stable entity.** Known to
   be wrong at both ends (collisions and fragmentation); accepted because no
   better key exists, and quantified in §1 so the error is visible.
4. **Rows within the file are complete** — no transactions were dropped from
   the observation window in a way that correlates with the target. This is
   unverifiable from the data and is the assumption we would most want to
   check with the data owner.
5. **The label window is closed** for the training period — i.e. chargebacks
   for late-period transactions have had time to arrive. If not, fraud in the
   most recent weeks is *under*-labelled, which would bias the chronological
   test block downward. The observed *rise* in fraud rate toward the end of
   the period argues against a severe version of this problem, but does not
   rule it out.

---

## 8. One-line summary

Transactions are generated by a mostly-stationary consumer population; fraud
labels are generated by an adversary who adapts, filtered through a detection
policy we cannot observe. We therefore build a **ranking model over a
chronologically ordered, entity-clustered, drifting stream**, and we treat
every design choice in the rest of this project as an answer to one of those
three words: *ranking*, *chronological*, *drifting*.
