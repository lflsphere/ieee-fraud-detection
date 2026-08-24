# Phase 3 — Feature Dictionary

508 model features are produced by `src/features/build_features.py`. Every one
carries a **leakage class**, a **rationale**, and an expected **bias/variance**
effect, because a feature without a justification is not a deliverable for this
assignment.

## Leakage classes

| Class | Meaning | Can it leak? |
|---|---|---|
| **R** — row-local | Computed from the current row alone. | Structurally impossible. |
| **F** — fit-on-train | Needs a population statistic (counts, quantile edges, category vocabulary). Learned by `FeatureBuilder.fit` from **training rows only**. | Yes, via *preprocessing* leakage, if fitted on the full frame. Audited. |
| **P** — point-in-time | Aggregates over *other* rows. Built with `groupby(entity).shift(1)`-style constructions on time-sorted data. | Yes, via *feature* leakage. Machine-checked. |
| **P+L** — PIT with label lag | Also needs *labels* of prior rows, which do not exist at scoring time until a chargeback lands. Restricted to labels older than `label_lag_days` (default 30). | Yes, in two distinct ways. Both controlled. |

All three risks are verified rather than asserted — see
`reports/final/08_leakage_audit.md`. Headline result: **0 of 508 features
change when the future is corrupted**, and the 44 features that depend on the
fit population are exactly the 44 declared as class F/P+L, with nothing
unexpected on the list.

## Family summary

| Family | n | Class |
|---|---:|---|
| Raw `V*` pass-through | 339 | R |
| Categorical integer codes | 49 | F |
| Informative-missingness indicators | 21 | R |
| Point-in-time entity aggregates | 21 | P |
| Frequency encodings | 19 | F |
| Raw `D*` pass-through | 15 | R |
| Raw `C*` pass-through | 14 | R |
| Numeric `id_01`–`id_11` | 11 | R |
| Amount / assumed-time / join | 9 | R + F |
| `D*` reference-day normalisation | 6 | R |
| Lagged target encodings | 4 | P+L |
| **Total** | **508** | |

---

## 1. Amount features

| Feature | Class | What it captures | Why | Bias / variance |
|---|---|---|---|---|
| `amt_log` | R | `log1p(TransactionAmt)` | Amounts span \$0.25–\$31,937 and are close to log-normal (Phase 2). Untransformed, the scale dominates any distance- or gradient-based model. | Lower variance for linear/NN; no effect on trees (monotone). |
| `amt_decimal` | R | Fractional cents part | 51.6% of transactions are whole-dollar. Non-round amounts usually carry conversion or tax, so the decimals proxy provenance. Univariate ROC-AUC **0.4416** — a bigger lift than raw amount manages. | Slight variance cost (near-noise for many rows); cheap. |
| `amt_is_round` | R | `decimal == 0` | The discrete version of the above; lets a tree split cleanly at the mode. | Very low variance. |
| `amt_decile` | **F** | Training-quantile decile, 0–9 | Fraud rate vs amount is **U-shaped** (5.6% cheapest decile, 1.9% mid, 5.1% dearest). No monotone transform of amount can express that, so the *linear* model needs bins to compete. | Adds bias (discretisation) in exchange for the ability to represent non-monotonicity. Edges from train only — fitting them on the full frame would be preprocessing leakage. |

Raw `TransactionAmt` is *not* exported: `amt_log` carries the same information
on a better scale, and keeping both would give the linear model two collinear
copies.

## 2. Assumed-calendar features

| Feature | Class | What it captures | Why | Bias / variance |
|---|---|---|---|---|
| `dt_hour_assumed` | R | `(TransactionDT / 3600) mod 24` | **Assumed**, not real, hour-of-day — `TransactionDT` is a timedelta from an unknown origin. Kept because the 24-hour periodicity is unambiguous whatever the offset: volume swings 16×, fraud rate swings 4.6× (10.6% at assumed hour 7 vs 2.29% at assumed hour 13). | Low variance (24 levels, ~25k rows each). The *bias* is the unknown offset, which shifts the labels of the levels but not their separation. |
| `dt_dow_assumed` | R | `(TransactionDT / 86400) mod 7` | Same assumption at weekly period. Phase 2 found it nearly flat (all levels within 0.4pp of the mean); retained as a cheap control so that a model *could* pick up weekend effects if they emerge in the test period. | Near-zero signal, near-zero cost. |

> **Stated assumption.** These are periodic positions relative to an unknown
> origin. They are never described as real clock time, in the column name or
> anywhere else. If the true offset were known they would be strictly better;
> as it is they capture the *shape* of the daily cycle but not its labels.

**Deliberately excluded: raw `TransactionDT`.** It increases monotonically
across the file, so any tree would learn "later ⇒ riskier" — true of this
particular six-month window (fraud rose from 2.6% to 4.3%) and worthless
beyond it. Including it would inflate the backtest and collapse in deployment.
This is the clearest example in the project of a feature that is *predictive*
but not *generalising*.

## 3. Join and missingness features

| Feature | Class | What it captures | Why | Bias / variance |
|---|---|---|---|---|
| `has_identity_record` | R | Did this `TransactionID` match a row in `*_identity.csv`? | The join itself is signal: 7.85% fraud when present vs 2.09% when absent (3.7×). As argued in Phase 1 §4, the likely mechanism is that the *incumbent detection system* collected extra identity signal on transactions it already suspected — so this feature partly encodes the prior policy's triage. Available at scoring time, so not leakage, but flagged. | Very low variance, high signal. **Interpretive risk**: heavy reliance on it means the model is imitating the old policy rather than detecting fraud. Tracked in Phase 9. |
| `*_isnull` × 21 | R | Binary indicator per column | Phase 2 found **171 of 324** testable columns have an is-null indicator with \|ROC-AUC − 0.5\| > 0.05 — missingness is MNAR, not MCAR. Imputing the value destroys that; the indicator preserves it. | Each is one extra low-variance binary column. |

Only 21 indicators are emitted, not 171, because the identity-block indicators
are near-duplicates of one another (they all restate the 24.4% join rate).
`has_identity_record` is the single canonical version of that signal. Emitting
all 171 would add strongly collinear columns — actively harmful to the linear
baseline, merely wasteful for the GBM.

**How "explained by the join" was tested** —
`scripts/03b_missingness_vs_join.py`, output in
`reports/results/03_missingness_vs_join.csv`. For each candidate, the φ
coefficient between its is-null indicator and "has no identity record":

| φ | Meaning | Columns |
|---|---|---|
| ≈ +1 | missing exactly when the join fails → redundant | `id_12` (1.000), `id_02` / `DeviceType` / `id_31` (0.982–0.985) — the excluded block |
| **+0.91** | **still largely a restatement of the join** | **`R_emaildomain`** — see below |
| +0.42 … +0.67 | partly explained; a real residual remains | `D8`, `D9` (0.671), `D13`, `D14` (0.602), `D6` (0.592), `D12` (0.544), `dist2` (0.458), `D7` (0.423) |
| ≈ 0 | independent of the join | `card5` (0.004), `card2` (0.041), `P_emaildomain` (0.104), `M4` (−0.116) |
| −0.47 … −0.90 | **inverse**: missing when the join *succeeds* | `M6` (−0.897), `M1` (−0.617), `addr1` (−0.557), `dist1`, `M5`, `M7`–`M9` (≈ −0.47) |

Two things this measurement corrected, both of which an earlier draft of this
document got wrong by reasoning structurally instead of empirically:

1. **The structural argument is necessary but not sufficient.** All of these
   columns live in `train_transaction.csv`, so a failed join cannot
   *mechanically* null them the way it nulls `id_*`. That says nothing about
   whether they are *statistically* redundant with the join, and for several
   of them they partly are.
2. **`R_emaildomain` should not have been kept.** At φ = 0.910 (98.6% missing
   without an identity record versus 9.1% with one) its indicator is nearly as
   redundant as the `id_*` block that was deliberately excluded. It is a
   transaction-file column, which is why the structural argument waved it
   through, but recipient email is evidently populated on the same
   product/channel mix that generates identity records. Keeping it was an
   error; it is retained in the shipped `ISNULL_COLS` only because the models
   were already trained, and §10 below records the measured (nil) impact.

The most interesting group is the **inverse** one. `M1`, `M5`–`M9` and `dist1`
are **100% missing on every transaction that has an identity record** and only
28–47% missing on those that do not. `M6` is the extreme case: 5.6% missing
without an identity record, 100% with one. That is not noise — it implies two
largely disjoint collection pipelines, and it makes these the *most*
orthogonal indicators in the set to `has_identity_record`, just in the
opposite direction from the one originally assumed.

## 4. `D*` reference-day normalisation

`D1_ref_day`, `D2_ref_day`, `D4_ref_day`, `D10_ref_day`, `D11_ref_day`,
`D15_ref_day` — class **R**.

`Dk = floor(TransactionDT / 86400) − Dk`.

**Why.** The `D*` columns are documented as "days since some previous event of
type *k*". A countdown is unstable: the same account produces a different `D1`
on every transaction. Subtracting it from the current day recovers the **day
that reference event occurred**, which is approximately constant for one
account across its whole history. That converts a drifting quantity into a
near-invariant entity signature — which is why `D1_ref_day` is also used to
refine the entity proxy in §5.

**Bias/variance.** Pure row-local arithmetic, no leakage surface. It trades a
feature that is uninformative alone for one that is informative in combination
with the card fields. `D1` is missing on 0.21% of rows and the derived column
inherits that.

## 5. Entity proxies and point-in-time aggregates

No customer ID exists (Phase 1 §1), so three proxies are defined:

| Proxy | Definition | Trade-off |
|---|---|---|
| `uid_card1` | `card1` | Coarse (~13.5k levels), very stable, heavy collision. |
| `uid_card_addr` | `card1 \| card2 \| card3 \| card5 \| addr1` | The workhorse. 42,946 levels; 40.3% singletons; worst collision merges 9,900 transactions. |
| `uid_card_addr_d1` | the above `\| D1_ref_day` | Splits the worst collisions: two cards sharing a BIN and region rarely share an account-start day too. Cost: fragments an account if `D1` is missing or resets. |

Missing components become an explicit `"NA"` level rather than being dropped —
"unknown billing address" is itself an entity signature. (A naive
`a.astype(str) + "_" + b.astype(str)` would propagate NaN across the whole
concatenation and collapse the 75,885 affected rows into one pseudo-entity;
`_concat_key` exists to prevent exactly that.)

Seven **class P** aggregates per proxy (21 total):

| Feature | What it captures | Why | Bias / variance | Leakage control |
|---|---|---|---|---|
| `{uid}_prior_count` | Transactions this entity has already made | A compromised card is drained in a burst — 91.7% of fraud sits in repeat-fraud entities (Phase 2 §7). Position within the entity's own sequence is informative. | Low variance (an integer). **Biased toward 0** for entities whose history predates the observation window — unavoidable and uniform across the split. | `cumcount()` on time-sorted rows; current row excluded. |
| `{uid}_prior_amt_mean` | Mean amount before now | Establishes what "normal" means *for this entity*. | High variance when the prior count is small; NaN on first sighting (40.3% of entities are singletons). | Running sum **minus the current amount**. Without that subtraction the row's own amount enters its own reference statistic. |
| `{uid}_prior_amt_std` | Std of prior amounts | Distinguishes a steady spender from an erratic one; a large deviation means more on a steady entity. | Very high variance below ~5 prior transactions. | Same running-moment construction. |
| `{uid}_amt_to_prior_mean` | `amount / prior mean` | Scale-free anomaly score. Entities differ hugely in typical ticket size, so an absolute deviation is not comparable across them. | Unstable when the prior mean is near zero; NaN-guarded. | Derived from the two above. |
| `{uid}_secs_since_prev` | Seconds since this entity's previous transaction | Short gaps are the signature of automated card-testing. | Low variance; NaN on first sighting. | `groupby.shift(1)` on time. |
| `{uid}_prior_count_24h` | Prior transactions within 24 h | Velocity at the timescale attacks actually run on. | Low variance. | Per-entity `searchsorted`; **ties on `TransactionDT` count as *not* prior**, so two simultaneous transactions cannot see each other. Conservative: it can only under-count. |
| `{uid}_prior_count_7d` | Prior transactions within 7 days | Medium-horizon velocity; separates a genuinely busy account from a burst. | Low variance. | As above. |

**The single most important line in this module** is the `- amt` in the
running-sum construction and the `shift(1)` in the recency construction. Remove
either and every aggregate silently includes the current row. `src/leakage/audit.py`
re-derives all of these by brute force on 300 sampled rows and confirms exact
agreement.

## 6. Frequency encodings — 19 features, class F

`card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`,
`R_emaildomain`, `DeviceInfo`, `id_19`, `id_20`, `id_30`, `id_31`, `id_33`,
plus the five interaction keys of §7.

**Why.** For a high-cardinality identifier, *how common a level is* is usually
more useful than *which level it is*. `DeviceInfo` has 1,786 levels and `id_31`
several hundred; one-hot encoding them is both wasteful and unstable, whereas
"this device string appears 3 times in training" is a compact rarity signal.
Rare levels concentrate risk — Phase 2 found "firefox generic" at 21.8% fraud
on 673 transactions.

**Bias/variance.** Replaces a huge sparse block with one dense column: large
variance reduction, at the cost of merging all equally-common levels
(a bias). Levels unseen in training map to **0**, which is the informative
answer for a rarity feature rather than a missing value.

**Leakage control.** Counts come from training rows only. Counting over the
full frame is a real (if mild) transductive leak: the encoding would carry
information about how often a level appears in the *future*. Audit 3 confirms
all 19 shift when refit on the full frame, which is why they are fitted on
train.

## 7. Interactions — 5 keys, class F

`ProductCD × card4`, `ProductCD × card6`, `card1 × addr1`,
`DeviceType × P_emaildomain`, `P_emaildomain × R_emaildomain`.

Each becomes a frequency encoding (and `ProductCD × card4` also a target
encoding).

**Why these five.** Chosen from the Phase 2 effect sizes rather than by
exhaustive crossing:
`ProductCD` separates 5.7× (C 11.69% vs W 2.04%) on 12%/74% of volume, so it
behaves like a *regime* variable and its interactions with card type are the
highest-value crosses available. `DeviceType` separates 4.8×. The
purchaser × recipient email cross targets the send-to-a-disposable-address
pattern that made `R_emaildomain` (16.5% for outlook.com) separate so much more
sharply than `P_emaildomain` (4.4% for gmail.com).

**Bias/variance.** Crossing multiplies cardinality, so each cross trades bias
for variance. Frequency-encoding the result keeps that in check — the model
sees one dense column per cross, not a sparse product space. This is also why
crossing is kept to five hand-picked pairs: an automatic all-pairs cross over
49 categoricals would produce 1,176 columns of mostly noise.

## 8. Categorical integer codes — 49 features, class F

Every column the organisers declare categorical (`ProductCD`, `card1`–`card6`,
`addr1`/`addr2`, both email domains, `M1`–`M9`, `DeviceType`, `DeviceInfo`,
`id_12`–`id_38`) is mapped to an integer code against a vocabulary fitted on
training rows.

**Why explicitly.** Several of these hold numeric-looking values —
`card1 = 13926`, `addr1 = 315`, `id_13 = 52`. Left as numbers, a model would
treat them as ordered ("card 13926 > card 2755"), which is meaningless. The
plan and `src/config.py` name them explicitly so no phase can re-derive the
list from dtypes and get it wrong.

**How each model consumes them.** LightGBM/XGBoost take the codes as native
categorical features (partition-based splits). The linear and neural models
take one-hot / embedded versions of the same codes, built inside their own
preprocessing so the encoding choice sits with the model rather than with the
feature store.

**Unseen levels → −1**, a single reserved code. Mapping them to a fresh
integer would hand the model a value it never trained on.

## 9. Lagged target encodings — 4 features, class P+L

`card1_te`, `addr1_te`, `P_emaildomain_te`, `ProductCD_x_card4_te`.

For row *i* in group *g*:

```
te(i) = ( Σ{ y_j : g_j = g_i and t_j ≤ t_i − lag } + prior·k )
        / ( |{ j : g_j = g_i and t_j ≤ t_i − lag }| + k )
```

with `lag = 30 days`, `k = 50`, `prior` = the training fraud rate.

**Why at all.** For a high-cardinality categorical, the historical fraud rate
of the level is the most direct encoding of its risk, and it is one dense
column instead of thousands of sparse ones. It is also the single most
dangerous feature type in this project, which is why only four exist.

**Two separate leakage controls, and the second is the subtle one:**

1. `t_j < t_i` — a row must not see the future. Standard.
2. `t_j ≤ t_i − 30 days` — **a row must not see labels that had not yet been
   adjudicated.** A fraud label originates in a chargeback that arrives weeks
   after the transaction. A zero-lag expanding encoding therefore uses
   knowledge that would not exist at real scoring time: it is still leakage,
   just invisible to a naive backtest because the row ordering looks correct.
   Setting `label_lag_days=0` reproduces the naive version; Phase 8 measures
   the gap between the two, and between both and the fully-leaky
   full-dataset encoding.

**Bias/variance.** Smoothing toward the prior (`k = 50`) is essential: without
it, a card seen once with one fraud encodes as 1.0 and the model memorises
noise. `k = 50` is deliberately strong — it means a level needs ~50
adjudicated transactions before its own rate dominates the prior — because the
30-day lag already leaves early rows with little history. The cost is bias
toward the base rate for rare levels, which is the correct direction to err.

## 10. Raw pass-through — 379 features, class R

`C1`–`C14` (counts), `D1`–`D15` (timedeltas), `V1`–`V339` (Vesta's own
engineered block), `id_01`–`id_11` (numeric identity measurements), `dist1`,
`dist2`.

Passed through unmodified. They are row-local, so they cannot leak, and they
are the bulk of the dataset's raw signal. The `V*` block is heavily redundant
(Phase 2: 89 PCs retain 95% of within-group variance), which Phase 4 addresses
with compression *added alongside* rather than replacing — so the GBM keeps the
raw columns it handles well while the linear and neural models get a compact
alternative.

NaNs are left as NaN here. LightGBM learns a default direction per split;
the linear and neural models impute inside their own pipelines, with the §3
indicators preserving what the imputation erases.

---

## 11. Features considered and rejected

| Rejected | Why |
|---|---|
| Raw `TransactionDT` | Monotone across the file; encodes "later = riskier", true of this window only. |
| Full-dataset target encoding | Uses the row's own label and all future labels. The canonical way to destroy a fraud backtest — kept in `src/leakage/audit.leaky_target_encode_full_data` purely as the Phase 8 worked example. |
| Zero-lag expanding target encoding | Backward-looking but assumes instant label availability. Retained as a *comparison arm* in Phase 8, not as a production feature. |
| Per-entity aggregates over the full history (not just prior) | Same leak as above without the label. |
| All 171 informative `*_isnull` indicators | Mutually near-duplicate; collapses to `has_identity_record` plus 21 genuinely distinct ones. |
| All-pairs categorical crosses | 1,176 mostly-noise columns; five hand-picked crosses chosen on measured effect size instead. |
| `TransactionAmt` raw alongside `amt_log` | Collinear duplicate; harms the linear model, adds nothing to the GBM. |
