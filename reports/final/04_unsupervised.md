# Phase 4 — Unsupervised Representation of the `V*` Block

**Code:** `src/unsupervised/cluster_v_features.py`, driver
`scripts/04_cluster_v_block.py`
**Figures:** `reports/figures/04_cluster_stability.png`,
`reports/figures/04_cluster_fraud_alignment.png`
**Tables:** `reports/results/04_cluster_stability.csv`,
`reports/results/04_cluster_alignment.csv`

---

## 1. Why compress at all

This step exists because of a measured property of the data, not because
clustering is on the syllabus. Phase 2 established two facts about the 339
anonymised `V*` columns:

* they fall into **15 groups that share an exact missingness pattern** —
  columns computed by the same upstream Vesta process go missing together, so
  hashing each column's null-mask recovers the block structure with no
  threshold and no hyperparameter; and
* **within** those groups they are severely collinear. In `V1–V11` the first
  principal component alone explains 94.2% of variance and 61% of column pairs
  correlate above |r| = 0.9.

Collinearity is not equally costly to all three model families. Gradient
boosting is essentially immune — it splits on one of a set of near-duplicate
columns and ignores the rest. The linear baseline suffers directly (unstable,
mutually-cancelling coefficients) and the neural network suffers indirectly
(first-layer capacity spent re-learning the same direction 11 times). So the
compression is built as an **additional representation**, evaluated as an A/B
in Phase 7, rather than assumed to be an improvement.

## 2. Method

**Per-group PCA.** Each of the 15 missingness groups is imputed with its
*training* median, standardised, and reduced to the number of components
needed for 95% of within-group variance, capped at 12 per group. Grouping
before PCA (rather than one PCA over all 339 columns) matters: a global PCA
would have to impute across the ~76%-missing identity-linked groups and the
~0%-missing groups together, and the leading components would then encode
*missingness* rather than behaviour.

Result: **339 columns → 126 components.** Group-level detail:

| Group | Range | Cols | PCs | Var retained | Missing rate |
|---:|---|---:|---:|---:|---:|
| 00 | V1–V11 | 11 | 7 | 97.4% | 47.3% |
| 01 | V12–V34 | 23 | 9 | 95.8% | 12.9% |
| 02 | V35–V52 | 18 | 9 | 95.2% | 28.6% |
| 03 | V53–V74 | 22 | 9 | 95.1% | 13.1% |
| 04 | V75–V94 | 20 | 9 | 95.1% | 15.1% |
| 05 | V95–V137 | 43 | 12 | 86.6% | 0.05% |
| 06 | V138–V163 | 18 | 6 | 95.4% | 86.1% |
| 07 | V143–V166 | 11 | 3 | 96.0% | 86.1% |
| 08 | V167–V216 | 31 | 8 | 96.4% | 76.4% |
| 09 | V169–V210 | 19 | 10 | 96.6% | 76.3% |
| 10 | V217–V278 | 46 | 12 | 95.2% | 77.9% |
| 11 | V220–V272 | 16 | 6 | 96.3% | 76.1% |
| 12 | V279–V321 | 32 | 12 | 95.0% | 0.00% |
| 13 | V281–V315 | 11 | 7 | 96.5% | 0.21% |
| 14 | V322–V339 | 18 | 7 | 96.1% | 86.1% |

Only group 05 (`V95–V137`, the fully-populated block) fails to reach 95% within
the 12-component cap, at 86.6% — it is the least redundant part of the block,
which is consistent with it also having the lowest mean pairwise correlation in
Phase 2 (0.277).

**K-means on the concatenated components**, producing three features:
`v_cluster` (the assignment), `v_cluster_dist` (distance to the assigned
centroid — "how typical is this row of its own cluster") and
`v_cluster_margin` (gap between nearest and second-nearest centroid — low
margin means the assignment is ambiguous, so a model can discount it).

**Leakage discipline.** The imputer medians, the scaler, the PCA rotations and
the K-means centroids are all fitted on the **training block only**. Fitting
PCA on the full frame would encode the covariance structure of the future into
the rotation — a preprocessing leak that is easy to miss because no label is
involved. The label is used **nowhere** in this module; §4 looks at fraud rates
only *after* the partition is already fixed.

## 3. Choosing `k` on stability, not on a round number

`k` was selected by a sweep over {4, 6, 8, 10, 12, 16}, refitting under 5
random seeds per `k` on a common 60,000-row subsample of the training block and
measuring the **mean pairwise adjusted Rand index** between seed pairs. A
partition that only appears under one initialisation is an artefact, and
feeding it downstream as a feature would inject seed noise into the model.

| k | mean ARI | worst pair | Silhouette | Inertia |
|---:|---:|---:|---:|---:|
| 4 | 0.991 | 0.982 | 0.457 | 15,130,682 |
| **6** | **0.945** | **0.869** | **0.477** | 13,477,752 |
| 8 | **0.560** | **0.282** | 0.477 | 12,326,076 |
| 10 | 0.972 | 0.950 | 0.136 | 11,384,395 |
| 12 | 0.956 | 0.928 | 0.162 | 10,519,227 |
| 16 | 0.812 | 0.645 | 0.177 | 9,605,743 |

**This sweep changed the design.** The initial default was `k = 8`, and `k = 8`
turns out to be the *least* reproducible setting in the whole grid — mean ARI
0.56, with one seed pair agreeing at only 0.28. Inspecting the k=8 solution
explains why: it splits off two degenerate clusters of 81 and 2 rows, and which
rows land in them flips with the seed. Silhouette also collapses from ~0.47 to
~0.14 between k=8 and k=10, so the compact structure genuinely runs out at
around six to eight clusters.

`k = 6` was adopted: ARI 0.945 (worst pair 0.869) with the joint-best
silhouette in the grid. `k = 4` is more stable still but coarser, and the
alignment table below shows the extra resolution at k=6 is where the useful
separation lives.

## 4. Does the unsupervised partition align with fraud?

Computed **after** the clustering was fixed. The label was not used to choose
`k`, to select components, or to fit anything.

| `v_cluster` | n | Fraud rate | Lift vs 3.50% base |
|---:|---:|---:|---:|
| 5 | 4,853 | **40.80%** | **11.7×** |
| 1 | 66,615 | 9.54% | 2.73× |
| 2 | 1,661 | 2.71% | 0.77× |
| 0 | 508,656 | 2.41% | 0.69× |
| 3 | 8,670 | 0.44% | 0.13× |
| 4 | 85 | 0.00% | 0.00× |

This is a strong result. A partition built with **no access to the label at
all** isolates a cluster of 4,853 transactions — 0.8% of the data — in which
**two of every five transactions are fraudulent**, an 11.7× lift. Cluster 1
adds a further 66,615 rows at 2.7× lift. Between them, clusters 5 and 1 hold
0.8% + 11.3% of volume and a disproportionate share of all fraud.

The honest caveat is that this does **not** demonstrate the clustering has
discovered something the supervised models could not. The `V*` block is
Vesta's own engineered risk feature set — it is already, in effect, the output
of somebody else's fraud model. Finding that a dense region of that space is
fraud-heavy confirms the block is informative; it does not prove the *cluster
ID* adds information beyond what a GBM extracts from the raw columns directly.
That question is exactly what Phase 7's A/B answers, and it is why the feature
views are defined as they are:

| View | Contents | Question it answers |
|---|---|---|
| `base` | the 508 Phase-3 features | reference |
| `unsup` | `base` + 126 PCA components + 3 cluster features | does the representation **add** anything? |
| `compact` | Phase-3 features with the 339 raw `V*` **replaced** by the 126 components + cluster features | can the compression **substitute** for the raw block? |

`compact` is the view that matters for the linear and neural models: if it
matches or beats `base` for them, the collinearity argument in §1 is confirmed
and the model gets a 213-column-smaller input for free.

## 5. Limitations

* **K-means assumes isotropic, roughly equal-variance clusters** in the PCA
  space. The 85-row cluster 4 shows it is also being used as an outlier bin,
  which is arguably a useful side effect but is not what the algorithm is for.
  Hierarchical clustering or a Gaussian mixture would handle elongated clusters
  better; both are substantially more expensive at 590,540 rows and the
  stability result at k=6 was good enough not to need them.
* **Median imputation before PCA distorts the high-missingness groups.** For
  the ~86%-missing groups the components are dominated by the imputed value.
  This is partly mitigated by grouping on missingness first (so the distortion
  is confined to those groups rather than spread across all 339 columns), and
  the missingness itself is preserved separately by Phase 3's indicators.
* **The clustering is fitted once, on the training block, and frozen.** Under
  the non-stationary DGP of Phase 1, the `V*` distribution itself drifts, so
  cluster assignments in the final holdout are made against centroids that are
  up to five months stale. That is the correct simulation of deployment, but it
  means the cluster features should be expected to decay — a point Phase 9
  returns to.
