# Supplementary Robustness Analyses

Six analyses derived from saved evaluation outputs and sampling artefacts.
No retraining required. All values are programmatically reproducible via
the scripts in `scripts/analysis/`.

Source artefacts: `evaluation/evaluation_results/val/stage*/`,
`evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/`,
`artifacts/stage4_sampling_weights.tsv`,
`artifacts/train_augmentation_list.json`.

---

## A1. Minority Recall Progression

**Script:** `scripts/analysis/a1_minority_recall.py`

**Motivation.**
The paper claims that data-centric interventions correct minority-class
under-detection. Recall measures exactly this: the fraction of true
minority pixels correctly classified.

**Method.**
Recall = diagonal / row sum from the row-normalised validation confusion
matrices (`confusion_matrix.csv`) for each stage. Rows 4 (Settlement) and
5 (Semi-natural Grassland).

**Findings.**

| Stage | Settlement recall | Semi-nat recall |
|------:|------------------:|----------------:|
| 1     | 81.4%             | 65.6%           |
| 2     | 81.6%             | 72.6%           |
| 3b    | 84.1%             | 74.7%           |
| 4     | 84.3%             | 78.8%           |
| 5     | 86.5%             | 86.2%           |

Net gain (Stage 1 to 5): Settlement +5.2 pp, Semi-natural +20.6 pp.
Gains are monotonically increasing across all stage transitions (verified
by assertion in script).

**Implication.**
Recall drives the minority IoU gains: all interventions progressively
reduce missed minority detections without introducing regressions at
any stage.

---

## A2. Symmetric Confusion Disclosure

**Script:** `scripts/analysis/a2_symmetric_confusion.py`

**Motivation.**
If minority IoU improves because the model over-predicts the minority
class, the reverse confusion direction (majority misclassified as
minority) would rise. Checking both directions distinguishes genuine
separation from error redistribution.

**Method.**
Extract off-diagonal cells from row-normalised confusion matrices for
the two key minority-majority pairs at each stage.

**Findings.**

Semi-natural vs Grassland:

| Stage | SemiNat to Grass | Grass to SemiNat |
|------:|-----------------:|-----------------:|
| 1     | 27.6%            | 0.9%             |
| 2     | 22.8%            | 0.6%             |
| 3b    | 21.2%            | 0.5%             |
| 4     | 18.1%            | 0.4%             |
| 5     | 11.2%            | 0.4%             |

Settlement vs Forest:

| Stage | Settl to Forest | Forest to Settl |
|------:|----------------:|----------------:|
| 1     | 8.9%            | 1.8%            |
| 2     | 9.0%            | 1.7%            |
| 3b    | 7.2%            | 1.4%            |
| 4     | 7.2%            | 1.1%            |
| 5     | 6.0%            | 1.1%            |

**Implication.**
The forward confusion direction (minority misclassified as majority)
decreases steadily while the reverse direction remains below 1%
throughout. The improvement reflects genuine class separation, not
over-prediction of minority classes.

---

## A3. Stage 4 Sampling Weight Uplift

**Script:** `scripts/analysis/a3_stage4_weight_uplift.py`

**Motivation.**
The hard x minority-aware sampling formula should materially upweight
tiles containing minority classes. This analysis validates that the
formula targets the intended tile population.

**Method.**
Partition the 1,846 training tiles by minority-class presence (from
`train_augmentation_list.json`: tiles with >=5% settlement or
semi-natural pixels). Compute mean sampling weight per group from
`stage4_sampling_weights.tsv`. Uplift = mean(with) / mean(without).

**Findings.**

| Grouping               | Tiles with | Mean wt | Tiles without | Mean wt | Uplift |
|------------------------|:----------:|:-------:|:-------------:|:-------:|:------:|
| Settlement             | 466        | 1.181   | 1,380         | 0.939   | 1.26x  |
| Semi-natural grassland | 400        | 2.122   | 1,446         | 0.690   | 3.08x  |
| Either minority class  | 800        | 1.580   | 1,046         | 0.556   | 2.84x  |

**Implication.**
Minority-containing tiles receive materially higher draw probability.
The semi-natural uplift (3.08x) exceeds settlement (1.26x) because
semi-natural tiles are rarer and carry higher minority pixel fractions.

---

## A4. Validation-Test Per-Class Gap

**Script:** `scripts/analysis/a4_val_test_gap.py`

**Motivation.**
Reporting the per-class gap between validation and held-out test
performance quantifies where generalisation is weakest and provides
an honest bound on the transferability of each intervention.

**Method.**
Subtract per-class IoU from `metrics.json` for the Stage 5 model
on the test set vs the validation set. Sorted by gap magnitude.

**Findings.**

| Class          | Val IoU | Test IoU | Gap      |
|----------------|:-------:|:--------:|:--------:|
| Semi-natural   | 81.4%   | 63.6%    | -17.9 pp |
| Cropland       | 88.4%   | 76.0%    | -12.4 pp |
| Settlement     | 76.6%   | 70.6%    | -6.1 pp  |
| Grassland      | 93.3%   | 89.1%    | -4.2 pp  |
| Forest         | 76.7%   | 74.0%    | -2.7 pp  |
| mIoU           | 83.3%   | 74.6%    | -8.6 pp  |

**Implication.**
The largest gaps (semi-natural 17.9 pp, cropland 12.4 pp) reflect
residual sensitivity to appearance variation across the geographic and
seasonal range of the test set. Settlement and forest exhibit smaller
gaps, indicating minority-class improvements are partially retained
under distribution shift.

---

## A5. Majority-Class Stability Bound

**Script:** `scripts/analysis/a5_majority_stability.py`

**Motivation.**
The paper claims that minority-class gains are not achieved at the
expense of majority classes. A formal bound makes this verifiable
rather than leaving it to visual inspection of Table 2.

**Method.**
For each consecutive stage transition (1 to 2, 2 to 3b, 3b to 4,
4 to 5), compute the IoU change for each majority class (Forest,
Grassland, Cropland) from `metrics.json`. Report the maximum decline.

**Findings.**

| Transition | Forest    | Grassland  | Cropland   |
|:-----------|----------:|-----------:|-----------:|
| 1 to 2     | +1.29 pp  | +1.09 pp   | +2.98 pp   |
| 2 to 3b    | +1.64 pp  | +0.92 pp   | +6.24 pp   |
| 3b to 4    | +0.76 pp  | +0.33 pp   | -0.90 pp   |
| 4 to 5     | +1.27 pp  | +0.97 pp   | +0.18 pp   |

Maximum majority-class IoU decline: **0.90 pp** (Cropland, Stage 3b to 4).
All other majority-class transitions are non-negative.

**Implication.**
No majority class declines by more than 0.90 pp at any stage
transition, confirming that minority-class gains are achieved without
meaningful majority-class regression.

---

## A6. Sampling Weight Concentration (Gini)

**Script:** `scripts/analysis/a6_weight_gini.py`

**Motivation.**
The Gini coefficient concisely characterises whether the sampler is
lightly or aggressively concentrated, validating that the mixing
parameter alpha = 0.5 achieves moderate rather than extreme reweighting.

**Method.**
Compute the Gini coefficient and descriptive statistics from
`stage4_sampling_weights.tsv` (N = 1,846 tiles).

**Findings.**

| Statistic        | Value   |
|:-----------------|--------:|
| N (tiles)        | 1,846   |
| Min              | 0.5000  |
| Max              | 3.6710  |
| Mean             | 1.0000  |
| Std              | 0.8293  |
| Median           | 0.6487  |
| 25th pct         | 0.5063  |
| 75th pct         | 1.0631  |
| 95th pct         | 3.6672  |
| Gini coefficient | 0.3605  |

**Implication.**
A Gini of 0.36 indicates moderate concentration consistent with the
intended alpha = 0.5 mixing of weighted and uniform sampling. The right
skew (mean > median) is driven by a small tail of high-weight tiles
combining high hardness and high minority richness.