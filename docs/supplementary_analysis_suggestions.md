# Supplementary Analysis Suggestions

> Derived exclusively from saved evaluation outputs, confusion matrices,
> and sampling artefacts. No retraining required.
>
> Available files:
> - `evaluation/evaluation_results/val/stage{1,2,3a,3b,4,5}/confusion_matrix.{csv,npy}`
> - `evaluation/evaluation_results/val/stage*/metrics.json`
> - `evaluation/evaluation_results/test/stage6_final_kd_ftunetformer/metrics.json`
> - `artifacts/stage4_sampling_weights.tsv`
> - `artifacts/train_augmentation_list.json`

---

## A1. Per-class recall and precision progression across stages

**What to compute.**
From each stage confusion matrix, derive per-class TP, FP, FN and compute
precision and recall separately (in addition to the IoU already reported).
Tabulate recall for Settlement and Semi-natural Grassland at every stage.

**Why it strengthens the paper.**
The paper claims that data-centric strategies correct minority class
under-detection. Recall measures exactly this (missed minority pixels).
If recall drives the IoU gains while precision remains stable or improves
modestly, this is direct mechanistic evidence that the interventions work
by increasing exposure rather than by suppressing false alarms.
Conversely, if a stage improves IoU but degrades precision, it signals
over-prediction, which is a distinct failure mode worth disclosing.

**Required files.** `confusion_matrix.csv` for all 6 val stages.

**Effort.** Low — all values are algebraically derivable from the
existing confusion matrices in a single script.

**Where it belongs.** Results § Ablation Study, as a companion table
to Table 2. Alternatively, Appendix.

---

## A2. Symmetric confusion tracking (both misclassification directions)

**What to compute.**
The paper currently reports semi-natural → grassland confusion (one
direction). Also compute grassland → semi-natural and, for settlement,
both settlement → forest and forest → settlement at each stage.

Specifically, from row-normalised CMs extract:
- CM[2,5]: grassland predicted as semi-natural (false minority positives)
- CM[5,2]: semi-natural predicted as grassland (missed minority)
- CM[1,4]: forest predicted as settlement (false minority positives)
- CM[4,1]: settlement predicted as forest (missed minority)

**Why it strengthens the paper.**
If minority IoU improves because recall rises but grassland→semi-nat
confusion also rises, the model is trading majority precision for minority
recall. If both directions improve (or the reverse direction stays flat),
the improvement is genuine separation rather than a redistribution of
errors. This directly addresses a plausible reviewer concern.

**Required files.** `confusion_matrix.csv` for stages 1, 2, 3b, 4, 5.

**Effort.** Low.

**Where it belongs.** Results § Ablation Analysis, as an extension of
the current confusion matrix discussion. Could be a small supplementary
table (two rows × five stages).

---

## A3. Sampling weight vs. minority tile membership

**What to compute.**
Using `train_augmentation_list.json` (lists tile IDs containing
settlement / semi-natural above the 5% threshold) and
`stage4_sampling_weights.tsv`, compute:
- Mean weight of tiles *with* settlement vs. *without*
- Mean weight of tiles *with* semi-natural vs. *without*
- Mean weight of tiles with *either* minority class vs. neither

**Why it strengthens the paper.**
This validates that the hard × minority-aware formula actually targets
minority-containing tiles at sampling time, not just hard majority tiles.
It is a direct empirical check of the sampling strategy's behaviour,
addressing the question of whether minority tiles receive materially
higher draw probability.

**Required files.**
`artifacts/stage4_sampling_weights.tsv`,
`artifacts/train_augmentation_list.json`.

**Effort.** Low — a simple group-mean comparison, ~20 lines of code.

**Where it belongs.** Methods § Hard × Minority-Aware Sampling, as a
one-sentence or one-table empirical validation. Or Appendix.

---

## A4. Val–test generalisation gap per class

**What to compute.**
The final model (Stage 5) achieves val mIoU = 83.3% but test mIoU =
74.6%, a gap of 8.6 pp. Per class:

| Class             | Val IoU | Test IoU | Gap   |
|-------------------|---------|----------|-------|
| Forest land       | 76.7%   | 74.0%    | −2.7  |
| Grassland         | 93.3%   | 89.1%    | −4.2  |
| Cropland          | 88.4%   | 75.9%    | −12.5 |
| Settlement        | 76.6%   | 70.6%    | −6.1  |
| Semi-nat.         | 81.4%   | 63.6%    | −17.8 |

The semi-natural grassland gap (17.8 pp) and cropland gap (12.5 pp) are
notably large.

**Why it strengthens the paper.**
Reporting test-set performance already appears in the paper, but
explicitly tabulating per-class val–test gaps quantifies where
generalisation is weakest and provides an honest upper-bound caveat for
each class. It also identifies semi-natural grassland as the class where
the val improvements are least fully retained at test time, which can be
contextualised within the semantic ambiguity discussion in § 4.4.

**Required files.**
`evaluation_results/val/stage6_kd/stage6_kd/metrics.json`,
`evaluation_results/test/stage6_final_kd_ftunetformer/metrics.json`.

**Effort.** Low — values already computed (see above).

**Where it belongs.** Results § Additional Results, or Discussion
§ Limitations.

---

## A5. Majority-class IoU stability: formalised bound

**What to compute.**
For each stage transition (1→2, 2→3b, 3b→4, 4→5), compute the change
in IoU for each majority class (Forest, Grassland, Cropland) and report
the maximum absolute drop across all transitions and majority classes.

**Why it strengthens the paper.**
The paper asserts that "minority performance gains are not achieved at
the expense of majority classes." This is currently supported only by
visual inspection of Table 2. A single sentence stating the maximum
observed majority-class drop (e.g. "no majority class IoU declined by
more than X pp at any stage transition") gives this claim a concrete,
verifiable bound rather than leaving it implicit.

**Required files.** `metrics.json` for all val stages.

**Effort.** Low — one pass over the already-loaded metrics.

**Where it belongs.** Results § Ablation Analysis, as a single
supporting sentence.

---

## A6. Sampling weight concentration (Gini coefficient)

**What to compute.**
Compute the Gini coefficient of the Stage 4 weight distribution, and
the share of total expected draw probability held by the top 10% of
tiles (those above the 90th weight percentile).

Key values (from existing artefact, 1846 tiles):
- 44.6% of tiles cluster near the uniform floor (w < 0.6)
- 26.8% receive above-mean weight (w > 1.0)
- 10.6% receive more than double the mean weight (w > 2.0)

**Why it strengthens the paper.**
The Gini coefficient concisely characterises whether the sampler is
lightly or aggressively concentrated. A moderate Gini (e.g. 0.35–0.50)
supports the paper's claim of "gentle" reweighting mixed with uniform
sampling, while a very high Gini would be a red flag for instability.
It also validates that the mixing parameter α = 0.5 achieves a meaningful
but not extreme concentration.

**Required files.** `artifacts/stage4_sampling_weights.tsv`.

**Effort.** Low — `scipy.stats` or a 5-line manual Gini calculation.

**Where it belongs.** Methods § Hard × Minority-Aware Sampling, as a
one-line characterisation of the weight distribution. Or Appendix
alongside the weight histogram (FigureXX).

---

## A7. Minority-class total error mass reduction across stages

**What to compute.**
For each stage, compute the total off-diagonal mass for the minority
class rows (rows 4 and 5 in the 6×6 CM), normalised by the row sum.
This is 1 − diagonal for each minority row, i.e. total misclassification
rate per minority class.

Plot as a two-line chart (one per minority class) across the five main
stages.

**Why it strengthens the paper.**
Figure 9 already plots IoU; a total error-mass chart is complementary
because IoU conflates false positives and false negatives from *all*
classes, while this metric focuses purely on how often the model makes
any error when a minority pixel is present. It provides an alternative
summary of the same improvement that is more directly interpretable to
readers unfamiliar with IoU decomposition.

**Required files.** `confusion_matrix.csv` for all 6 val stages.

**Effort.** Low.

**Where it belongs.** Appendix, or as an inset panel in Figure 9.

---

## Priority order

| # | Analysis | Impact | Effort | Placement |
|---|----------|--------|--------|-----------|
| A1 | Per-class recall/precision progression | High | Low | Results / Appendix |
| A2 | Symmetric confusion (both directions) | High | Low | Results |
| A4 | Val–test per-class gap table | High | Low | Results / Discussion |
| A3 | Weight vs. minority tile membership | Medium | Low | Methods / Appendix |
| A5 | Majority-class stability bound | Medium | Low | Results (1 sentence) |
| A6 | Gini coefficient of weight distribution | Medium | Low | Methods / Appendix |
| A7 | Minority error-mass reduction chart | Low | Low | Appendix |

A1, A2, and A4 are the highest-priority additions: they directly
substantiate the three central claims of the paper (recall-driven minority
improvement, genuine class separation rather than error redistribution,
and honest reporting of generalisation limits).
