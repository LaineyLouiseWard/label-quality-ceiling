# Analysis Index

One-line description and file path for every quantitative analysis in this repository.

## Main ablation

| Analysis | Description | Files |
|----------|-------------|-------|
| Stage 1–5 ablation (val) | mIoU, mF1, OA and per-class IoU for all five ablation stages on the validation set | `evaluation/evaluation_results/val/stage*/metrics.json` |
| Final model (test) | Held-out test-set metrics for the Stage 5 KD model | `evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/metrics.json` |
| Confusion matrices | Row-normalised 6×6 confusion matrices for all val stages | `evaluation/evaluation_results/val/stage*/confusion_matrix.{csv,npy}` |
| KD pixel transitions (Fig 11) | Per-class fraction of pixels corrected vs newly erred by KD relative to baseline | `scripts/figures/Figure11.py` → `figures/Figure11.pdf` |

## Supplementary

| Analysis | Description | Files |
|----------|-------------|-------|
| **A1** Minority recall progression | Recall for Settlement and Semi-natural Grassland at each stage, derived from confusion matrices | `docs/robustness_analyses.md` § A1 |
| **A2** Symmetric confusion | Both confusion directions for minority pairs across stages; confirms genuine class separation | `docs/robustness_analyses.md` § A2 |
| **A3** Stage 4 weight uplift | Mean sampling weight for minority-containing vs. majority-only tiles (uplift: 1.26× settlement, 3.08× semi-natural) | `docs/robustness_analyses.md` § A3 |
| **A4** Val–test per-class gap | Per-class IoU gap between validation and held-out test sets; largest for semi-natural (17.9 pp) and cropland (12.4 pp) | `docs/robustness_analyses.md` § A4 |
| **A5** Majority-class stability | Maximum majority-class IoU decline across all stage transitions (0.90 pp, Cropland Stage 3b→4) | `docs/robustness_analyses.md` § A5 |
| **A6** Gini coefficient | Gini of Stage 4 weight distribution = 0.36; confirms moderate concentration consistent with α=0.5 mixing | `docs/robustness_analyses.md` § A6 |
| **A7** Minority error-mass reduction | Total off-diagonal mass for minority rows across stages (1 − diagonal per minority class) | Not implemented (optional future analysis) |
| Weight distribution (Fig 4) | Histogram of Stage 4 sampling weights with mean/median/95th-pct reference lines | `scripts/figures/Figure04.py` → `figures/Figure04.pdf` |
| Paper–code audit | Numerical verification of all Table 1, Table 2, and in-text claims against saved outputs | `docs/paper_code_consistency_audit.md` |
