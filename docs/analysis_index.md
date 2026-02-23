# Analysis Index

One-line description and file path for every quantitative analysis in this repository.

## Main ablation

| Analysis | Description | Files |
|----------|-------------|-------|
| Stage 1–5 ablation (val) | mIoU, mF1, OA and per-class IoU for all five ablation stages on the validation set | `evaluation/evaluation_results/val/stage*/metrics.json` |
| Final model (test) | Held-out test-set metrics for the Stage 5 KD model | `evaluation/evaluation_results/test/stage6_final_kd_ftunetformer/metrics.json` |
| Confusion matrices | Row-normalised 6×6 confusion matrices for all val stages | `evaluation/evaluation_results/val/stage*/confusion_matrix.{csv,npy}` |
| KD pixel transitions (Fig 10) | Per-class fraction of pixels corrected vs newly erred by KD relative to baseline | `scripts/figures/Figure10.py` → `figures/Figure10.pdf` |

## Supplementary

| Analysis | Description | Files |
|----------|-------------|-------|
| **A1** Minority recall progression | Recall for Settlement and Semi-natural Grassland at each stage, derived from confusion matrices | `docs/minority_recall_progression.md` |
| **A2** Symmetric confusion | Both confusion directions for minority pairs across stages; confirms genuine class separation | `docs/symmetric_confusion_disclosure.tex` |
| **A3** Stage 4 weight uplift | Mean sampling weight for minority-containing vs. majority-only tiles (uplift: 1.26× settlement, 3.08× semi-natural) | `docs/stage4_weight_uplift.md` |
| **A4** Val–test per-class gap | Per-class IoU gap between validation and held-out test sets; largest for semi-natural (17.9 pp) and cropland (12.4 pp) | `docs/val_test_gap_disclosure.tex` |
| **A5** Majority-class stability | Maximum majority-class IoU decline across all stage transitions (0.90 pp, Cropland Stage 3b→4) | `docs/majority_stability.txt` |
| **A6** Gini coefficient | Gini of Stage 4 weight distribution = 0.36; confirms moderate concentration consistent with α=0.5 mixing | `docs/stage4_weight_gini.tex`, `docs/stage4_weight_gini.md` |
| Weight distribution (Fig XX) | Histogram of Stage 4 sampling weights with mean/median/95th-pct reference lines | `scripts/figures/FigureXX.py` → `figures/FigureXX.pdf` |
| Weight summary stats | Full descriptive statistics for Stage 4 sampling weights | `docs/stage4_weight_summary.md` |
| Paper–code audit | Numerical verification of all Table 1, Table 2, and in-text claims against saved outputs | `docs/paper_code_consistency_audit.md` |
