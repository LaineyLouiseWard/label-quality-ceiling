# Minority Class Recall Progression

Recall computed from row-normalised validation confusion matrices
(recall = TP / row sum = fraction of true minority pixels correctly classified).
Stages 3a and 5-cropping omitted; table follows the main ablation sequence.

| Stage | Components                            | Settlement recall | Semi-nat recall |
|-------|---------------------------------------|------------------:|----------------:|
| 1     | Baseline                              | 81.4%             | 65.6%           |
| 2     | + Minority-aware replication          | 81.6%             | 72.6%           |
| 3b    | + OpenEarthMap fine-tuning            | 84.1%             | 74.7%           |
| 4     | + Hard × minority-aware sampling      | 84.3%             | 78.8%           |
| 5     | + Knowledge distillation              | 86.5%             | 86.2%           |

**Net gain (Stage 1 → 5):** Settlement +5.1 pp, Semi-natural +20.6 pp.

## Notes

- Recall is derived directly from `confusion_matrix.csv` (row 4 = Settlement,
  row 5 = Seminatural Grassland; diagonal / row sum).
- Source files: `evaluation/evaluation_results/val/stage*/confusion_matrix.csv`
- Gains are monotonic for both minority classes across all stage transitions.
- The large semi-natural recall gain at Stage 5 (+7.4 pp over Stage 4) confirms
  that knowledge distillation primarily resolves missed semi-natural detections
  rather than reducing false positives. Combined with the symmetric confusion
  results (semi-nat→grassland falls from 28% to 11%), this indicates genuine
  improved class separation, not over-prediction of semi-natural.
