# Stage 4 Sampling Weight Summary

Source: `artifacts/stage4_sampling_weights.tsv`

## Statistics

| Statistic | Value |
|-----------|-------|
| N (tiles) | 1846 |
| Min       | 0.500003 |
| Max       | 3.671043 |
| Mean      | 0.999983 |
| Std       | 0.829276 |
| Median    | 0.648739 |
| 25th pct  | 0.506304 |
| 75th pct  | 1.063076 |
| 95th pct  | 3.667243 |

## Notes

- Weights are computed offline from the Stage 3b checkpoint using pixel-wise
  error rate (hardness) and minority class pixel fraction (richness).
- Parameters: β=0.5, γ=1.0, α_mix=0.5, clip [5th, 95th], ε=1e-6.
- Mean ≈ 1.0 by construction (weights are normalised then mixed with uniform).
- Replicated tiles inherit the base-tile weight.
