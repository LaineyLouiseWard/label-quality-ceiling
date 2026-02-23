# Stage 4 Sampling Weight Gini Coefficient

Source: `artifacts/stage4_sampling_weights.tsv` (N = 1,846 tiles)

| Statistic | Value |
|-----------|-------|
| Gini coefficient | 0.3605 |
| Min | 0.5000 |
| Max | 3.6710 |
| Mean | 1.0000 |
| Std | 0.8293 |
| Median | 0.6487 |

A Gini of 0 = perfectly uniform weights; Gini of 1 = all weight on one tile.
0.36 sits in the moderate range, consistent with α = 0.5 mixing with uniform
sampling. The right skew (mean > median) is driven by a small tail of
high-weight tiles combining high hardness and high minority richness.
