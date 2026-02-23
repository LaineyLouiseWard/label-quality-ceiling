# Stage 4 Sampling Weight Uplift

> Validates that the hard × minority-aware sampling formula
> (Equation 1) materially upweights minority-containing tiles.
>
> Sources: `artifacts/stage4_sampling_weights.tsv`,
> `artifacts/train_augmentation_list.json`
> (tiles containing ≥5% settlement or semi-natural pixels).
> N = 1,846 training tiles.

## Uplift factors

| Grouping | Tiles with | Mean weight | Tiles without | Mean weight | **Uplift** |
|----------|:----------:|:-----------:|:-------------:|:-----------:|:----------:|
| Settlement | 466 | 1.182 | 1,380 | 0.939 | **1.26×** |
| Semi-natural grassland | 400 | 2.122 | 1,446 | 0.690 | **3.08×** |
| Either minority class | 800 | 1.580 | 1,046 | 0.556 | **2.84×** |

Uplift = mean weight of minority-containing tiles / mean weight of
complement tiles.

## Interpretation

Minority-containing tiles are drawn at materially higher rates than
majority-only tiles. The semi-natural uplift (3.08×) exceeds the
settlement uplift (1.26×), consistent with semi-natural tiles having
higher richness scores under the sampling formula:
semi-natural grassland is rarer in the dataset (400/1,846 tiles)
and its tiles tend to carry higher minority pixel fractions, both of
which increase the computed weight. The combined minority uplift
(2.84×) confirms the sampler targets the intended tile population
rather than hard majority examples.
