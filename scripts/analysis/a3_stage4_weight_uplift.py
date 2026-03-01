#!/usr/bin/env python3
"""A3: Stage 4 sampling weight uplift for minority-containing tiles.

Partitions tiles by minority class presence (from augmentation list),
computes mean weight per group, and reports uplift ratios.

Run:
  python scripts/analysis/a3_stage4_weight_uplift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_weights_tsv, load_augmentation_list


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    print("=" * 60)
    print("A3: Stage 4 Sampling Weight Uplift")
    print("=" * 60)

    weights = load_weights_tsv()
    aug = load_augmentation_list()

    settlement_ids = set(aug["settlement_images"])
    seminatural_ids = set(aug["seminatural_images"])
    either_ids = settlement_ids | seminatural_ids

    all_ids = set(weights.keys())
    n_total = len(all_ids)

    groups = [
        ("Settlement",             settlement_ids),
        ("Semi-natural grassland", seminatural_ids),
        ("Either minority class",  either_ids),
    ]

    print(f"\nTotal tiles: {n_total}")
    print(f"\n{'Grouping':<26} {'With':>6} {'Mean wt':>9} {'Without':>8} {'Mean wt':>9} {'Uplift':>8}")
    print("-" * 70)

    for name, minority_set in groups:
        with_ids = all_ids & minority_set
        without_ids = all_ids - minority_set

        w_with = [weights[i] for i in with_ids]
        w_without = [weights[i] for i in without_ids]

        mean_with = mean(w_with)
        mean_without = mean(w_without)
        uplift = mean_with / mean_without

        print(f"{name:<26} {len(w_with):>6} {mean_with:>9.3f} {len(w_without):>8} {mean_without:>9.3f} {uplift:>7.2f}x")

    print()


if __name__ == "__main__":
    main()