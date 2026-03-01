#!/usr/bin/env python3
"""A6: Gini coefficient and summary statistics for Stage 4 sampling weights.

Computes the Gini coefficient from the weight distribution and prints
descriptive statistics.

Run:
  python scripts/analysis/a6_weight_gini.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_weights_tsv


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumsum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumsum += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    total = cumsum
    if total == 0:
        return 0.0
    return weighted_sum / (n * total)


def percentile(sorted_vals: list[float], p: float) -> float:
    """Simple linear-interpolation percentile."""
    n = len(sorted_vals)
    k = (n - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def main() -> None:
    print("=" * 60)
    print("A6: Stage 4 Sampling Weight Gini Coefficient")
    print("=" * 60)

    weights = load_weights_tsv()
    vals = sorted(weights.values())
    n = len(vals)

    mean_val = sum(vals) / n
    var = sum((v - mean_val) ** 2 for v in vals) / n
    std_val = var ** 0.5
    median_val = percentile(vals, 50)

    g = gini_coefficient(list(weights.values()))

    print(f"\n{'Statistic':<20} {'Value':>12}")
    print("-" * 34)
    print(f"{'N (tiles)':<20} {n:>12}")
    print(f"{'Min':<20} {vals[0]:>12.4f}")
    print(f"{'Max':<20} {vals[-1]:>12.4f}")
    print(f"{'Mean':<20} {mean_val:>12.4f}")
    print(f"{'Std':<20} {std_val:>12.4f}")
    print(f"{'Median':<20} {median_val:>12.4f}")
    print(f"{'25th pct':<20} {percentile(vals, 25):>12.4f}")
    print(f"{'75th pct':<20} {percentile(vals, 75):>12.4f}")
    print(f"{'95th pct':<20} {percentile(vals, 95):>12.4f}")
    print(f"{'Gini coefficient':<20} {g:>12.4f}")

    print()


if __name__ == "__main__":
    main()