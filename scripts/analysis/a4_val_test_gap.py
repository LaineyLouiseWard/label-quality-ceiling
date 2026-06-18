#!/usr/bin/env python3
"""A4: Validation-test per-class IoU gap for the Stage 5 (KD) model.

Loads val and test metrics.json, computes per-class IoU difference,
and prints sorted by gap magnitude.

Run:
  python scripts/analysis/a4_val_test_gap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_metrics, REPO_ROOT

VAL_METRICS = REPO_ROOT / "evaluation/evaluation_results/val/stage4_kd/metrics.json"
TEST_METRICS = REPO_ROOT / "evaluation/evaluation_results/test/stage4_kd/metrics.json"

# Foreground classes only (skip Background)
FOREGROUND_CLASSES = [
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Seminatural Grassland",
]

SHORT_NAMES = {
    "Forest land": "Forest",
    "Grassland": "Grassland",
    "Cropland": "Cropland",
    "Settlement": "Settlement",
    "Seminatural Grassland": "Semi-natural",
}


def main() -> None:
    print("=" * 60)
    print("A4: Validation-Test Per-Class IoU Gap (Stage 5)")
    print("=" * 60)

    val = load_metrics(VAL_METRICS)
    test = load_metrics(TEST_METRICS)

    val_iou = val["per_class_iou"]
    test_iou = test["per_class_iou"]

    gaps = []
    for cls in FOREGROUND_CLASSES:
        v = val_iou[cls] * 100
        t = test_iou[cls] * 100
        gap = t - v
        gaps.append((SHORT_NAMES[cls], v, t, gap))

    # Sort by gap magnitude (largest drop first)
    gaps.sort(key=lambda x: x[3])

    print(f"\n{'Class':<16} {'Val IoU':>9} {'Test IoU':>10} {'Gap':>10}")
    print("-" * 48)

    for name, v, t, gap in gaps:
        print(f"{name:<16} {v:>8.1f}% {t:>9.1f}% {gap:>+9.1f} pp")

    val_miou = val["mIoU_excluding_bg"] * 100
    test_miou = test["mIoU_excluding_bg"] * 100
    print("-" * 48)
    print(f"{'mIoU':<16} {val_miou:>8.1f}% {test_miou:>9.1f}% {test_miou - val_miou:>+9.1f} pp")

    print()


if __name__ == "__main__":
    main()