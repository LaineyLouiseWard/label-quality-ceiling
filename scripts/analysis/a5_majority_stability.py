#!/usr/bin/env python3
"""A5: Majority-class stability bound across stage transitions.

For each consecutive stage transition, computes the change in IoU for
each majority class (Forest, Grassland, Cropland) and reports the
maximum observed decline.

Run:
  python scripts/analysis/a5_majority_stability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import STAGES, load_val_metrics, CLASS_NAMES, MAJORITY_INDICES


# Keys in metrics.json for majority classes (indices 1-3)
MAJORITY_KEYS = ["Forest land", "Grassland", "Cropland"]


def main() -> None:
    print("=" * 60)
    print("A5: Majority-Class IoU Stability Bound")
    print("=" * 60)

    # Load per-class IoU for each stage
    stage_ious: list[tuple[str, dict[str, float]]] = []
    for label, stage_dir in STAGES:
        m = load_val_metrics(stage_dir)
        stage_ious.append((label, m["per_class_iou"]))

    # Compute deltas for each transition
    print(f"\n{'Transition':<14} {'Forest':>10} {'Grassland':>12} {'Cropland':>11}")
    print("-" * 50)

    max_decline = 0.0
    max_decline_class = ""
    max_decline_transition = ""

    for i in range(len(stage_ious) - 1):
        label_a, iou_a = stage_ious[i]
        label_b, iou_b = stage_ious[i + 1]
        transition = f"{label_a} -> {label_b}"

        deltas = []
        for cls_key in MAJORITY_KEYS:
            delta = (iou_b[cls_key] - iou_a[cls_key]) * 100
            deltas.append(delta)
            if delta < -max_decline:
                max_decline = -delta
                max_decline_class = cls_key
                max_decline_transition = transition

        print(f"{transition:<14} {deltas[0]:>+9.2f} pp {deltas[1]:>+10.2f} pp {deltas[2]:>+9.2f} pp")

    print("-" * 50)
    print(f"\nMaximum majority-class IoU decline: {max_decline:.2f} pp")
    print(f"  Class:      {max_decline_class}")
    print(f"  Transition: {max_decline_transition}")

    print()


if __name__ == "__main__":
    main()