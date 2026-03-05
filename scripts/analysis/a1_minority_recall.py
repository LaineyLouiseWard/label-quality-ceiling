#!/usr/bin/env python3
"""A1: Minority class recall progression across ablation stages.

Computes recall (= diagonal / row sum) for Settlement and Semi-natural
Grassland from saved confusion matrices. Asserts monotonic increase.

Run:
  python scripts/analysis/a1_minority_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import STAGES, load_confusion_matrix, IDX_SETTLEMENT, IDX_SEMINATURAL


def recall_from_cm(cm: list[list[int]], class_idx: int) -> float:
    row = cm[class_idx]
    row_sum = sum(row)
    if row_sum == 0:
        return 0.0
    return row[class_idx] / row_sum


def main() -> None:
    print("=" * 60)
    print("A1: Minority Class Recall Progression")
    print("=" * 60)

    settlement_recalls: list[float] = []
    seminatural_recalls: list[float] = []

    print(f"\n{'Stage':<8} {'Settlement recall':>20} {'Semi-nat recall':>18}")
    print("-" * 50)

    for label, stage_dir in STAGES:
        cm = load_confusion_matrix(stage_dir)
        r_set = recall_from_cm(cm, IDX_SETTLEMENT)
        r_sem = recall_from_cm(cm, IDX_SEMINATURAL)
        settlement_recalls.append(r_set)
        seminatural_recalls.append(r_sem)
        print(f"{label:<8} {r_set * 100:>19.1f}% {r_sem * 100:>17.1f}%")

    # Net deltas
    delta_set = (settlement_recalls[-1] - settlement_recalls[0]) * 100
    delta_sem = (seminatural_recalls[-1] - seminatural_recalls[0]) * 100
    print("-" * 50)
    print(f"{'Delta':<8} {delta_set:>+19.1f} pp {delta_sem:>+16.1f} pp")

    # Monotonicity check
    set_mono = all(b >= a for a, b in zip(settlement_recalls, settlement_recalls[1:]))
    sem_mono = all(b >= a for a, b in zip(seminatural_recalls, seminatural_recalls[1:]))

    print(f"\nMonotonic increase (Settlement):    {'PASS' if set_mono else 'FAIL'}")
    print(f"Monotonic increase (Semi-natural):  {'PASS' if sem_mono else 'FAIL'}")

    if not set_mono:
        print("\n  WARNING: Settlement recall is not monotonically increasing")
    if not sem_mono:
        print("\n  WARNING: Semi-natural recall is not monotonically increasing")

    if set_mono and sem_mono:
        print("\nAll monotonicity checks passed.")


if __name__ == "__main__":
    main()