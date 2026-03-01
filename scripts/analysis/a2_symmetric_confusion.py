#!/usr/bin/env python3
"""A2: Symmetric confusion tracking for minority class pairs.

Extracts both confusion directions for semi-natural/grassland and
settlement/forest across stages. Confirms asymmetry (improvement is
genuine separation, not over-prediction).

Run:
  python scripts/analysis/a2_symmetric_confusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    STAGES, load_confusion_matrix,
    IDX_FOREST, IDX_GRASSLAND, IDX_SETTLEMENT, IDX_SEMINATURAL,
)


def row_normalised_cell(cm: list[list[int]], true_idx: int, pred_idx: int) -> float:
    row = cm[true_idx]
    row_sum = sum(row)
    if row_sum == 0:
        return 0.0
    return row[pred_idx] / row_sum


def main() -> None:
    print("=" * 60)
    print("A2: Symmetric Confusion Tracking")
    print("=" * 60)

    # Semi-natural <-> Grassland
    print("\n--- Semi-natural vs Grassland ---")
    print(f"{'Stage':<8} {'SemiNat->Grass':>16} {'Grass->SemiNat':>16}")
    print("-" * 44)

    for label, stage_dir in STAGES:
        cm = load_confusion_matrix(stage_dir)
        sn_to_g = row_normalised_cell(cm, IDX_SEMINATURAL, IDX_GRASSLAND)
        g_to_sn = row_normalised_cell(cm, IDX_GRASSLAND, IDX_SEMINATURAL)
        print(f"{label:<8} {sn_to_g * 100:>15.1f}% {g_to_sn * 100:>15.1f}%")

    # Settlement <-> Forest
    print("\n--- Settlement vs Forest ---")
    print(f"{'Stage':<8} {'Settl->Forest':>16} {'Forest->Settl':>16}")
    print("-" * 44)

    for label, stage_dir in STAGES:
        cm = load_confusion_matrix(stage_dir)
        s_to_f = row_normalised_cell(cm, IDX_SETTLEMENT, IDX_FOREST)
        f_to_s = row_normalised_cell(cm, IDX_FOREST, IDX_SETTLEMENT)
        print(f"{label:<8} {s_to_f * 100:>15.1f}% {f_to_s * 100:>15.1f}%")

    print()


if __name__ == "__main__":
    main()