#!/usr/bin/env python3
"""
_gen_mapping_values.py — reproducible value source for oem_mapping.tex.

Loads the frozen teacher confusion (artifacts/teacher_oem_gt_confusion.npz) and prints the
three quantities the OEM<->Bio mapping schematic embeds, so the .tex numbers are NEVER
hand-transcribed:

  (a) SOFT transition matrix  = soft array, row-normalised x100 (each row sums to 100),
      9 OEM rows x 6 student columns. This is build_mapping_from_confusion(mode="B") x100.
  (b) per-row argmax of (a)   = the hard pre-training map; asserted == OEM_TO_STUDENT_PRETRAIN.
  (c) teacher-pixel SHARE per OEM row = HARD array row-sums / grand-total x100.
      (Use the HARD array for the "dominant signal" badges: Agriculture=66.8%. The SOFT
       array gives 63.4% for Agriculture and MUST NOT be used for shares.)

Run: conda run -n ClassImbalance python scripts/figures/_gen_mapping_values.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir() and (parent / "artifacts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


REPO = find_repo_root()
sys.path.insert(0, str(REPO))
from geoseg.taxonomy import (  # noqa: E402
    OEM_NATIVE_CLASSES,
    STUDENT_CLASSES,
    OEM_TO_STUDENT_PRETRAIN,
)

NPZ = REPO / "artifacts" / "teacher_oem_gt_confusion.npz"


def main() -> None:
    d = np.load(NPZ)
    soft = d["soft"].astype(np.float64)   # (9, 6)
    hard = d["hard"].astype(np.float64)   # (9, 6)

    # (a) soft, row-normalised x100
    soft_pct = 100.0 * soft / soft.sum(axis=1, keepdims=True).clip(min=1e-12)

    # (b) per-row argmax == hard pre-training map
    argmax = soft_pct.argmax(axis=1)
    for i in range(9):
        assert argmax[i] == OEM_TO_STUDENT_PRETRAIN[i], (
            f"row {i} ({OEM_NATIVE_CLASSES[i]}): argmax={argmax[i]} "
            f"!= PRETRAIN={OEM_TO_STUDENT_PRETRAIN[i]}"
        )

    # (c) teacher-pixel SHARE per OEM row, from HARD array (NOT soft)
    hard_rowsum = hard.sum(axis=1)
    share_pct = 100.0 * hard_rowsum / hard_rowsum.sum()

    print("# (a) SOFT transition matrix, row-normalised x100 (mode B):")
    header = "OEM \\ Student".ljust(13) + "".join(c[:6].rjust(8) for c in STUDENT_CLASSES)
    print(header)
    for i in range(9):
        row = OEM_NATIVE_CLASSES[i].ljust(13) + "".join(f"{soft_pct[i, j]:8.1f}" for j in range(6))
        print(row)

    print("\n# (b) per-row argmax (== OEM_TO_STUDENT_PRETRAIN, asserted OK):")
    for i in range(9):
        j = argmax[i]
        print(f"{OEM_NATIVE_CLASSES[i]:<13} -> {STUDENT_CLASSES[j]:<12} "
              f"conf={soft_pct[i, j]:5.1f}%")

    print("\n# (c) teacher-pixel SHARE per OEM row (HARD array row-sums / total x100):")
    for i in range(9):
        print(f"{OEM_NATIVE_CLASSES[i]:<13} {share_pct[i]:6.1f}%")

    print("\n# machine-readable (for embedding into .tex):")
    print("SOFT_PCT =", np.round(soft_pct, 1).tolist())
    print("ARGMAX   =", argmax.tolist())
    print("SHARE_PCT=", np.round(share_pct, 1).tolist())


if __name__ == "__main__":
    main()
