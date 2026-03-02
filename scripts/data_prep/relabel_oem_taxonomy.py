#!/usr/bin/env python3
"""
Relabel OpenEarthMap (OEM) masks into the target 6-class taxonomy.

Input:
  in_root/
    images/*.tif
    masks/*.tif   (OEM integer labels)

Output:
  out_root/
    images/*.tif  (copied or symlinked)
    masks/*.png   (uint8 label masks, remapped)

Notes:
- Background (0) is a semantic class, not ignore.
- No ignore_index is introduced here.
- Any unmapped IDs default to background (0).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict

from scripts.data_prep.convert_oem_masks import convert_oem_mask_tif_to_png

# OEM 8-class IDs (1..8) -> Biodiversity 6-class IDs (0..5)
OEM_ID_TO_TARGET6: Dict[int, int] = {
    0: 0,  # (if any background/no-data appears)
    1: 0,  # Bareland -> background
    2: 2,  # Rangeland -> grassland
    3: 4,  # Developed space -> settlement
    4: 4,  # Road -> settlement
    5: 1,  # Tree -> forest
    6: 0,  # Water -> background
    7: 3,  # Agriculture land -> cropland
    8: 4,  # Building -> settlement
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """Populate dst with src using symlink or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "symlink":
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-root",
        default="data/openearthmap_filtered",
        help="Filtered OEM dataset (expects images/ and masks/)",
    )
    ap.add_argument(
        "--out-root",
        default="data/openearthmap_relabelled",
        help="Output directory for relabelled OEM dataset",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to populate images",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    in_images = in_root / "images"
    in_masks = in_root / "masks"
    out_images = out_root / "images"
    out_masks = out_root / "masks"

    if args.overwrite and out_root.exists() and any(out_root.iterdir()):
        shutil.rmtree(out_root)

    ensure_dir(out_images)
    ensure_dir(out_masks)

    mask_files = sorted(in_masks.glob("*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No OEM masks found in {in_masks}")

    written = 0
    skipped = 0

    for msk in mask_files:
        stem = msk.stem
        img = in_images / f"{stem}.tif"
        if not img.exists():
            continue

        out_img = out_images / f"{stem}.tif"
        out_msk = out_masks / f"{stem}.png"

        if out_msk.exists() and not args.overwrite:
            skipped += 1
            continue

        link_or_copy(img, out_img, args.mode)

        convert_oem_mask_tif_to_png(
            msk,
            out_msk,
            id_map=OEM_ID_TO_TARGET6,
            default_value=0,
        )
        written += 1

    print("[relabel_oem_taxonomy]")
    print(f"  input root:  {in_root}")
    print(f"  output root: {out_root.resolve()}")
    print(f"  masks written: {written}")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
