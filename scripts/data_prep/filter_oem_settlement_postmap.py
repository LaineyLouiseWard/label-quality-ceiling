#!/usr/bin/env python3
"""
Pass-2 OEM filter (post-mapping): drop tiles where *settlement* dominates.

Expects (6-class mapped OEM):
  in_root/
    images/*.tif
    masks/*.png   (uint8 labels in {0..5})

Writes:
  out_root/
    images/*.tif
    masks/*.png

Filter:
  drop if 100 * (mask == settlement_id).sum() / mask.size  > threshold

Notes:
- Assumes no ignore_index in masks.
- Pairs by filename stem.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
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


def settlement_percentage(mask_path: Path, settlement_id: int) -> float:
    m = np.array(Image.open(mask_path))
    if m.size == 0:
        return 0.0
    return 100.0 * float((m == settlement_id).sum()) / float(m.size)


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter mapped OEM by settlement dominance (post-map pass).")
    ap.add_argument("--in-root", default="data/openearthmap_relabelled", help="Mapped OEM root (images/, masks/)")
    ap.add_argument(
        "--out-root",
        default="data/openearthmap_relabelled_filtered",
        help="Output root for filtered mapped OEM (images/, masks/)",
    )
    ap.add_argument("--threshold", type=float, default=50.0, help="Max %% settlement allowed (strictly > drops).")
    ap.add_argument("--settlement-id", type=int, default=4, help="Settlement class id in 6-class taxonomy.")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="How to populate output images/masks.")
    ap.add_argument("--overwrite", action="store_true", help="If set, remove existing outputs for re-run.")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    in_images = in_root / "images"
    in_masks = in_root / "masks"
    out_images = out_root / "images"
    out_masks = out_root / "masks"

    if not in_images.is_dir() or not in_masks.is_dir():
        raise FileNotFoundError(f"Expected {in_images} and {in_masks} to exist.")

    ensure_dir(out_images)
    ensure_dir(out_masks)

    mask_files = sorted(in_masks.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {in_masks} (expected *.png)")

    kept = dropped = missing_img = 0

    for msk in tqdm(mask_files, desc="Filtering (postmap settlement)"):
        stem = msk.stem
        img = in_images / f"{stem}.tif"
        if not img.exists():
            missing_img += 1
            continue

        pct = settlement_percentage(msk, args.settlement_id)
        if pct > args.threshold:
            dropped += 1
            continue

        dst_img = out_images / img.name
        dst_msk = out_masks / msk.name

        if args.overwrite:
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            if dst_msk.exists() or dst_msk.is_symlink():
                dst_msk.unlink()

        link_or_copy(img, dst_img, args.mode)
        link_or_copy(msk, dst_msk, args.mode)
        kept += 1

    print("\n[filter_oem_settlement_postmap]")
    print(f"  in_root:    {in_root}")
    print(f"  out_root:   {out_root.resolve()}")
    print(f"  threshold:  {args.threshold}% (drop if strictly >)")
    print(f"  settlement_id: {args.settlement_id}")
    print(f"  kept:       {kept}")
    print(f"  dropped:    {dropped}")
    print(f"  missing_img:{missing_img}")


if __name__ == "__main__":
    main()
