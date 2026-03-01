#!/usr/bin/env python3
"""
Prepare OpenEarthMap into a teacher-training split.

Supports two input layouts:

A) RAW-by-region:
  raw_root/
    <region>/
      images/
      labels/

B) Filtered-flat:
  raw_root/
    images/
    masks/   (or labels/)

Output:
  out_root/
    train/images, train/masks
    val/images,   val/masks

Notes:
- Pairs images and labels by filename stem.
- Deterministic global train/val split.
- Creates symlinks by default (no large copies); can also copy.
- Consistent CLI: --raw-root / --out-root / --val-frac / --seed / --mode / --overwrite
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.data_prep.convert_oem_masks import quick_mask_stats

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTS = (".png", ".tif", ".tiff")


def ensure_clean_out_root(out_root: Path, overwrite: bool) -> None:
    if out_root.exists():
        if overwrite:
            shutil.rmtree(out_root)
        else:
            if any(out_root.iterdir()):
                raise FileExistsError(
                    f"{out_root} exists and is not empty. Use --overwrite to regenerate."
                )
    out_root.mkdir(parents=True, exist_ok=True)


def ensure_dirs(out_root: Path) -> None:
    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "masks").mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def safe_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _collect_pairs_from_dirs(img_dir: Path, lbl_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    lbls: Dict[str, Path] = {
        p.stem: p for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTS
    }

    pairs: List[Tuple[Path, Path]] = []
    for img_p in imgs:
        mask_p = lbls.get(img_p.stem)
        if mask_p is not None:
            pairs.append((img_p, mask_p))
    return pairs


def find_pairs(raw_root: Path) -> List[Tuple[Path, Path]]:
    """
    Find (image, label) pairs under raw_root.

    Works for:
    - raw_root/<region>/{images,labels}
    - raw_root/{images,masks} or raw_root/{images,labels}
    """
    raw_root = Path(raw_root)

    # Case B: filtered-flat
    flat_images = raw_root / "images"
    flat_masks = raw_root / "masks"
    flat_labels = raw_root / "labels"
    if flat_images.is_dir() and (flat_masks.is_dir() or flat_labels.is_dir()):
        lbl_dir = flat_masks if flat_masks.is_dir() else flat_labels
        return _collect_pairs_from_dirs(flat_images, lbl_dir)

    # Case A: raw-by-region
    pairs: List[Tuple[Path, Path]] = []
    for region_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        img_dir = region_dir / "images"
        lbl_dir = region_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        pairs.extend(_collect_pairs_from_dirs(img_dir, lbl_dir))

    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path to OEM root (raw-by-region OR filtered-flat).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/openearthmap_teacher"),
        help="Output root for teacher split.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction (default: 0.1).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic split.",
    )
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to write outputs (default: symlink).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing out-root before writing.",
    )
    args = ap.parse_args()

    raw_root: Path = args.raw_root
    out_root: Path = args.out_root

    if not raw_root.is_dir():
        raise FileNotFoundError(f"--raw-root not found: {raw_root}")

    if not (0.0 < args.val_frac < 1.0):
        raise ValueError("--val-frac must be between 0 and 1")

    pairs = find_pairs(raw_root)
    if not pairs:
        raise RuntimeError(
            f"No (image, label) pairs found under {raw_root}. "
            "Expected either raw_root/<region>/{images,labels} OR raw_root/{images,masks|labels}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_val = int(round(len(pairs) * args.val_frac))
    n_val = max(1, min(n_val, len(pairs) - 1))  # guard against empty split
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    ensure_clean_out_root(out_root, overwrite=args.overwrite)
    ensure_dirs(out_root)

    writer = safe_symlink if args.mode == "symlink" else safe_copy

    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = out_root / split_name / "images"
        msk_out = out_root / split_name / "masks"

        for img_p, mask_p in split_pairs:
            writer(img_p, img_out / img_p.name)
            writer(mask_p, msk_out / mask_p.name)

    # Sanity stats on validation masks only
    val_masks = [m for _, m in val_pairs]
    quick_mask_stats(val_masks)

    print("[prepare_oem_teacher_data]")
    print(f"  raw-root: {raw_root}")
    print(f"  out-root: {out_root.resolve()}")
    print(f"  mode:     {args.mode}")
    print(f"  seed:     {args.seed}")
    print(f"  split:    train={len(train_pairs)}, val={len(val_pairs)} (val-frac={args.val_frac})")


if __name__ == "__main__":
    main()
