#!/usr/bin/env python3
"""
Create a combined Biodiversity + OEM dataset.

Splits:
- train: biodiversity_split/train + OEM relabelled (6-class), prefixed with 'oem_'
- val:   biodiversity_split/val only
- test:  biodiversity_split/test images ONLY (no masks)

Expected inputs:
  bio_root/
    train/images/*.tif
    train/masks/*.png
    val/images/*.tif
    val/masks/*.png
    test/images/*.tif

  oem_root/
    images/*.tif
    masks/*.png

Output:
  out_root/
    train/images, train/masks
    val/images,   val/masks
    test/images
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional


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


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def transfer_dir(
    src_dir: Path,
    dst_dir: Path,
    *,
    ext: str,
    prefix: Optional[str] = None,
    mode: str = "symlink",
    require_nonempty: bool = True,
) -> int:
    """
    Transfer all files matching *ext from src_dir to dst_dir.
    If require_nonempty=False, allow src_dir to have zero matches (used for test masks case).
    Returns number of files transferred.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    files = sorted(src_dir.glob(f"*{ext}"))
    if require_nonempty and not files:
        raise FileNotFoundError(f"No files '*{ext}' found in {src_dir}")

    writer = safe_symlink if mode == "symlink" else safe_copy

    n = 0
    for f in files:
        name = f"{prefix}_{f.name}" if prefix else f.name
        writer(f, dst_dir / name)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Create biodiversity+OEM combined dataset.")
    ap.add_argument("--out-root", type=Path, default=Path("data/biodiversity_oem_combined"))
    ap.add_argument("--bio-root", type=Path, default=Path("data/biodiversity_split"))
    ap.add_argument("--oem-root", type=Path, default=Path("data/openearthmap_relabelled_filtered"))
    ap.add_argument("--oem-prefix", type=str, default="oem")
    ap.add_argument("--img-ext", type=str, default=".tif")
    ap.add_argument("--mask-ext", type=str, default=".png")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root
    bio_root = args.bio_root
    oem_root = args.oem_root

    # sanity
    for p in [
        bio_root / "train" / "images",
        bio_root / "train" / "masks",
        bio_root / "val" / "images",
        bio_root / "val" / "masks",
        bio_root / "test" / "images",
        oem_root / "images",
        oem_root / "masks",
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected path: {p}")

    ensure_clean_out_root(out_root, overwrite=args.overwrite)

    # ------------------------------------------------------------------
    # Biodiversity: train + val (images + masks)
    # ------------------------------------------------------------------
    n_train_img = transfer_dir(
        bio_root / "train" / "images",
        out_root / "train" / "images",
        ext=args.img_ext,
        mode=args.mode,
        require_nonempty=True,
    )
    n_train_msk = transfer_dir(
        bio_root / "train" / "masks",
        out_root / "train" / "masks",
        ext=args.mask_ext,
        mode=args.mode,
        require_nonempty=True,
    )

    n_val_img = transfer_dir(
        bio_root / "val" / "images",
        out_root / "val" / "images",
        ext=args.img_ext,
        mode=args.mode,
        require_nonempty=True,
    )
    n_val_msk = transfer_dir(
        bio_root / "val" / "masks",
        out_root / "val" / "masks",
        ext=args.mask_ext,
        mode=args.mode,
        require_nonempty=True,
    )

    # ------------------------------------------------------------------
    # Biodiversity: test (images ONLY)
    # ------------------------------------------------------------------
    n_test_img = transfer_dir(
        bio_root / "test" / "images",
        out_root / "test" / "images",
        ext=args.img_ext,
        mode=args.mode,
        require_nonempty=True,
    )

    # ------------------------------------------------------------------
    # OEM relabelled → train only (images + masks) with prefix
    # ------------------------------------------------------------------
    prefix = args.oem_prefix.strip()
    prefix = prefix[:-1] if prefix.endswith("_") else prefix  # avoid double "__"
    oem_prefix = prefix

    n_oem_img = transfer_dir(
        oem_root / "images",
        out_root / "train" / "images",
        ext=args.img_ext,
        prefix=oem_prefix,
        mode=args.mode,
        require_nonempty=True,
    )
    n_oem_msk = transfer_dir(
        oem_root / "masks",
        out_root / "train" / "masks",
        ext=args.mask_ext,
        prefix=oem_prefix,
        mode=args.mode,
        require_nonempty=True,
    )

    print("[create_biodiversity_oem_combined]")
    print(f"  out-root: {out_root.resolve()}")
    print(f"  mode:     {args.mode}")
    print(f"  train:    bio imgs={n_train_img}, bio masks={n_train_msk}, oem imgs={n_oem_img}, oem masks={n_oem_msk}")
    print(f"  val:      imgs={n_val_img}, masks={n_val_msk}")
    print(f"  test:     imgs={n_test_img} (no masks)")


if __name__ == "__main__":
    main()
