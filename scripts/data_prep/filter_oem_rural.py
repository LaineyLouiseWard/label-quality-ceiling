#!/usr/bin/env python3
"""
Filter raw OpenEarthMap tiles to keep predominantly rural scenes.

A tile is dropped if the proportion of built environment
(developed + road + building) exceeds a threshold.

Input (RAW OEM):
  raw_root/
    <region>/
      images/*.tif
      labels/*.tif

Output:
  out_root/
    images/*.tif
    masks/*.tif

Images are namespaced as: oem_<region>_<tile_id>.tif
"""

from pathlib import Path
import argparse
import shutil

import numpy as np
import rasterio
from tqdm import tqdm


# OEM raw class IDs
BUILT_CLASSES = {3, 4, 8}  # developed, road, building


def built_env_percentage(label_path: Path) -> float:
    """Return % of pixels belonging to built-environment classes."""
    with rasterio.open(label_path) as src:
        m = src.read(1)

    total = m.size
    if total == 0:
        return 0.0

    built = np.isin(m, list(BUILT_CLASSES)).sum()
    return 100.0 * built / total


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter OEM tiles by built-environment dominance")
    ap.add_argument("--raw-root", required=True, help="Path to raw OpenEarthMap root")
    ap.add_argument("--out-root", required=True, help="Output folder for filtered OEM")
    ap.add_argument("--threshold", type=float, default=50.0, help="Max %% built environment allowed")
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    img_out = out_root / "images"
    msk_out = out_root / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0

    for region in tqdm(sorted(raw_root.iterdir()), desc="Regions"):
        label_dir = region / "labels"
        image_dir = region / "images"

        if not label_dir.is_dir() or not image_dir.is_dir():
            continue

        for label_path in label_dir.glob("*.tif"):
            pct = built_env_percentage(label_path)
            if pct > args.threshold:
                dropped += 1
                continue

            img_path = image_dir / label_path.name
            if not img_path.exists():
                continue

            stem = f"oem_{region.name}_{label_path.stem}"
            dst_label = msk_out / f"{stem}.tif"
            dst_img = img_out / f"{stem}.tif"

            # --- overwrite handling (remove stale outputs) ---
            if args.overwrite:
                if dst_label.exists():
                    dst_label.unlink()
                if dst_img.exists():
                    dst_img.unlink()

            if args.mode == "symlink":
                if not dst_label.exists():
                    dst_label.symlink_to(label_path.resolve())
                if not dst_img.exists():
                    dst_img.symlink_to(img_path.resolve())
            else:
                shutil.copy2(label_path, dst_label)
                shutil.copy2(img_path, dst_img)

            kept += 1

    print("\n[filter_oem_rural]")
    print(f"  threshold: {args.threshold}% built environment")
    print(f"  kept:      {kept}")
    print(f"  dropped:   {dropped}")
    print(f"  output:   {out_root.resolve()}")


if __name__ == "__main__":
    main()
