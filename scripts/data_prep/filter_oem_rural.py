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

    if out_root.exists() and any(out_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_root}\n"
                "Pass --overwrite to regenerate (stale tiles may otherwise persist)."
            )
        shutil.rmtree(out_root)

    img_out = out_root / "images"
    msk_out = out_root / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0

    if not raw_root.is_dir():
        raise FileNotFoundError(f"OEM raw-root not found: {raw_root}")

    regions = sorted(p for p in raw_root.iterdir() if p.is_dir())
    valid_regions = [r for r in regions if (r / "labels").is_dir() and (r / "images").is_dir()]
    total_labels = sum(len(list((r / "labels").glob("*.tif"))) for r in valid_regions)

    print(f"\n[filter_oem_rural] Input diagnostics:")
    print(f"  raw-root:      {raw_root}")
    print(f"  subdirs found: {len(regions)}")
    print(f"  valid regions: {len(valid_regions)} (with images/ and labels/)")
    print(f"  total labels:  {total_labels}")

    if total_labels == 0:
        raise FileNotFoundError(
            f"No OEM label tiles (*.tif) found under {raw_root}.\n"
            f"  subdirs found: {len(regions)}, valid regions: {len(valid_regions)}\n"
            "  Check that --raw-root points to the directory containing "
            "<region>/{images,labels}/ subdirectories."
        )

    for region in tqdm(valid_regions, desc="Regions"):
        label_dir = region / "labels"
        image_dir = region / "images"

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
    print(f"  output:    {out_root.resolve()}")

    if kept == 0:
        raise RuntimeError(
            f"All {dropped} OEM tiles exceeded the {args.threshold}% built-environment "
            f"threshold — 0 tiles kept.\n"
            f"  Check --raw-root ({raw_root}) and --threshold ({args.threshold})."
        )


if __name__ == "__main__":
    main()
