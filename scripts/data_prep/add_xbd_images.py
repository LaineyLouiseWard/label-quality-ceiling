#!/usr/bin/env python3
"""
Restore the xBD-derived RGB images that OpenEarthMap's ``_wo_xBD`` download omits.

The ``OpenEarthMap_wo_xBD`` release ships every label but, for Maxar-licensing reasons,
withholds the 813 image tiles that were cropped from xView2/xBD. Those 813 ``*.tif``
images are recovered here from the original xBD ``*_pre_disaster.png`` files (which the
user downloads from xView2 after registering) using OEM's own provenance map
``xbd_files.csv`` (``<xbd_png>,<oem_tile>.tif``).

xBD PNGs and the matching OEM labels are both 1024x1024, so the conversion is a direct
3-band uint8 PNG -> TIF write with NO resize. The script is idempotent: tiles whose image
already exists are skipped unless ``--overwrite`` is given, so it can be re-run as more
xView2 subsets (test/hold) are downloaded.

Usage:
  PYTHONPATH=. python scripts/data_prep/add_xbd_images.py \
      --source /tmp/xbd_stage ~/Downloads \
      --oem-root data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def find_missing_tiles(oem_root: Path) -> set[str]:
    """OEM tiles ('<tile>.tif') that have a label but no image on disk."""
    missing: set[str] = set()
    for region in sorted(p for p in oem_root.iterdir() if p.is_dir()):
        img_dir, lbl_dir = region / "images", region / "labels"
        if not lbl_dir.is_dir():
            continue
        img_stems = {p.stem for p in img_dir.iterdir()} if img_dir.is_dir() else set()
        for lbl in lbl_dir.iterdir():
            if lbl.suffix.lower() in (".tif", ".tiff", ".png") and lbl.stem not in img_stems:
                missing.add(f"{lbl.stem}.tif")
    return missing


def tile_to_region(oem_root: Path) -> dict[str, Path]:
    """Map '<tile>.tif' -> the region's images/ dir (derived from where its label lives)."""
    out: dict[str, Path] = {}
    for region in oem_root.iterdir():
        lbl_dir = region / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl in lbl_dir.iterdir():
            if lbl.suffix.lower() in (".tif", ".tiff", ".png"):
                out[f"{lbl.stem}.tif"] = region / "images"
    return out


def index_source_pngs(sources: list[Path]) -> dict[str, Path]:
    """basename '<xbd>.png' -> path, scanning each source root recursively."""
    idx: dict[str, Path] = {}
    for src in sources:
        if not src.exists():
            continue
        for p in src.rglob("*_pre_disaster.png"):
            idx.setdefault(p.name, p)  # first occurrence wins (stable)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=Path,
        nargs="+",
        required=True,
        help="One or more roots to scan recursively for xBD '*_pre_disaster.png' files.",
    )
    ap.add_argument(
        "--oem-root",
        type=Path,
        default=Path("data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD"),
        help="OpenEarthMap_wo_xBD root (regions with images/ and labels/).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Rewrite images that already exist.")
    args = ap.parse_args()

    oem_root: Path = args.oem_root
    if not oem_root.is_dir():
        raise FileNotFoundError(f"--oem-root not found: {oem_root}")

    csv_path = oem_root / "xbd_files.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Provenance map not found: {csv_path}")

    oem_to_xbd: dict[str, str] = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                oem_to_xbd[row[1].strip()] = row[0].strip()

    missing = find_missing_tiles(oem_root)
    regions = tile_to_region(oem_root)
    sources_idx = index_source_pngs(args.source)

    print(f"[add_xbd_images] OEM tiles missing an image: {len(missing)}")
    print(f"[add_xbd_images] source PNGs indexed:        {len(sources_idx)}")

    placed, skipped, unmapped, not_found = 0, 0, [], []
    for tile in sorted(missing):
        xbd_png = oem_to_xbd.get(tile)
        if xbd_png is None:
            unmapped.append(tile)
            continue
        dst = regions[tile] / tile
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        src = sources_idx.get(xbd_png)
        if src is None:
            not_found.append((tile, xbd_png))
            continue

        arr = np.asarray(Image.open(src).convert("RGB"), dtype=np.uint8)
        if arr.shape[:2] != (1024, 1024):
            print(f"  WARN unexpected size {arr.shape} for {src.name} -> writing as-is (no resize)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="RGB").save(dst, format="TIFF")
        placed += 1

    total = len(missing)
    covered = placed + skipped
    print(f"\n[add_xbd_images] placed={placed} skipped(existing)={skipped} "
          f"=> {covered}/{total} tiles now have an image")
    if unmapped:
        print(f"[add_xbd_images] {len(unmapped)} missing tiles not in xbd_files.csv (unexpected): "
              f"{unmapped[:5]}")
    if not_found:
        import collections
        by_event = collections.Counter(xbd.split("_")[0] for _, xbd in not_found)
        print(f"[add_xbd_images] {len(not_found)} tiles still have NO source PNG "
              f"(these xBD images live in xView2 test/hold subsets you haven't supplied):")
        for ev, c in sorted(by_event.items()):
            print(f"    {ev:28s} {c}")

    # Non-zero exit only if something is genuinely wrong (mapping gap), NOT for the
    # expected test/hold shortfall — that is a documented, user-driven choice.
    sys.exit(1 if unmapped else 0)


if __name__ == "__main__":
    main()
