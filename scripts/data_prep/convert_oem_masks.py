#!/usr/bin/env python3
"""
Utilities for reading, remapping, and exporting OpenEarthMap (OEM) segmentation masks.

Purpose:
- Convert OEM mask TIFFs (or PNGs) into uint8 PNG label masks.
- Optionally remap OEM taxonomy IDs into a target taxonomy.
- Preserve semantic background (label 0).
- DOES NOT introduce or enforce ignore_index (handled downstream).

Notes:
- Output masks are uint8 with class IDs in [0, N].
- Any source IDs not in the mapping are set to default_value (typically 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import rasterio
from PIL import Image


# ---------------------------------------------------------------------
# Mask readers
# ---------------------------------------------------------------------

def read_mask_tif(mask_tif: Path) -> np.ndarray:
    """Read a single-band mask TIFF into int64 numpy."""
    mask_tif = Path(mask_tif)
    with rasterio.open(mask_tif) as src:
        if src.count != 1:
            raise ValueError(f"Expected 1-band mask, got {src.count}: {mask_tif}")
        arr = src.read(1)
    return arr.astype(np.int64, copy=False)


def read_mask_png(mask_png: Path) -> np.ndarray:
    """Read a PNG mask into int64 numpy."""
    mask_png = Path(mask_png)
    return np.array(Image.open(mask_png)).astype(np.int64, copy=False)


def read_mask_any(mask_path: Path) -> np.ndarray:
    """Read TIFF or PNG mask into int64 numpy."""
    mask_path = Path(mask_path)
    suf = mask_path.suffix.lower()
    if suf in (".tif", ".tiff"):
        return read_mask_tif(mask_path)
    if suf == ".png":
        return read_mask_png(mask_path)
    raise ValueError(f"Unsupported mask extension: {mask_path}")


# ---------------------------------------------------------------------
# Remapping + saving
# ---------------------------------------------------------------------

def remap_ids(
    src_ids: np.ndarray,
    id_map: Dict[int, int],
    default_value: int = 0,
) -> np.ndarray:
    """
    Remap integer class IDs using a lookup table.

    Any source ID not present in id_map is assigned default_value.
    Background (0) is preserved unless explicitly remapped.
    """
    src_ids = np.asarray(src_ids, dtype=np.int64)
    if src_ids.size == 0:
        return src_ids.astype(np.uint8)

    max_id = int(src_ids.max())
    if max_id < 0:
        raise ValueError("Mask contains negative IDs; expected non-negative class IDs.")

    # Guardrail: extremely large IDs are almost certainly an error
    if max_id > 10_000:
        raise ValueError(f"Unusually large class ID detected (max={max_id}). Check mask encoding.")

    lut = np.full((max_id + 1,), default_value, dtype=np.int64)
    for k, v in id_map.items():
        k = int(k)
        if 0 <= k <= max_id:
            lut[k] = int(v)

    out = lut[src_ids]
    return out.astype(np.uint8, copy=False)


def save_mask_png(mask: np.ndarray, out_png: Path) -> None:
    """Save a 2D uint8 mask as PNG (mode 'L')."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    Image.fromarray(mask.astype(np.uint8, copy=False), mode="L").save(out_png)


def convert_oem_mask_tif_to_png(
    mask_tif: Path,
    out_png: Path,
    *,
    id_map: Dict[int, int],
    default_value: int = 0,
) -> None:
    """
    Convert an OEM mask from TIFF to PNG, applying an optional ID remapping.
    """
    src = read_mask_tif(mask_tif)
    tgt = remap_ids(src, id_map=id_map, default_value=default_value)
    save_mask_png(tgt, out_png)


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

def quick_mask_stats(mask_paths: List[Path], max_checks: int = 200) -> None:
    """
    Print quick sanity statistics for a sample of masks.

    Reports:
    - unique labels observed
    - maximum label value
    - presence of label 255 (if any)
    """
    sample = mask_paths[:max_checks]
    if not sample:
        print("[MaskStats] No masks to check.")
        return

    uniques = set()
    maxv = -1
    has_255 = 0

    for p in sample:
        arr = read_mask_any(p)
        uniques.update(np.unique(arr).tolist())
        maxv = max(maxv, int(arr.max()))
        if (arr == 255).any():
            has_255 += 1

    uniques_sorted = sorted(int(u) for u in uniques)
    print(f"[MaskStats] Checked {len(sample)} masks.")
    print(
        f"[MaskStats] Unique labels (sample): "
        f"{uniques_sorted[:40]}{' ...' if len(uniques_sorted) > 40 else ''}"
    )
    print(f"[MaskStats] Max label (sample): {maxv}")
    print(f"[MaskStats] Masks containing 255 (sample): {has_255}")
