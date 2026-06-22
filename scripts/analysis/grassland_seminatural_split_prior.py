#!/usr/bin/env python3
"""
Ground the KD Rangeland-split prior (alpha) in the data.

SUPERSEDED (2026-06-19): the campaign KD map is the full grounded confusion
(build_mapping_from_confusion("B")), which has no Rangeland-split alpha. This script and the
legacy name-based split (formerly geoseg/taxonomy.oem_to_student_kd, now removed) are kept only
as the historical record of how the alpha prior was estimated. See docs/KD_MAPPING_GROUNDING.md.

The teacher's OEM "Rangeland" soft mass was split Grassland(alpha) / Seminatural(1-alpha) under
the old KD map. This script reports the empirical prior alpha = Grassland / (Grassland + Seminatural)
from the *training* labels only, so the chosen value could be justified (not tuned on the metric).

Grassland = class 2, Seminatural = class 5 (geoseg/taxonomy.STUDENT_CLASSES).
Counts are over raw label pixels; Background (0) is reported but excluded from the ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

GRASSLAND, SEMINATURAL = 2, 5
NAMES = {0: "Background", 1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data/biodiversity_split/train",
                    help="Split root containing masks/ (default: training split)")
    args = ap.parse_args()

    mask_dir = Path(args.data_root) / "masks"
    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No masks in {mask_dir}")

    counts = np.zeros(6, dtype=np.int64)
    for p in masks:
        m = np.asarray(Image.open(p))
        u, c = np.unique(m, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            if 0 <= k <= 5:
                counts[k] += v

    total = int(counts.sum())
    g, s = int(counts[GRASSLAND]), int(counts[SEMINATURAL])
    alpha = g / (g + s) if (g + s) > 0 else float("nan")

    print(f"Training masks analysed: {len(masks)}  ({args.data_root})")
    print(f"{'class':<12}{'pixels':>14}{'% of all':>10}")
    for k in range(6):
        print(f"{NAMES[k]:<12}{int(counts[k]):>14,}{100*counts[k]/total:>9.2f}%")
    print("-" * 36)
    print(f"Grassland : Seminatural  =  {g:,} : {s:,}  =  {g/s:.2f} : 1" if s else "no seminatural pixels")
    print(f"Empirical split prior  alpha = G/(G+S) = {alpha:.4f}  "
          f"(currently set to 0.70)")


if __name__ == "__main__":
    main()
