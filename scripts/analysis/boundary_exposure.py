#!/usr/bin/env python3
"""
scripts/analysis/boundary_exposure.py

Boundary-exposure vs baseline difficulty (§4.4). For each foreground class, the fraction of
its ground-truth pixels lying within a fixed band of a class boundary ("boundary exposure"),
regressed against that class's baseline per-class IoU. Tests whether class difficulty is a
geometric property (boundary exposure) rather than a function of class frequency.

Ireland-only: uses the 219 Irish validation masks (data/biodiversity_split/val/masks), the same
set the per-class IoU is scored on. Boundary distance = Euclidean distance (scipy EDT) from each
class pixel to the nearest edge of its own class, in metres (GSD 0.5 m/px).

Baseline per-class IoU is read from analysis/eval_219/per_class_iou.json (stage1_baseline, the
10-seed per-seed mean on the same 219 Irish tiles).

Output: analysis/label_ceiling/boundary_exposure.json (+ printed summary).
Run: PYTHONPATH=. python scripts/analysis/boundary_exposure.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

GSD_M = 0.5
BAND_M = 2.0
NAMES = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Seminatural"}


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "analysis").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def load_mask(path: str) -> np.ndarray:
    from PIL import Image
    m = np.array(Image.open(path))
    return m[..., 0] if m.ndim == 3 else m


def main() -> None:
    root = find_repo_root()
    masks = sorted(glob.glob(str(root / "data/biodiversity_split/val/masks/*")))
    iou = json.load(open(root / "analysis/eval_219/per_class_iou.json"))["stage1_baseline"]["per_class_iou_mean"]

    within = {c: [] for c in NAMES}   # per-pixel indicator lists, per class
    for f in masks:
        m = load_mask(f)
        for c in NAMES:
            b = m == c
            if not b.any():
                continue
            dist_m = distance_transform_edt(b) * GSD_M   # m to nearest own-class edge
            within[c].append((dist_m[b] <= BAND_M).astype(np.float32))

    exposure = {NAMES[c]: float(100.0 * np.concatenate(within[c]).mean()) for c in NAMES}
    x = np.array([exposure[NAMES[c]] for c in NAMES])
    y = np.array([iou[NAMES[c]] * 100.0 for c in NAMES])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    r2 = float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    out = {
        "n_tiles": len(masks),
        "band_m": BAND_M,
        "gsd_m": GSD_M,
        "distance": "euclidean EDT to own-class boundary",
        "exposure_pct_within_band": {k: round(v, 1) for k, v in exposure.items()},
        "baseline_iou_pct": {NAMES[c]: round(iou[NAMES[c]] * 100, 1) for c in NAMES},
        "R2_iou_vs_exposure": round(r2, 3),
        "ols_slope": round(float(slope), 3),
    }
    (root / "analysis/label_ceiling/boundary_exposure.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
