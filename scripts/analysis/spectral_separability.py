#!/usr/bin/env python3
"""
scripts/analysis/spectral_separability.py

Spectral-separability probe (§4.4): do the available bands separate the dominant confuser pair,
semi-natural grassland vs (improved) grassland? Computes NDVI and NDWI from the 4-band Pleiades
imagery and reports Cohen's d between the two classes' pixel distributions. A small |d| means the
spectral information to tell them apart is not present in the input.

Ireland-only: the 219 Irish validation tiles (data/biodiversity_split/val/{images,masks}).
Band order is auto-detected: band 3 (0-indexed) is NIR; Red is whichever of bands 0/2 yields the
higher forest NDVI (vegetation is bright in NIR, dark in Red). Green is band 1.

Cohen's d = (mean_semi - mean_grass) / pooled_sd (pooled over the two classes' pixels).

Output: analysis/label_ceiling/spectral_separability.json (+ printed summary).
Run: PYTHONPATH=. python scripts/analysis/spectral_separability.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import rasterio

SEMI, GRASS, FOREST = 5, 2, 1


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "analysis").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def main() -> None:
    root = find_repo_root()
    imgs = sorted(glob.glob(str(root / "data/biodiversity_split/val/images/*.tif")))
    mdir = root / "data/biodiversity_split/val/masks"

    # detect Red band (0 or 2) by which gives higher forest NDVI on a sample
    ndvi_test = {0: [], 2: []}
    sample = imgs[:40]
    for f in sample:
        with rasterio.open(f) as s:
            a = s.read().astype(np.float32)
        m = np.array(__import__("PIL.Image", fromlist=["Image"]).open(str(mdir / Path(f).name.replace(".tif", ".png"))))
        m = m[..., 0] if m.ndim == 3 else m
        nir = a[3]
        fg = (m == FOREST)
        for rb in (0, 2):
            red = a[rb]
            den = nir + red
            ndvi = np.where(den != 0, (nir - red) / den, np.nan)
            v = ndvi[fg & np.isfinite(ndvi)]
            if v.size:
                ndvi_test[rb].append(np.nanmean(v))
    red_band = 0 if np.nanmean(ndvi_test[0]) >= np.nanmean(ndvi_test[2]) else 2
    green_band, nir_band = 1, 3

    semi_ndvi, grass_ndvi, semi_ndwi, grass_ndwi = [], [], [], []
    n_tiles = 0
    for f in imgs:
        mp = mdir / Path(f).name.replace(".tif", ".png")
        if not mp.exists():
            continue
        with rasterio.open(f) as s:
            a = s.read().astype(np.float32)
        m = np.array(__import__("PIL.Image", fromlist=["Image"]).open(str(mp)))
        m = m[..., 0] if m.ndim == 3 else m
        red, green, nir = a[red_band], a[green_band], a[nir_band]
        valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(nir) & (red + green + nir != 0)
        dv = nir + red
        ndvi = np.where((dv != 0) & valid, (nir - red) / dv, np.nan)
        dw = green + nir
        ndwi = np.where((dw != 0) & valid, (green - nir) / dw, np.nan)
        for cls, nd, nw in [(SEMI, semi_ndvi, semi_ndwi), (GRASS, grass_ndvi, grass_ndwi)]:
            sel = (m == cls) & valid
            nd.append(ndvi[sel & np.isfinite(ndvi)])
            nw.append(ndwi[sel & np.isfinite(ndwi)])
        n_tiles += 1

    sN, gN = np.concatenate(semi_ndvi), np.concatenate(grass_ndvi)
    sW, gW = np.concatenate(semi_ndwi), np.concatenate(grass_ndwi)
    out = {
        "n_tiles": n_tiles,
        "band_order": {"red": red_band, "green": green_band, "nir": nir_band},
        "n_semi_pixels": int(len(sN)), "n_grassland_pixels": int(len(gN)),
        "NDVI": {"semi_mean": round(float(sN.mean()), 4), "grass_mean": round(float(gN.mean()), 4),
                 "cohens_d": round(cohens_d(sN, gN), 3)},
        "NDWI": {"semi_mean": round(float(sW.mean()), 4), "grass_mean": round(float(gW.mean()), 4),
                 "cohens_d": round(cohens_d(sW, gW), 3)},
    }
    (root / "analysis/label_ceiling/spectral_separability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
