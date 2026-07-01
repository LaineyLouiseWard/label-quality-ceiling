#!/usr/bin/env python3
"""
scripts/analysis/terrain_separability.py

Terrain-separability probe (§4.4): does elevation / slope separate the dominant confuser pair,
semi-natural grassland vs (improved) grassland? Parallels spectral_separability.py, but the input
is a DEM rather than the Pleiades bands.

Why this replaces the earlier terrain probe: the original number (within-scene Cohen's d ~1.56)
was computed from the opentopodata point-sample cache (a handful of points per tile), which is not
reproducible and inflates within-tile d. Here we take a real DEM RASTER (Copernicus GLO-30, EPSG:4326,
metres), reproject it onto each tile's exact grid, and compute d over the actual class pixels.

Two views, mirroring the spatial-autocorrelation caveat (Meyer 2019, Mannel 2011):
  - POOLED d: all semi pixels vs all grass pixels across tiles. Dominated by BETWEEN-tile absolute
    height, i.e. a geographic shortcut (which tile a pixel is in), not a within-scene landcover cue.
  - WITHIN-TILE d: per tile (both classes present, >=200 px each), d between the two classes' elevation
    within that scene; report the median and sign-consistency. This is the cue a per-image model could
    actually use to disambiguate the pair.

DEM: data/dem/Copernicus_DSM_COG_10_*.tif (4 one-degree GLO-30 cells covering the SW-Ireland tiles).
Ireland-only: the 219 Irish validation tiles (data/biodiversity_split/val/{images,masks}).

Output: analysis/label_ceiling/terrain_separability.json (+ printed summary).
Run: PYTHONPATH=. python scripts/analysis/terrain_separability.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

SEMI, GRASS = 5, 2
MIN_PX = 200  # per class, per tile, for a within-tile comparison


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "analysis").is_dir():
            return parent
    raise RuntimeError("repo root not found")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def build_dem(root: Path):
    """Merge the GLO-30 cells and derive a slope (deg) raster on the native ~30 m grid."""
    files = sorted(glob.glob(str(root / "data/dem/*.tif")))
    srcs = [rasterio.open(f) for f in files]
    elev, tf = merge(srcs)
    elev = elev[0].astype("float32")
    crs = srcs[0].crs
    # pixel size in metres at this latitude (~52 N): dx = res_lon*111320*cos(lat), dy = res_lat*111320
    res_x, res_y = abs(tf.a), abs(tf.e)
    lat0 = 52.0
    dx = res_x * 111320.0 * np.cos(np.radians(lat0))
    dy = res_y * 111320.0
    gy, gx = np.gradient(elev, dy, dx)  # m per m
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    for s in srcs:
        s.close()
    return elev, slope, tf, crs, files


def onto_tile(field, src_tf, src_crs, ds):
    dst = np.zeros((ds.height, ds.width), dtype="float32")
    reproject(source=field, destination=dst,
              src_transform=src_tf, src_crs=src_crs,
              dst_transform=ds.transform, dst_crs=ds.crs,
              resampling=Resampling.bilinear)
    return dst


def main() -> None:
    root = find_repo_root()
    elev, slope, tf, crs, dem_files = build_dem(root)
    imgs = sorted(glob.glob(str(root / "data/biodiversity_split/val/images/*.tif")))
    mdir = root / "data/biodiversity_split/val/masks"

    semi_e, grass_e, semi_s, grass_s = [], [], [], []      # pooled pixel pools
    wt_elev_d, wt_slope_d = [], []                          # within-tile d
    tile_mean_semi, tile_mean_grass = [], []               # between-tile abs-height means
    within_relief = []
    n_tiles = 0
    for f in imgs:
        mp = mdir / Path(f).name.replace(".tif", ".png")
        if not mp.exists():
            continue
        with rasterio.open(f) as ds:
            e = onto_tile(elev, tf, crs, ds)
            sl = onto_tile(slope, tf, crs, ds)
        m = np.array(Image.open(str(mp)).convert("L"))
        if m.shape != e.shape:
            m = np.array(Image.fromarray(m).resize((e.shape[1], e.shape[0]), Image.NEAREST))
        within_relief.append(float(e.std()))
        es, eg = e[m == SEMI], e[m == GRASS]
        ss, sg = sl[m == SEMI], sl[m == GRASS]
        semi_e.append(es); grass_e.append(eg); semi_s.append(ss); grass_s.append(sg)
        if es.size >= MIN_PX and eg.size >= MIN_PX:
            wt_elev_d.append(cohens_d(es, eg))
            wt_slope_d.append(cohens_d(ss, sg))
            tile_mean_semi.append(float(es.mean())); tile_mean_grass.append(float(eg.mean()))
        n_tiles += 1

    sE, gE = np.concatenate(semi_e), np.concatenate(grass_e)
    sS, gS = np.concatenate(semi_s), np.concatenate(grass_s)
    wt_elev_d = np.array(wt_elev_d); wt_slope_d = np.array(wt_slope_d)
    # between-tile: one abs-height mean per tile, semi vs grass tile-level
    bt_d = cohens_d(np.array(tile_mean_semi), np.array(tile_mean_grass))

    out = {
        "dem": [Path(f).name for f in dem_files],
        "dem_source": "Copernicus GLO-30 (ESA, EPSG:4326, metres), reprojected per tile (bilinear)",
        "n_tiles": n_tiles,
        "n_within_tile_pairs": int(len(wt_elev_d)),
        "within_tile_relief_std_m_median": round(float(np.median(within_relief)), 2),
        "elevation": {
            "pooled_cohens_d": round(cohens_d(sE, gE), 3),
            "within_tile_cohens_d_median": round(float(np.median(wt_elev_d)), 3),
            "within_tile_cohens_d_iqr": [round(float(np.percentile(wt_elev_d, 25)), 3),
                                          round(float(np.percentile(wt_elev_d, 75)), 3)],
            "within_tile_frac_semi_higher": round(float(np.mean(wt_elev_d > 0)), 2),
            "between_tile_cohens_d": round(bt_d, 3),
            "semi_mean_m": round(float(sE.mean()), 1), "grass_mean_m": round(float(gE.mean()), 1),
        },
        "slope_deg": {
            "pooled_cohens_d": round(cohens_d(sS, gS), 3),
            "within_tile_cohens_d_median": round(float(np.median(wt_slope_d)), 3),
            "semi_mean_deg": round(float(sS.mean()), 2), "grass_mean_deg": round(float(gS.mean()), 2),
        },
    }
    (root / "analysis/label_ceiling/terrain_separability.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
