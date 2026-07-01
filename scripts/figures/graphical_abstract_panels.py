#!/usr/bin/env python3
"""
scripts/figures/graphical_abstract_panels.py

Generate the three raster panels for the redesigned graphical abstract
(diagnostic narrative: a strong vision transformer maps rural Ireland well;
the residual error is a thin boundary shell = a label-quality ceiling).

Panels (saved to figures/graphical_abstract/):
  ga_panel1_rgb.png    Satellite RGB tile (Pleiades), dataset-matched normalisation.
  ga_panel2_map.png    The model's segmentation (mean_pred), full class palette.
  ga_panel3_error.png  Desaturated map + glowing error pixels concentrated on class edges.

All three use REAL current-model outputs: the prediction is the 10-seed ensemble argmax
of the final ADE20K shipped model (stage3_clsbal), reconstructed from the per-seed softmax
dumps (sonic/results/seed*/analysis/seed_softmax/stage3_clsbal). Tile biodiversity_0957 is
chosen: it contains all five land classes in a balanced, visually clean scene, and ~95% of
its error lies within 8 px (~4 m) of a class boundary. The ensemble prediction + GT are
cached to figures/graphical_abstract/ga_source_maps.npz so the figure rebuilds without the
(large, gitignored) softmax dumps.

Run from repo root:
  python scripts/figures/graphical_abstract_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage as ndi


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("repo root not found")


REPO = find_repo_root()
TILE = "biodiversity_0957"
SEEDS = [f"seed{n}" for n in range(42, 52)]
OUT = REPO / "figures" / "graphical_abstract"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "ga_source_maps.npz"  # committed: ensemble pred + GT for TILE

# Class palette (geoseg/taxonomy.py STUDENT_PALETTE)
PALETTE = np.array(
    [
        [0, 0, 0],        # 0 Background
        [250, 62, 119],   # 1 Forest
        [168, 232, 84],   # 2 Grassland
        [242, 180, 92],   # 3 Cropland
        [59, 141, 247],   # 4 Settlement
        [255, 214, 33],   # 5 Seminatural
    ],
    dtype=np.uint8,
)


def normalize_percentile(img: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile stretch, matching biodiversity_dataset._normalize_percentile."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        band = img[:, :, c].astype(np.float32)
        valid = band[(band != 0) & ~np.isnan(band)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2)
        out[:, :, c] = band
    return out


def read_rgb(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        data = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)
    data = np.where(np.isnan(data), 0, data)
    data = normalize_percentile(data)
    data = (data * 255).clip(0, 255).astype(np.uint8)
    return data[:, :, :3] if data.shape[2] >= 3 else np.repeat(data, 3, axis=2)


def boundary_band(gt: np.ndarray, width: int = 8) -> np.ndarray:
    b = np.zeros_like(gt, dtype=bool)
    for ax in (0, 1):
        diff = np.diff(gt, axis=ax) != 0
        sl0 = [slice(None)] * 2
        sl0[ax] = slice(0, -1)
        b[tuple(sl0)] |= diff
        sl1 = [slice(None)] * 2
        sl1[ax] = slice(1, None)
        b[tuple(sl1)] |= diff
    return ndi.binary_dilation(b, iterations=width)


def colourise(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def load_pred_gt() -> tuple[np.ndarray, np.ndarray]:
    """Return (gt, ensemble-argmax prediction) for TILE.

    Prefers the small cached npz; otherwise rebuilds the 10-seed ensemble from the
    per-seed softmax dumps (verified to reproduce the paper's mean_pred exactly) and caches it.
    """
    if CACHE.exists():
        z = np.load(CACHE)
        return z["gt"].astype(np.int64), z["pred"].astype(np.int64)

    gt = np.array(Image.open(REPO / "data" / "biodiversity_split" / "val" / "masks" / f"{TILE}.png")).astype(np.int64)
    acc = None
    for s in SEEDS:
        sm = np.load(REPO / "sonic" / "results" / s / "analysis" / "seed_softmax"
                     / "stage3_clsbal" / s / f"{TILE}.npy").astype(np.float32)
        acc = sm if acc is None else acc + sm
    pred = acc.argmax(0).astype(np.int64)
    np.savez_compressed(CACHE, gt=gt.astype(np.uint8), pred=pred.astype(np.uint8))
    return gt, pred


def main() -> None:
    # --- load real assets -----------------------------------------------------
    rgb_path = None
    for split in ("val", "test", "train"):
        cand = REPO / "data" / "biodiversity_split" / split / "images" / f"{TILE}.tif"
        if cand.exists():
            rgb_path = cand
            break
    if rgb_path is None:
        sys.exit(f"RGB tif for {TILE} not found under data/biodiversity_split/*/images")

    gt, pred = load_pred_gt()
    rgb = read_rgb(rgb_path)

    H, W = gt.shape

    # --- Panel 1: satellite RGB ----------------------------------------------
    Image.fromarray(rgb).save(OUT / "ga_panel1_rgb.png")

    # --- Panel 2: the model's segmentation -----------------------------------
    Image.fromarray(colourise(pred)).save(OUT / "ga_panel2_map.png")

    # --- Panel 3: desaturated map + glowing boundary-located error -----------
    seg = colourise(pred).astype(np.float32)
    lum = seg @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grey = np.repeat(lum[:, :, None], 3, axis=2)
    # muted "solved" base: near-grey with a whisper of class colour, darkened so the
    # glowing seams read with high contrast (the interior is calm and correct).
    base = (0.12 * seg + 0.88 * grey) * 0.42 + 8.0
    base = base.clip(0, 255)

    # real errors (prediction != ground truth); these sit on class edges (98% within 8 px)
    err = (pred != gt)
    err_core = ndi.binary_dilation(err, iterations=1)  # thicken 1 px for visibility

    # two-scale glow: a soft wide halo + a tight bright bloom, amber -> white core.
    ef = err_core.astype(np.float32)
    halo_wide = ndi.gaussian_filter(ef, sigma=3.4)
    halo_tight = ndi.gaussian_filter(ef, sigma=1.2)
    halo_wide /= max(halo_wide.max(), 1e-6)
    halo_tight /= max(halo_tight.max(), 1e-6)
    amber = np.array([255, 176, 59], dtype=np.float32)    # electric amber glow
    core = np.array([255, 255, 240], dtype=np.float32)    # near-white hot core

    out = base.copy()
    # screen-blend both halos (wide for atmosphere, tight for punch)
    for halo, gain in ((halo_wide, 1.0), (halo_tight, 1.0)):
        glow_rgb = (gain * halo)[:, :, None] * amber[None, None, :]
        out = 255.0 - (255.0 - out) * (255.0 - glow_rgb.clip(0, 255)) / 255.0
    # lay the bright cores on top
    out[err_core] = core
    out = out.clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(OUT / "ga_panel3_error.png")

    # --- stats for caption sanity check --------------------------------------
    band = boundary_band(gt, 8)
    frac = 100 * (err & band).sum() / max(err.sum(), 1)
    print(f"[panels] tile={TILE}  size={W}x{H}")
    print(f"[panels] error rate={100*err.mean():.2f}%  |  {frac:.1f}% of error within 8 px of a boundary")
    print(f"[panels] classes present (GT): "
          + ", ".join(f"{['Bg','Forest','Grass','Crop','Settle','SemiNat'][c]} {100*np.mean(gt==c):.0f}%"
                       for c in range(6) if np.mean(gt == c) > 0.005))
    print(f"[panels] wrote 3 PNGs to {OUT}")


if __name__ == "__main__":
    main()
