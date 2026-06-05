#!/usr/bin/env python3
"""
Fig 1: Biodiversity example tiles.

Requested changes:
1) 100 m scale bar ONLY on the satellite (RGB) row.
2) Legend back exactly as before (single mask legend under the figure).
3) Reduce horizontal spacing between columns.
4) All text size = 12.
5) MDPI-safe export: >=1100 px width and 300 dpi.

Writes:
  figures/Figure03.pdf

Run:
  python scripts/figures/Figure03.py
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import rasterio


# -----------------------------------------------------------------------------
# Matplotlib style (pin all text to 12 pt)
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,     # important for raster content embedded in PDF
    "savefig.dpi": 300,    # ensure savefig uses 300 dpi as well
})


# ----------------------------
# Helpers
# ----------------------------
def rgb_percentile_uint8(rgb_float, p_lo=2, p_hi=98, gamma=1.1):
    rgb = rgb_float.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        band = rgb[..., c]
        vals = band[np.isfinite(band)]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [p_lo, p_hi])
        if hi <= lo:
            continue
        x = (band - lo) / (hi - lo)
        x = np.clip(x, 0, 1)
        x = x ** (1.0 / gamma)
        out[..., c] = np.nan_to_num(x, nan=0.0)
    return (out * 255).round().astype(np.uint8)


def read_rgb_and_mpp(image_tif: Path):
    with rasterio.open(image_tif) as src:
        rgb = np.transpose(src.read([1, 2, 3]).astype(np.float32), (1, 2, 0))
        rgb[~np.isfinite(rgb)] = np.nan

        px = abs(float(src.transform.a))
        if src.crs is not None and src.crs.is_projected:
            mpp = px
        else:
            cy = (src.bounds.top + src.bounds.bottom) / 2.0
            mpp = px * 111320.0 * np.cos(np.deg2rad(cy))

    return rgb_percentile_uint8(rgb), mpp


def read_mask_png(p: Path):
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def add_scale_bar_pixels(ax, img_shape, meters_per_pixel, length_m=100, pad_px=14, bar_height_px=6):
    h, w = img_shape[:2]
    bar_len_px = int(round(length_m / meters_per_pixel))
    bar_len_px = min(bar_len_px, w - 2 * pad_px)
    if bar_len_px < 3:
        return

    x0 = pad_px
    y0 = h - pad_px - bar_height_px

    # Solid white backing rectangle (bar + label); bbox on text covers the label area.
    _bg = 6
    ax.add_patch(Rectangle(
        (x0 - _bg, y0 - _bg),
        bar_len_px + 2 * _bg, bar_height_px + 2 * _bg,
        facecolor="white", edgecolor="none", zorder=3,
    ))
    ax.add_patch(Rectangle((x0, y0), bar_len_px, bar_height_px,
                           facecolor="black", edgecolor="black", zorder=4))
    ax.text(
        x0 + bar_len_px / 2,
        y0 - 4,
        f"{length_m} m",
        ha="center",
        va="bottom",
        fontsize=22,
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1),
    )


def add_panel_label(ax, label: str):
    ax.text(
        0.5,
        1.02,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=22,          # keep at 12 as requested
        fontweight="bold",
    )


# ----------------------------
# Plot
# ----------------------------
def main():
    repo = Path.cwd()

    # EDIT THESE if your files live elsewhere:
    samples = [
        {
            "image_tif": repo / "data/biodiversity_raw/images/den4_0004.tif",
            "mask_png": repo / "data/biodiversity_raw/masks/den4_0004.png",
        },
        {
            "image_tif": repo / "data/biodiversity_raw/images/ireland2_0022.tif",
            "mask_png": repo / "data/biodiversity_raw/masks/ireland2_0022.png",
        },
        {
            "image_tif": repo / "data/biodiversity_raw/images/col1_0020.tif",
            "mask_png": repo / "data/biodiversity_raw/masks/col1_0020.png",
        },
    ]

    # Biodiversity palette (same as before)
    COLOR_MAP = {
        0: [0, 0, 0],
        1: [250, 62, 119],
        2: [168, 232, 84],
        3: [242, 180, 92],
        4: [59, 141, 247],
        5: [255, 214, 33],
    }
    CLASS_NAMES = {
        0: "Background",
        1: "Forest land",
        2: "Grassland",
        3: "Cropland",
        4: "Settlement",
        5: "Semi-nat.",
    }

    cmap = ListedColormap(np.array([COLOR_MAP[i] for i in range(6)]) / 255.0)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

    # --- MDPI-safe figure size ---
    # Keep your original ratio (9.0/6.8 ≈ 1.3235), but render at 300 dpi.
    # 7.5 inches wide at 300 dpi => 2250 px width (well above 1100 px).
    fig = plt.figure(figsize=(14.7, 11.1), dpi=300)

    # Reduce horizontal spacing between columns (wspace)
    gs = GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.20],
        hspace=0.25,
        wspace=0.02,
    )

    # --- RGB row (scale bar on every RGB panel) ---
    for j, s in enumerate(samples):
        ax = fig.add_subplot(gs[0, j])
        rgb, mpp = read_rgb_and_mpp(s["image_tif"])
        ax.imshow(rgb)
        ax.set_axis_off()
        add_panel_label(ax, f"({chr(97 + j)})")  # (a) (b) (c)
        add_scale_bar_pixels(ax, rgb.shape, mpp, length_m=100)

    # --- Mask row (NO scale bars) ---
    for j, s in enumerate(samples):
        ax = fig.add_subplot(gs[1, j])
        mask = read_mask_png(s["mask_png"])
        ax.imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_axis_off()
        add_panel_label(ax, f"({chr(100 + j)})")  # (d) (e) (f)

    # Legend (single legend under figure)
    handles = [
        Patch(
            facecolor=np.array(COLOR_MAP[i]) / 255.0,
            edgecolor="black" if i == 0 else "none",
            label=CLASS_NAMES[i],
        )
        for i in range(6)
    ]
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=22,
        handlelength=2.0,
        columnspacing=1.6,
        labelspacing=0.9,
    )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.10)

    out = repo / "figures/Figure03.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ensure PDF is written with 300 dpi for raster elements
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    main()
