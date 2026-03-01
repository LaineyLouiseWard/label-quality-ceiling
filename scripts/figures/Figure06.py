#!/usr/bin/env python3
"""
Fig 6: OEM taxonomy mapping example (same as the notebook), but as a .py script.

Panels:
(a) RGB tile
(b) OEM 8-class mask (raw)
(c) OEM mask mapped to Biodiversity 6-class taxonomy (relabelled)

Changes vs notebook:
- Slightly increased horizontal spacing between panels (wspace).

Defaults target a tile that contains all 8 OEM classes:
  data/openearthmap_raw/.../dolnoslaskie/labels/dolnoslaskie_25.tif

Run:
  python scripts/figures/Figure06.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
from PIL import Image
import rasterio


# ----------------------------
# Matplotlib styling
# ----------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "legend.fontsize": 10,
    }
)

# ----------------------------
# OEM 8-class legend (IDs 1..8, 0 = background/unlabeled)
# ----------------------------
OEM8_HEX = {
    1: "#800000",  # Bareland
    2: "#00FF24",  # Rangeland
    3: "#949494",  # Developed space
    4: "#FFFFFF",  # Road
    5: "#226126",  # Tree
    6: "#0045FF",  # Water
    7: "#4BB549",  # Agriculture land
    8: "#DE1F07",  # Building
}
OEM8_NAMES = {
    1: "Bareland",
    2: "Rangeland",
    3: "Dev. space",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Agri. land",
    8: "Building",
}

# ----------------------------
# Biodiversity 6-class palette (project)
# ----------------------------
COLOR_MAP = {
    0: [0, 0, 0],
    1: [250, 62, 119],   # Forest
    2: [168, 232, 84],   # Grassland
    3: [242, 180, 92],   # Cropland
    4: [59, 141, 247],   # Settlement
    5: [255, 214, 33],   # Semi-natural
}
CLASS_NAMES = {
    0: "Background",
    1: "Forest land",
    2: "Grassland",
    3: "Cropland",
    4: "Settlement",
    5: "Semi-nat.",
}


# ----------------------------
# Helpers (as per notebook)
# ----------------------------
def rgb_percentile_uint8(rgb_float: np.ndarray, p_lo: float = 2, p_hi: float = 98, gamma: float = 1.1) -> np.ndarray:
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


def read_rgb_and_mpp(image_tif: Path, band_order: tuple[int, int, int] = (1, 2, 3)) -> tuple[np.ndarray, float]:
    with rasterio.open(image_tif) as src:
        rgb = np.transpose(src.read(list(band_order)).astype(np.float32), (1, 2, 0))
        rgb[~np.isfinite(rgb)] = np.nan

        px = abs(float(src.transform.a))
        if src.crs is not None and src.crs.is_projected:
            mpp_x = px
        else:
            cy = (src.bounds.top + src.bounds.bottom) / 2.0
            mpp_x = px * 111320.0 * np.cos(np.deg2rad(cy))

    return rgb_percentile_uint8(rgb), float(mpp_x)


def read_mask_any(mask_path: Path) -> np.ndarray:
    mask_path = Path(mask_path)
    suf = mask_path.suffix.lower()
    if suf in [".tif", ".tiff"]:
        with rasterio.open(mask_path) as src:
            return src.read(1)
    if suf == ".png":
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr
    raise ValueError(f"Unsupported mask format: {mask_path}")


def add_scale_bar_pixels(
    ax: plt.Axes,
    img_shape: tuple[int, int, int] | tuple[int, int],
    meters_per_pixel: float,
    length_m: float = 100,
    pad_px: int = 14,
    bar_height_px: int = 6,
) -> None:
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
    ax.add_patch(Rectangle((x0, y0), bar_len_px, bar_height_px, facecolor="black", edgecolor="black", zorder=4))
    ax.text(
        x0 + bar_len_px / 2,
        y0 - 5,
        f"{int(length_m)} m",
        ha="center",
        va="bottom",
        fontsize=27,
        zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1),
    )


def add_north_arrow_pixels(ax: plt.Axes, img_shape: tuple[int, int, int] | tuple[int, int], pad_px: int = 18, size_px: int = 36) -> None:
    h, w = img_shape[:2]
    x = w - pad_px
    y = h - pad_px
    ax.annotate(
        "N",
        xy=(x, y - size_px),
        xytext=(x, y),
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.0),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
    )


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    raise RuntimeError("Could not find repo root (expected to find /data).")


# ----------------------------
# Plot
# ----------------------------
def make_fig(rgb_tif: Path, mask8_tif: Path, mask6_png: Path, out_pdf: Path, scale_m: float = 100.0) -> None:
    rgb8, mpp = read_rgb_and_mpp(rgb_tif)
    mask8 = read_mask_any(mask8_tif)
    mask6 = read_mask_any(mask6_png)

    # OEM 8-class cmap (render 0 as black via "under"; do NOT include in legend)
    cmap8 = ListedColormap([to_rgb(OEM8_HEX[i]) for i in range(1, 9)])
    cmap8.set_under("black")
    norm8 = BoundaryNorm(np.arange(0.5, 9.5, 1), cmap8.N)

    # Biodiversity 6-class cmap
    biodiv_colors = np.array([COLOR_MAP[i] for i in range(6)]) / 255.0
    cmap6 = ListedColormap(biodiv_colors)
    norm6 = BoundaryNorm(np.arange(-0.5, 6.5), cmap6.N)

    # Figure layout: 3 panels + legend row (keeps panel sizes stable)
    fig = plt.figure(figsize=(13.5*2, 8.5*2), dpi=300)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[1.0, 0.28],
        hspace=0.05,
        wspace=0.12,  # <-- slightly increased horizontal spacing vs the notebook
    )

    fig.subplots_adjust(bottom=0.25)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    leg_a = fig.add_subplot(gs[1, 0])
    leg_b = fig.add_subplot(gs[1, 1])
    leg_c = fig.add_subplot(gs[1, 2])

    # (a) RGB
    ax_a.imshow(rgb8)
    ax_a.set_axis_off()
    ax_a.text(0.5, 1.02, "(a)", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=27, fontweight="bold")
    add_scale_bar_pixels(ax_a, rgb8.shape, mpp, length_m=scale_m)
    add_north_arrow_pixels(ax_a, rgb8.shape)

    # (b) OEM 8-class
    ax_b.imshow(mask8, cmap=cmap8, norm=norm8, interpolation="nearest", resample=False)
    ax_b.set_axis_off()
    ax_b.text(0.5, 1.02, "(b)", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=27, fontweight="bold")

    # (c) mapped 6-class
    ax_c.imshow(mask6, cmap=cmap6, norm=norm6, interpolation="nearest", resample=False)
    ax_c.set_axis_off()
    ax_c.text(0.5, 1.02, "(c)", transform=ax_c.transAxes, ha="center", va="bottom", fontsize=27, fontweight="bold")

    # Legends row (no effect on panel sizing)
    for ax in (leg_a, leg_b, leg_c):
        ax.set_axis_off()

    oem_handles = [
        Patch(
            facecolor=to_rgb(OEM8_HEX[i]),
            edgecolor="black" if i == 4 else "none",  # road is white -> outline
            label=OEM8_NAMES[i],
        )
        for i in range(1, 9)
    ]
    leg_b.legend(
        handles=oem_handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=28,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.0,
        labelspacing=0.55,
        bbox_to_anchor=(0.5, 0.75),
    )

    # Biodiversity legend (include Background)
    biodiv_handles = [
        Patch(
            facecolor=np.array(COLOR_MAP[i]) / 255.0,
            edgecolor="black" if i == 0 else "none",  # outline helps the black swatch
            label=CLASS_NAMES[i],
        )
        for i in [0, 1, 2, 3, 4, 5]
    ]

    leg_c.legend(
        handles=biodiv_handles,
        loc="center",
        ncol=2,          # <-- change to 2 so it fits nicely with 6 entries
        frameon=False,
        fontsize=27,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=1.0,
        labelspacing=0.55,
        bbox_to_anchor=(0.5, 0.89),
    )


    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-tif", type=str, default=None)
    ap.add_argument("--mask8-tif", type=str, default=None)
    ap.add_argument("--mask6-png", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--scale-m", type=float, default=100.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo = find_repo_root()

    # Default: daressalaam_28 (contains all 8 OEM classes)
    region = "dolnoslaskie"
    raw_tile = "dolnoslaskie_25"
    relab_tile = f"oem_{region}_{raw_tile}"

    rgb_tif = Path(args.rgb_tif) if args.rgb_tif else repo / f"data/openearthmap_relabelled/images/{relab_tile}.tif"
    mask6_png = Path(args.mask6_png) if args.mask6_png else repo / f"data/openearthmap_relabelled/masks/{relab_tile}.png"
    mask8_tif = Path(args.mask8_tif) if args.mask8_tif else repo / (
        f"data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD/{region}/labels/{raw_tile}.tif"
    )
    out_pdf = Path(args.out) if args.out else repo / "figures/Figure06.pdf"

    for p in [rgb_tif, mask6_png, mask8_tif]:
        if not p.exists():
            raise FileNotFoundError(p)

    make_fig(rgb_tif=rgb_tif, mask8_tif=mask8_tif, mask6_png=mask6_png, out_pdf=out_pdf, scale_m=args.scale_m)


if __name__ == "__main__":
    main()
