#!/usr/bin/env python3
"""
scripts/figures/class_distributions.py

Combined class-distribution comparison of the two datasets used in this study
(review comment C14: present the Biodiversity and OpenEarthMap distributions
alongside each other for easier comparison).

Merges the former standalone figures:
  - Biodiversity class-imbalance summary  (was Figure07.ipynb)
  - filtered OpenEarthMap class distribution (was Figure02.ipynb)

Layout (2 rows x 2 columns); both datasets shown in the same 6-class Biodiversity
taxonomy for a like-for-like comparison (OEM harmonised per the pre-training mapping):
  Row 1  Biodiversity (target):    (a) tile presence   (b) mean pixel proportion
  Row 2  OpenEarthMap (auxiliary): (c) tile presence   (d) mean pixel proportion

Data:
  data/biodiversity_raw/masks/*.png       (6-class Biodiversity taxonomy)
  data/openearthmap_filtered/masks/*.tif  (native 8-class OEM, remapped here to 6 classes)

Output:
  figures/class_distributions.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root_for_imports() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir():
            return parent
    raise RuntimeError("Could not find repo root for imports")


repo_root = find_repo_root_for_imports()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import rasterio

from geoseg.taxonomy import OEM_TO_STUDENT_PRETRAIN


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    raise RuntimeError("Could not find repo root (need data/ and scripts/).")


REPO_ROOT = find_repo_root()

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "stix",
})

# -----------------------------------------------------------------------------
# Biodiversity (target dataset) — 6-class taxonomy, 5 foreground classes plotted
# -----------------------------------------------------------------------------
BIO_COLOR = {
    1: [250, 62, 119],   # Forest
    2: [168, 232, 84],   # Grassland
    3: [242, 180, 92],   # Cropland
    4: [59, 141, 247],   # Settlement
    5: [255, 214, 33],   # Semi-natural grassland
}
# Display names match the OEM<->Biodiversity mapping schematic (oem_mapping.tex)
# so the two figures read as a connected pair (shared palette + shared class names).
BIO_NAMES = {1: "Forest land", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Semi-nat."}
BIO_IDS = [1, 2, 3, 4, 5]

# -----------------------------------------------------------------------------
# OpenEarthMap (auxiliary dataset) — shown in the HARMONISED 6-class Biodiversity
# taxonomy used during pre-training (Table "Taxonomy harmonisation", main.tex).
# Native OEM class id -> Biodiversity class id (pre-training, hard labels), GROUNDED
# from the teacher's empirical OEM->target confusion (geoseg.taxonomy.OEM_TO_STUDENT_PRETRAIN;
# docs/KD_MAPPING_GROUNDING.md):
#   Tree(5)->Forest(1); Rangeland(2)+Water(6)+Agriculture(7)->Grassland(2);
#   Developed space(3)+Road(4)+Building(8)->Settlement(4); Bareland(1)->Semi-natural(5).
# No native class argmaxes to Cropland, so OEM contributes no Cropland exposure;
# Semi-natural is now seeded by Bareland (in pre-training, not KD-only), and Grassland
# absorbs Agriculture (Irish "agriculture" is pasture) plus Water.
# -----------------------------------------------------------------------------
OEM_NATIVE_TO_BIO = {k: v for k, v in OEM_TO_STUDENT_PRETRAIN.items() if k != 0}
# Reverse: Biodiversity foreground class id -> native OEM ids that map onto it
BIO_FROM_OEM = {k: [n for n, b in OEM_NATIVE_TO_BIO.items() if b == k] for k in BIO_IDS}


def biodiversity_distribution():
    mask_dir = REPO_ROOT / "data/biodiversity_raw/masks"
    paths = sorted(mask_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No Biodiversity masks in {mask_dir}")
    presence = {k: 0 for k in BIO_IDS}
    pixel_frac_sum = {k: 0.0 for k in BIO_IDS}
    for p in paths:
        mask = np.array(Image.open(p))
        total = mask.size
        for k in BIO_IDS:
            cnt = int(np.count_nonzero(mask == k))
            if cnt > 0:
                presence[k] += 1
            pixel_frac_sum[k] += cnt / total
    n = len(paths)
    tile_prop = np.array([presence[k] / n for k in BIO_IDS])
    pixel_prop = np.array([pixel_frac_sum[k] / n for k in BIO_IDS])
    print(f"Biodiversity: {n} tiles")
    return tile_prop, pixel_prop


def oem_distribution():
    """OEM composition in the harmonised 6-class Biodiversity taxonomy (pre-training
    mapping). Uses the same mean-per-tile-fraction definition as the Biodiversity
    panel so the two rows are directly comparable."""
    mask_dir = REPO_ROOT / "data/openearthmap_filtered/masks"
    paths = sorted(mask_dir.glob("*.tif")) + sorted(mask_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No OpenEarthMap masks in {mask_dir}")
    presence = {k: 0 for k in BIO_IDS}
    pixel_frac_sum = {k: 0.0 for k in BIO_IDS}
    for p in paths:
        if p.suffix.lower() in (".tif", ".tiff"):
            with rasterio.open(p) as src:
                mask = src.read(1)
        else:
            mask = np.array(Image.open(p))
        total = mask.size
        for k in BIO_IDS:
            natives = BIO_FROM_OEM[k]
            cnt = int(np.isin(mask, natives).sum()) if natives else 0
            if cnt > 0:
                presence[k] += 1
            pixel_frac_sum[k] += cnt / total
    n = len(paths)
    tile_prop = np.array([presence[k] / n for k in BIO_IDS])
    pixel_prop = np.array([pixel_frac_sum[k] / n for k in BIO_IDS])
    print(f"OpenEarthMap (harmonised 6-class): {n} tiles")
    return tile_prop, pixel_prop


def _barh(ax, ids, labels, colors, values, *, xlim, xticks, show_labels, letter, title):
    y = np.arange(len(ids))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_labels else [], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel("Proportion", fontsize=15)
    ax.text(0.5, 1.04, f"{letter} {title}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=13)
    # Annotate near-zero bars with an explicit "0.00" so an absent class reads as an
    # intentional informative zero, not a render glitch (e.g. OEM Cropland under grounding).
    span = xlim[1] - xlim[0]
    for yi, v in zip(y, values):
        if v < 0.005 * span:
            ax.text(0.006 * span, yi, "0.00", ha="left", va="center",
                    fontsize=11, color="#444444")


def main():
    bio_tile, bio_pix = biodiversity_distribution()
    oem_tile, oem_pix = oem_distribution()

    bio_labels = [BIO_NAMES[k] for k in BIO_IDS]
    bio_colors = [np.array(BIO_COLOR[k]) / 255.0 for k in BIO_IDS]

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(11, 7.0),
        gridspec_kw={"height_ratios": [len(BIO_IDS), len(BIO_IDS)]},
    )
    fig.subplots_adjust(wspace=0.18, hspace=0.85, left=0.2)

    # Row 1: Biodiversity (target)
    _barh(axes[0, 0], BIO_IDS, bio_labels, bio_colors, bio_tile,
          xlim=(0, 1.0), xticks=[0, 0.25, 0.5, 0.75, 1.0], show_labels=True,
          letter="(a)", title="Proportion of tiles\ncontaining each class")
    _barh(axes[0, 1], BIO_IDS, bio_labels, bio_colors, bio_pix,
          xlim=(0, 0.7), xticks=[0, 0.2, 0.4, 0.6], show_labels=False,
          letter="(b)", title="Mean pixel\nproportion per class")
    # Row 2: OpenEarthMap (auxiliary), harmonised to the same 6-class taxonomy
    _barh(axes[1, 0], BIO_IDS, bio_labels, bio_colors, oem_tile,
          xlim=(0, 1.0), xticks=[0, 0.25, 0.5, 0.75, 1.0], show_labels=True,
          letter="(c)", title="Proportion of tiles\ncontaining each class")
    _barh(axes[1, 1], BIO_IDS, bio_labels, bio_colors, oem_pix,
          xlim=(0, 0.7), xticks=[0, 0.2, 0.4, 0.6], show_labels=False,
          letter="(d)", title="Mean pixel\nproportion per class")

    # Dataset row labels
    fig.text(0.012, 0.74, "Biodiversity\n(target)", rotation=90,
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.30, "OpenEarthMap\n(auxiliary)", rotation=90,
             ha="center", va="center", fontsize=15, fontweight="bold")

    # Shared "Biodiversity classes" swatch key — canonical STUDENT_PALETTE colours and the
    # same display names used by the OEM<->Biodiversity mapping schematic (oem_mapping.tex),
    # so the distribution figure and the mapping schematic read as a connected pair.
    key_handles = [
        Patch(facecolor=np.array(BIO_COLOR[k]) / 255.0, edgecolor="none", label=BIO_NAMES[k])
        for k in BIO_IDS
    ]
    fig.legend(
        handles=key_handles, title="Biodiversity classes",
        loc="lower center", ncol=len(BIO_IDS), frameon=False,
        bbox_to_anchor=(0.55, 1.0), fontsize=12, title_fontsize=13,
        handlelength=1.1, handleheight=1.0, columnspacing=1.4,
    )

    out = REPO_ROOT / "figures" / "class_distributions.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    main()
