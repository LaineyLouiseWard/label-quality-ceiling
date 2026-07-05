#!/usr/bin/env python3
"""
scripts/figures/ablation_qualitative.py

Qualitative comparison across the 2x2 FACTORIAL CELLS on FOUR tiles (4 columns).

Layout (6 rows x 4 columns):
  Rows:    Satellite image, Ground truth, Baseline, Transfer-only, Sampler-only, Full
  Columns: (a)-(d) = four chosen validation tile IDs

Predictions are the argmax of the median-seed (seed 44) ensemble-member softmax dumped for
each factorial cell by scripts/analysis/dump_seed_softmax.py (single forward pass, no TTA,
matching the ablation convention). Using the dumped softmax keeps all four cells on the SAME
seed (44, the median used for the paper figures) and needs no checkpoints.

Cells -> softmax folders (under <softmax-root>/seed<seed>/analysis/seed_softmax/<cell>/seed<seed>/):
  Baseline      -> stage1_baseline
  Transfer-only -> stage2b_oem_finetune
  Sampler-only  -> stage_sampler_only
  Full          -> stage3_clsbal

Writes: figures/ablation_qualitative.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

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
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)


# -----------------------------------------------------------------------------
# Factorial cells (row key -> softmax folder name). Order = the figure's row order.
# -----------------------------------------------------------------------------
CELLS: List[Tuple[str, str]] = [
    ("baseline", "stage1_baseline"),
    ("transfer", "stage2b_oem_finetune"),
    ("sampler", "stage_sampler_only"),
    ("full", "stage3_clsbal"),
]
ROW_KEYS = ["img", "gt"] + [k for k, _ in CELLS]


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


# -----------------------------------------------------------------------------
# Classes & palette (canonical from dataset)
# -----------------------------------------------------------------------------
CLASS_NAMES: Dict[int, str] = {i: n for i, n in enumerate(CLASSES)}
if 5 in CLASS_NAMES:
    CLASS_NAMES[5] = "Semi-nat."

PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}

# Albumentations Normalize() defaults (used by dataset) for denormalising the display RGB
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# Highlight annotations drawn on the GT and every prediction-cell panel of a column,
# marking the same region down the column so the feature's evolution across cells is
# trackable (settlement consolidating, semi-natural recovering, roads remaining thin).
# Box colours use the study-area palette (Figure 3): orange = settlement cluster (b),
# teal = semi-natural region (d), blue = thin road features (c). Blue goes to (c), the only
# box with no settlement class, so it does not clash with the blue settlement colour.
# -----------------------------------------------------------------------------
ANNOTATED_ROWS = {"gt"} | {k for k, _ in CELLS}
HIGHLIGHT_ANNOTATIONS = [
    {"col": 1, "rows": ANNOTATED_ROWS, "rect": (100, 40, 300, 260), "color": "#E0843A", "lw": 6.0},
    {"col": 3, "rows": ANNOTATED_ROWS, "rect": (140, 40, 320, 340), "color": "#4F9D8E", "lw": 6.0},
    {"col": 2, "rows": ANNOTATED_ROWS, "rect": (20, 120, 380, 200), "color": "#3C6E9E", "lw": 6.0},
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(12):
        if (p / "scripts").is_dir() and (p / "geoseg").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root. Run from inside the repo.")


def softmax_pred(softmax_root: Path, cell: str, seed: int, img_id: str) -> np.ndarray:
    """argmax of the seed's dumped per-pixel softmax for one cell/tile -> class-id mask."""
    p = softmax_root / f"seed{seed}" / "analysis" / "seed_softmax" / cell / f"seed{seed}" / f"{img_id}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"softmax not found: {p}")
    sm = np.load(p).astype(np.float32)          # (6, H, W)
    return np.argmax(sm, axis=0).astype(np.uint8)


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy().astype(np.float32)
    img = img[:3] if img.shape[0] >= 3 else np.repeat(img, 3, axis=0)
    img = (img.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def colorize_mask(mask: np.ndarray, invalid: np.ndarray | None = None) -> np.ndarray:
    """Convert class-id mask -> RGB. If invalid provided, force those pixels to black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    if invalid is not None:
        out[invalid] = (0, 0, 0)
    return out


def make_legend_handles(include_background: bool = True) -> List[Patch]:
    keys = list(range(len(CLASS_NAMES))) if include_background else list(range(1, len(CLASS_NAMES)))
    return [
        Patch(facecolor=np.array(PALETTE[k]) / 255.0, edgecolor="none", label=CLASS_NAMES[k])
        for k in keys
    ]


# -----------------------------------------------------------------------------
# Data loading (RGB + GT, auto-find in val or test)
# -----------------------------------------------------------------------------
def _try_load_from_split(split_root: Path, img_id: str):
    if not split_root.exists():
        return None
    if split_root.name == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
    elif split_root.name == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))
    else:
        return None
    if img_id not in ds.img_ids:
        return None
    item = ds[ds.img_ids.index(img_id)]
    return item["img"], item["gt_semantic_seg"]


def load_tile_any_split(data_root: Path, img_id: str, split_order: List[str]):
    for sp in split_order:
        got = _try_load_from_split(data_root / sp, img_id)
        if got is not None:
            return got
    raise FileNotFoundError(f"{img_id} not found under {data_root} in splits {split_order}")


# -----------------------------------------------------------------------------
# Figure builder
# -----------------------------------------------------------------------------
def make_grid_figure(
    columns: List[Dict[str, np.ndarray]],
    col_titles: List[str],
    row_labels: List[str],
    legend_handles: List[Patch],
    annotations: List[Dict] | None = None,
) -> plt.Figure:
    n_cols = len(columns)
    n_rows = len(row_labels)

    fig = plt.figure(figsize=(2 * (3.1 * n_cols + 1.8), 2 * (2.55 * n_rows)))
    gs = GridSpec(
        nrows=n_rows + 1,
        ncols=n_cols + 1,
        figure=fig,
        width_ratios=[0.70] + [1.0] * n_cols,
        height_ratios=[1.0] * n_rows + [0.55],
        wspace=0.02,
        hspace=0.03,
    )

    row_fs = 50
    col_fs = 50
    leg_fs = 50

    for r, label in enumerate(row_labels):
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        ax_lab.text(0.98, 0.5, label, ha="right", va="center", fontsize=row_fs, fontweight="bold")

    anns = annotations if annotations else []

    for c in range(n_cols):
        col = columns[c]
        for r, k in enumerate(ROW_KEYS):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(col[k])
            ax.set_aspect("auto")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=col_fs, pad=26, fontweight="bold")
            for ann in anns:
                if ann["col"] == c and k in ann["rows"]:
                    x, y, w, h = ann["rect"]
                    ax.add_patch(Rectangle((x, y), w, h, linewidth=ann["lw"],
                                           edgecolor=ann["color"], facecolor="none"))

    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles, loc="center", ncol=3, frameon=False, fontsize=leg_fs,
        handlelength=2.2, handleheight=1.0, columnspacing=1.6, labelspacing=0.9,
        bbox_to_anchor=(0.55, 0.32),
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.985, bottom=0.06)
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-ids", default="biodiversity_1382,biodiversity_0259,biodiversity_2193,biodiversity_1366")
    ap.add_argument("--data-root", default="data/biodiversity_split")
    ap.add_argument("--split-order", default="val,test")
    ap.add_argument("--softmax-root", default="sonic/results",
                    help="root holding seed<seed>/analysis/seed_softmax/<cell>/seed<seed>/<tile>.npy")
    ap.add_argument("--seed", type=int, default=44, help="median seed used for the paper figures")
    ap.add_argument("--out-path", default="figures/ablation_qualitative.pdf")
    args = ap.parse_args()

    set_plot_style()
    repo_root = find_repo_root()

    img_ids = [s.strip() for s in args.img_ids.split(",") if s.strip()]
    split_order = [s.strip() for s in args.split_order.split(",") if s.strip()]
    data_root = (repo_root / args.data_root).resolve()
    softmax_root = (repo_root / args.softmax_root).resolve()

    columns: List[Dict[str, np.ndarray]] = []
    for img_id in img_ids:
        img_t, mask_t = load_tile_any_split(data_root, img_id, split_order)
        img_rgb = denormalize_to_uint8(img_t)
        gt = mask_t.detach().cpu().numpy().astype(np.uint8)
        invalid = (gt == 0)   # outside annotated AOI -> black, consistently

        col: Dict[str, np.ndarray] = {"img": img_rgb, "gt": colorize_mask(gt, invalid=invalid)}
        for key, cell in CELLS:
            pred = softmax_pred(softmax_root, cell, args.seed, img_id)
            col[key] = colorize_mask(pred, invalid=invalid)
        columns.append(col)
        print(f"{img_id}: loaded 4 cells (seed {args.seed}); outside pixels (gt==0): {int(invalid.sum())}")

    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    col_titles = [letters[i] if i < len(letters) else f"({i})" for i in range(len(columns))]

    row_labels = [
        "Satellite\nImage",
        "Ground\nTruth",
        "Baseline",
        "Transfer\nonly",
        "Sampler\nonly",
        "Full",
    ]

    handles = make_legend_handles(include_background=True)
    fig = make_grid_figure(columns, col_titles, row_labels, handles, annotations=HIGHLIGHT_ANNOTATIONS)

    out = (repo_root / args.out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    main()
