#!/usr/bin/env python3
"""
scripts/figures/Figure08.py

Qualitative ablation comparison on FOUR tiles, shown as 4 columns.

Layout (7 rows x 4 columns):
  Rows:    Satellite image, GT, Stage 1, Stage 2, Stage 3a, Stage 3b, Stage 4, Stage 5
  Columns: (a)–(d) = four chosen tile IDs

Shows RAW predictions (no extra AOI masking beyond GT==0 invalid masking used for display).
Writes:
  figures/Figure08.pdf
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
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
# display tweak only:
if 5 in CLASS_NAMES:
    CLASS_NAMES[5] = "Semi-nat."

PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}

# Albumentations Normalize() defaults (used by dataset)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# Highlight annotations drawn on specific (column, row) panels
# -----------------------------------------------------------------------------
# Each entry: col index (0-based), row keys to annotate, rect (x, y, w, h)
# in 512×512 pixel coords, edge colour, and linewidth.
HIGHLIGHT_ANNOTATIONS = [
    {  # H1: Compact settlement cluster — tile (b)
        "col": 1,
        "rows": {"gt", "s4"},
        "rect": (100, 40, 300, 260),
        "color": "#00FFFF",   # cyan
        "lw": 6.0,
    },
    {  # H2: Large semi-natural region — tile (d)
        "col": 3,
        "rows": {"gt", "s1", "s4"},
        "rect": (140, 40, 320, 340),
        "color": "#7B2FBE",   # purple
        "lw": 6.0,
    },
    {  # H3: Thin linear road/settlement features — tile (c)
        "col": 2,
        "rows": {"gt", "s5"},
        "rect": (20, 120, 380, 200),
        "color": "#FFFFFF",   # white
        "lw": 6.0,
    },
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


def resolve_ckpt(path_like: str) -> Path:
    p = Path(path_like).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {p}")
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    ckpts = list(p.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {p}")

    ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpts[0]


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("net."):
            cleaned[k.replace("net.", "", 1)] = v
        elif k.startswith("model."):
            cleaned[k.replace("model.", "", 1)] = v
        else:
            cleaned[k] = v

    net.load_state_dict(cleaned, strict=False)
    return net


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


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
# Data loading (auto-find in val or test)
# -----------------------------------------------------------------------------
def _try_load_from_split(split_root: Path, img_id: str):
    if not split_root.exists():
        return None

    if split_root.name == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
        if img_id not in ds.img_ids:
            return None
        idx = ds.img_ids.index(img_id)
        item = ds[idx]
        return item["img"], item["gt_semantic_seg"]

    if split_root.name == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))
        if img_id not in ds.img_ids:
            return None
        idx = ds.img_ids.index(img_id)
        item = ds[idx]
        return item["img"], item["gt_semantic_seg"]

    return None


def load_tile_any_split(data_root: Path, img_id: str, split_order: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    for sp in split_order:
        got = _try_load_from_split(data_root / sp, img_id)
        if got is not None:
            return got
    raise FileNotFoundError(f"{img_id} not found under {data_root} in splits {split_order}")


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=6, pretrained=False)


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

    row_fs = 42
    col_fs = 42
    leg_fs = 42

    for r, label in enumerate(row_labels):
        ax_lab = fig.add_subplot(gs[r, 0])
        ax_lab.axis("off")
        ax_lab.text(
            0.98, 0.5, label,
            ha="right", va="center",
            fontsize=row_fs, fontweight="bold",
        )

    anns = annotations if annotations else []

    keys = ["img", "gt", "s1", "s2", "s3a", "s3b", "s4", "s5"]
    for c in range(n_cols):
        col = columns[c]
        for r, k in enumerate(keys):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(col[k])
            ax.set_aspect("auto")
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=col_fs, pad=10, fontweight="bold")
            # Draw highlight rectangles
            for ann in anns:
                if ann["col"] == c and k in ann["rows"]:
                    x, y, w, h = ann["rect"]
                    ax.add_patch(Rectangle(
                        (x, y), w, h,
                        linewidth=ann["lw"],
                        edgecolor=ann["color"],
                        facecolor="none",
                    ))

    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=leg_fs,
        handlelength=2.2,
        handleheight=1.0,
        columnspacing=1.6,
        labelspacing=0.9,
        bbox_to_anchor=(0.55, 0.32),
    )

    fig.subplots_adjust(left=0.06, right=0.995, top=0.985, bottom=0.06)
    return fig


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-ids", default="biodiversity_1382,biodiversity_0259,biodiversity_2193,biodiversity_1366  ")
    ap.add_argument("--data-root", default="data/biodiversity_split")
    ap.add_argument("--split-order", default="val,test")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out-path", default="figures/Figure08.pdf")

    # Match your actual folder names under model_weights/biodiversity/
    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline")
    ap.add_argument("--stage2-ckpt", default="model_weights/biodiversity/stage2_replication")
    ap.add_argument("--stage3a-ckpt", default="model_weights/biodiversity/stage3a_pretrain")
    ap.add_argument("--stage3b-ckpt", default="model_weights/biodiversity/stage3b_finetune")
    ap.add_argument("--stage4-ckpt", default="model_weights/biodiversity/stage4_sampling")
    ap.add_argument("--stage5-ckpt", default="model_weights/biodiversity/stage5_kd")

    args = ap.parse_args()

    set_plot_style()
    repo_root = find_repo_root()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    img_ids = [s.strip() for s in args.img_ids.split(",") if s.strip()]
    split_order = [s.strip() for s in args.split_order.split(",") if s.strip()]
    data_root = (repo_root / args.data_root).resolve()

    ckpt_paths = {
        "s1":  resolve_ckpt(str((repo_root / args.stage1_ckpt).resolve())),
        "s2":  resolve_ckpt(str((repo_root / args.stage2_ckpt).resolve())),
        "s3a": resolve_ckpt(str((repo_root / args.stage3a_ckpt).resolve())),
        "s3b": resolve_ckpt(str((repo_root / args.stage3b_ckpt).resolve())),
        "s4":  resolve_ckpt(str((repo_root / args.stage4_ckpt).resolve())),
        "s5":  resolve_ckpt(str((repo_root / args.stage5_ckpt).resolve())),
    }

    nets: Dict[str, torch.nn.Module] = {}
    for k, ck in ckpt_paths.items():
        net = build_ftunetformer()
        net = load_net_from_lightning_ckpt(net, ck).to(device)
        nets[k] = net
        print(f"{k}: {ck}")

    columns: List[Dict[str, np.ndarray]] = []
    for img_id in img_ids:
        img_t, mask_t = load_tile_any_split(data_root, img_id, split_order)

        img_rgb = denormalize_to_uint8(img_t)
        gt = mask_t.detach().cpu().numpy().astype(np.uint8)

        # Outside AOI mask = GT background (0): show as black consistently
        invalid = (gt == 0)

        pred: Dict[str, np.ndarray] = {}
        for stage_key in ["s1", "s2", "s3a", "s3b", "s4", "s5"]:
            pred[stage_key] = predict_mask(nets[stage_key], img_t, device)

        columns.append({
            "img": img_rgb,
            "gt": colorize_mask(gt, invalid=invalid),
            "s1": colorize_mask(pred["s1"], invalid=invalid),
            "s2": colorize_mask(pred["s2"], invalid=invalid),
            "s3a": colorize_mask(pred["s3a"], invalid=invalid),
            "s3b": colorize_mask(pred["s3b"], invalid=invalid),
            "s4": colorize_mask(pred["s4"], invalid=invalid),
            "s5": colorize_mask(pred["s5"], invalid=invalid),
        })

        print(img_id, "outside pixels (gt==0):", int((gt == 0).sum()))

    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    col_titles = [letters[i] if i < len(letters) else f"({i})" for i in range(len(columns))]

    row_labels = [
        "Satellite\nImage",
        "Ground\nTruth",
        "Stage 1:\nBaseline",
        "Stage 2:\n+Replication",
        "Stage 3a:\n+OEM\nPre-Training",
        "Stage 3b:\n+OEM\nFine-Tuning",
        "Stage 4:\n+Hard × Minority\nSampling",
        "Stage 5:\n+Knowledge\nDistillation",
    ]

    handles = make_legend_handles(include_background=True)

    fig = make_grid_figure(columns, col_titles, row_labels, handles,
                           annotations=HIGHLIGHT_ANNOTATIONS)

    out = (repo_root / args.out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("Saved:", out)


if __name__ == "__main__":
    main()
