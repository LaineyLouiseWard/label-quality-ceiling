#!/usr/bin/env python3
"""
scripts/figures/Figure11.py

Quantify how Stage 3 (clsbal; final model) changes predictions vs Stage 1 baseline,
per class, over an entire split.

For each foreground class c (1..5), compute on GT pixels (gt==c), excluding gt==0:
  FIX  = baseline wrong  & final correct
  BREAK= baseline correct & final wrong

Reported as percentages of GT pixels of that class.

Outputs:
  figures/Figure11.pdf

Notes:
  - Uses BiodiversityValDataset / BiodiversityTestWithMasksDataset (same as Figure08.py).
  - Uses canonical dataset CLASSES + PALETTE (exact colours).
  - Paper naming: "Stage 3" (clsbal; final model) corresponds to repo checkpoint folder "stage3_clsbal".
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Repo discovery + imports
# -----------------------------------------------------------------------------
def find_repo_root_for_imports() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir():
            return parent
    raise RuntimeError("Could not find repo root for imports")


repo_root = find_repo_root_for_imports()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from geoseg.datasets.biodiversity_dataset import (  # noqa: E402
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)
from geoseg.models.ftunetformer import ft_unetformer  # noqa: E402


# -----------------------------------------------------------------------------
# Plot style (match your figure scripts)
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}",
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# -----------------------------------------------------------------------------
# Classes & palette (canonical)
# -----------------------------------------------------------------------------
CLASS_NAMES: Dict[int, str] = {i: n for i, n in enumerate(CLASSES)}
if 5 in CLASS_NAMES:
    CLASS_NAMES[5] = "Semi-nat."  # display tweak only

PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}

# Foreground classes only (exclude Background=0)
FOREGROUND_IDS = [1, 2, 3, 4, 5]
FOREGROUND_LABELS = [CLASS_NAMES[i] for i in FOREGROUND_IDS]

# Exact class colours (0–1)
CLASS_COLORS = {CLASS_NAMES[i]: (np.array(PALETTE[i], dtype=np.float32) / 255.0) for i in FOREGROUND_IDS}


# -----------------------------------------------------------------------------
# Checkpoint helpers (same logic as Figure08.py)
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


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=6, pretrained=False)


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


# -----------------------------------------------------------------------------
# Dataset iterator (val/test)
# -----------------------------------------------------------------------------
def iter_split_items(split_root: Path):
    """
    Yields (img_t, gt_t) for each tile in split.
    """
    if split_root.name == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
        for i in range(len(ds)):
            item = ds[i]
            yield item["img"], item["gt_semantic_seg"]
        return

    if split_root.name == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))
        for i in range(len(ds)):
            item = ds[i]
            yield item["img"], item["gt_semantic_seg"]
        return

    raise ValueError(f"Unsupported split: {split_root.name}")


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------
def compute_fix_break_rates(
    data_root: Path,
    split: str,
    net_baseline: torch.nn.Module,
    net_final: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Returns:
      fix_pct[class_name]   = 100 * fixes / GT_pixels(class)
      break_pct[class_name] = 100 * breaks / GT_pixels(class)
      gt_counts[class_name] = GT_pixels(class)  (for sanity/debug)
    """
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    # Accumulators on GT pixels only (exclude gt==0)
    gt_counts = {CLASS_NAMES[i]: 0 for i in FOREGROUND_IDS}
    fix_counts = {CLASS_NAMES[i]: 0 for i in FOREGROUND_IDS}
    break_counts = {CLASS_NAMES[i]: 0 for i in FOREGROUND_IDS}

    for img_t, gt_t in iter_split_items(split_root):
        gt = gt_t.detach().cpu().numpy().astype(np.uint8)
        valid = (gt != 0)

        if not np.any(valid):
            continue

        pred_base = predict_mask(net_baseline, img_t, device)
        pred_final = predict_mask(net_final, img_t, device)

        # Evaluate per class on GT pixels
        for cid in FOREGROUND_IDS:
            cls_name = CLASS_NAMES[cid]
            gt_c = (gt == cid) & valid
            n = int(gt_c.sum())
            if n == 0:
                continue

            b_correct = (pred_base == gt) & gt_c
            f_correct = (pred_final == gt) & gt_c

            fixes = int((~b_correct & f_correct).sum())     # baseline wrong -> final correct
            breaks = int((b_correct & ~f_correct).sum())    # baseline correct -> final wrong

            gt_counts[cls_name] += n
            fix_counts[cls_name] += fixes
            break_counts[cls_name] += breaks

    fix_pct = {}
    break_pct = {}
    for cid in FOREGROUND_IDS:
        cls = CLASS_NAMES[cid]
        denom = max(1, gt_counts[cls])
        fix_pct[cls] = 100.0 * fix_counts[cls] / denom
        break_pct[cls] = 100.0 * break_counts[cls] / denom

    return fix_pct, break_pct, gt_counts


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
def plot_fix_break_bars(
    fix_pct: Dict[str, float],
    break_pct: Dict[str, float],
    out_pdf: Path,
    title: str,
) -> None:
    labels = FOREGROUND_LABELS
    x = np.arange(len(labels))
    width = 0.36

    fig = plt.figure(figsize=(10.2, 4.9), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    # Same colour per class; differentiate via hatch
    fixes = [fix_pct[l] for l in labels]
    breaks = [break_pct[l] for l in labels]
    colours = [CLASS_COLORS[l] for l in labels]

    # FIX bars (solid)
    ax.bar(
        x - width / 2,
        fixes,
        width=width,
        color=colours,
        edgecolor="black",
        linewidth=0.6,
        label="Final model fixes baseline errors",
    )

    # BREAK bars (hatched, same colour)
    ax.bar(
        x + width / 2,
        breaks,
        width=width,
        color=colours,
        edgecolor="black",
        linewidth=0.6,
        hatch="///",
        alpha=0.95,
        label="Final model introduces new errors",
    )

    # Net improvement annotations above each class pair
    for i, l in enumerate(labels):
        net = fix_pct[l] - break_pct[l]
        bar_top = max(fixes[i], breaks[i])
        sign = "+" if net >= 0 else ""
        ax.text(
            x[i], bar_top + 0.6, f"{sign}{net:.1f} pp",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Pixels (\\% of GT class pixels)")
    ax.set_title(title)

    # Horizontal gridlines only
    ax.yaxis.grid(True, linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(False)

    # Legend: 2 columns, no frame
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    print("Saved:", out_pdf)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/biodiversity_split", help="Root containing val/ and test/")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Baseline and final (clsbal) checkpoints (match your repo folders)
    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline")
    ap.add_argument("--stage3-ckpt", default="model_weights/biodiversity/stage3_clsbal")  # paper Stage 3 (final)

    ap.add_argument("--out-pdf", default="figures/Figure11.pdf")

    args = ap.parse_args()

    set_plot_style()
    rr = find_repo_root()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data_root = (rr / args.data_root).resolve()
    ckpt_s1 = resolve_ckpt(str((rr / args.stage1_ckpt).resolve()))
    ckpt_final = resolve_ckpt(str((rr / args.stage3_ckpt).resolve()))

    print("Repo root:", rr)
    print("Data root:", data_root)
    print("Split:", args.split)
    print("Stage 1 ckpt:", ckpt_s1)
    print("Stage 3 (repo stage3_clsbal; final) ckpt:", ckpt_final)

    net_s1 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s1).to(device)
    net_final = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_final).to(device)

    fix_pct, break_pct, gt_counts = compute_fix_break_rates(
        data_root=data_root,
        split=args.split,
        net_baseline=net_s1,
        net_final=net_final,
        device=device,
    )

    print("GT pixel counts (by class):")
    for cls in FOREGROUND_LABELS:
        print(f"  {cls:12s}: {gt_counts[cls]}")

    out_pdf = (rr / args.out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    #title = f"Baseline (Stage 1) → final (Stage 3 clsbal) pixel transitions per class ({args.split} split)"
    plot_fix_break_bars(fix_pct, break_pct, out_pdf, title=None)


if __name__ == "__main__":
    main()
