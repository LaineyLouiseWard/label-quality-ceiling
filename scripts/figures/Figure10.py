#!/usr/bin/env python3
"""
scripts/figures/Figure10.py

Plot per-class IoU vs ablation stage with seed-std bands (mean +/- std over seeds).

Default behaviour:
  - Reads the seed-aggregated CSV:
      analysis/seed_aggregate/figure10_iou_val.csv
    (columns: stage_folder, stage_label, class, iou_mean_pct, iou_std_pct, n_seeds;
    class carries display names "Forest land" / "Semi-nat.").
  - Extracts per-class IoU for the 5 foreground classes (no Background).
  - Produces a single line plot (5 lines): central line = seed mean (iou_mean_pct),
    shaded band = seed mean +/- std (iou_std_pct). No bootstrap CI.

Dependencies:
  - Run `python scripts/analysis/aggregate_seeds.py` first (over all 5 seeds) to
    generate analysis/seed_aggregate/figure10_iou_val.csv.

Outputs:
  figures/Figure10.pdf
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "geoseg").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")


# -----------------------
# Repo + paths
# -----------------------
repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SEED_AGG_CSV = repo_root / "analysis" / "seed_aggregate" / "figure10_iou_val.csv"
OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "Figure10.pdf"

print("Repo root:", repo_root)
print("Seed-aggregate CSV:", SEED_AGG_CSV)
print("Output (pdf):", OUT_PDF)

# -----------------------
# Style (match your figure scripts)
# -----------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# -----------------------
# Stages to plot (paper order)
# -----------------------
# NOTE: 2a/2b are merged into a single Stage 2 (deployable OEM-transfer fine-tuned model);
# the non-deployed combined-set pre-trained checkpoint is not shown as a separate stage.
STAGES: List[Tuple[str, str]] = [
    ("1", "stage1_baseline"),
    ("2", "stage2b_oem_finetune"),
    ("3", "stage3_clsbal"),
]

# -----------------------
# Palette + class names (MUST match your masks/figures)
# -----------------------
COLOR_MAP = {
    0: [0, 0, 0],
    1: [250, 62, 119],   # Forest land
    2: [168, 232, 84],   # Grassland
    3: [242, 180, 92],   # Cropland
    4: [59, 141, 247],   # Settlement
    5: [255, 214, 33],   # Semi-natural
}

CLASS_NAMES = [
    "Background", "Forest land", "Grassland", "Cropland",
    "Settlement", "Semi-nat."
]

# Foreground classes to plot (must match keys used in CLASS_COLORS)
CLASSES = ["Forest land", "Grassland", "Cropland", "Settlement", "Semi-nat."]

# Convert COLOR_MAP to matplotlib-ready colours (0–1 range)
CLASS_COLORS = {
    "Forest land": np.array(COLOR_MAP[1], dtype=float) / 255.0,
    "Grassland":  np.array(COLOR_MAP[2], dtype=float) / 255.0,
    "Cropland":   np.array(COLOR_MAP[3], dtype=float) / 255.0,
    "Settlement": np.array(COLOR_MAP[4], dtype=float) / 255.0,
    "Semi-nat.":  np.array(COLOR_MAP[5], dtype=float) / 255.0,
}

LINESTYLE = {
    "Forest land": "-",
    "Grassland": "-",
    "Cropland": "-",
    "Settlement": "--",
    "Semi-nat.": "--",
}

# -----------------------
# Seed-aggregate CSV reading (mean +/- std over the 5 seeds)
# -----------------------
def load_seed_aggregate() -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Read analysis/seed_aggregate/figure10_iou_val.csv.

    The CSV carries one row per (stage_folder, class) with the across-seed mean
    and std of per-class IoU (in percent), plus display class names.

    Returns:
        stage_to_iou: stage_label -> {class_name -> iou_mean_pct}
        stage_to_std: stage_label -> {class_name -> iou_std_pct}
    """
    if not SEED_AGG_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {SEED_AGG_CSV}. "
            f"Run: python scripts/analysis/aggregate_seeds.py (over all 5 seeds)."
        )

    folder_to_label = {folder: label for label, folder in STAGES}

    stage_to_iou: Dict[str, Dict[str, float]] = {label: {} for label, _ in STAGES}
    stage_to_std: Dict[str, Dict[str, float]] = {label: {} for label, _ in STAGES}

    with open(SEED_AGG_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = row["stage_folder"].strip()
            cls = row["class"].strip()
            if folder not in folder_to_label or cls not in CLASSES:
                continue
            label = folder_to_label[folder]
            stage_to_iou[label][cls] = float(row["iou_mean_pct"])
            stage_to_std[label][cls] = float(row["iou_std_pct"])

    # Validate completeness
    for label, _ in STAGES:
        for cls in CLASSES:
            if cls not in stage_to_iou[label]:
                raise KeyError(
                    f"Seed-aggregate CSV missing class '{cls}' for stage '{label}'. "
                    f"Columns/values present: {list(stage_to_iou[label].keys())}"
                )

    return stage_to_iou, stage_to_std


def plot_iou_trends(
    stage_to_iou: Dict[str, Dict[str, float]],
    stage_to_std: Dict[str, Dict[str, float]],
) -> None:
    stage_labels = [s for s, _ in STAGES if s in stage_to_iou]
    x = list(range(len(stage_labels)))

    fig = plt.figure(figsize=(8.6, 5.1), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    for cls in CLASSES:
        y = [stage_to_iou[s][cls] for s in stage_labels]
        ax.plot(
            x, y,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=cls,
            color=CLASS_COLORS[cls],
            linestyle=LINESTYLE[cls],
        )
        y_lo = [stage_to_iou[s][cls] - stage_to_std[s][cls] for s in stage_labels]
        y_hi = [stage_to_iou[s][cls] + stage_to_std[s][cls] for s in stage_labels]
        ax.fill_between(x, y_lo, y_hi, color=CLASS_COLORS[cls], alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Stage", fontsize=16)
    ax.set_ylabel(r"IoU (\%)", fontsize=16)

    # Horizontal gridlines only
    ax.yaxis.grid(True, linewidth=0.7, alpha=0.35)
    ax.xaxis.grid(False)

    # Y limits (since 3a removed)
    ax.set_ylim(40, 100)

    # Legend: multi-column (avoid single stacked column)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
        fontsize=14,
        columnspacing=1.2,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.6,
    )

    ax.tick_params(axis="both", which="major", labelsize=14)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("Saved:", OUT_PDF)


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    stage_to_iou, stage_to_std = load_seed_aggregate()
    print("Loaded seed-aggregated IoU (mean +/- std) from", SEED_AGG_CSV.name)
    plot_iou_trends(stage_to_iou, stage_to_std)
