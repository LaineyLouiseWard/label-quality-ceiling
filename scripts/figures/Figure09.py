#!/usr/bin/env python3
"""
scripts/figures/Figure09.py

Plot per-class IoU vs ablation stage (Figure-ready).

Default behaviour:
  - Reads per-stage metrics.json under:
      evaluation/evaluation_results/val/<stage_folder>/metrics.json
  - Extracts per-class IoU for the 5 foreground classes (no Background).
  - Produces a single line plot (5 lines).

Fallback:
  - If metrics.json are missing or incompatible, uses Table 2 values embedded below.

Outputs:
  figures/Figure09.pdf
  figures/Figure09.png
"""

from __future__ import annotations

import json
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

VAL_DIR = repo_root / "evaluation" / "evaluation_results" / "val"
OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUT_DIR / "Figure09.pdf"
OUT_PNG = OUT_DIR / "Figure09.png"

print("Repo root:", repo_root)
print("Val metrics dir:", VAL_DIR)
print("Output (pdf):", OUT_PDF)

# -----------------------
# Style (match your figure scripts)
# -----------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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
# NOTE: per your request, 3a is omitted (no transient dip shown).
STAGES: List[Tuple[str, str]] = [
    ("1", "stage1_baseline"),
    ("2", "stage2_replication"),
    ("3b", "stage3b_finetune"),
    ("4", "stage4_sampling"),
    ("5", "stage6_kd/stage6_kd"),  # metrics nested under stage6_kd/stage6_kd/ on disk
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
# Robust metrics.json reading
# -----------------------
def read_metrics_json(stage_folder: str) -> Dict:
    p = VAL_DIR / stage_folder / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_class_iou(metrics: Dict) -> Dict[str, float]:
    """
    Try a few common schemas for per-class IoU.
    Expected output: dict mapping class name -> IoU (0-100 or 0-1; we standardise to %).
    """
    for key in ["class_iou", "iou_per_class", "per_class_iou", "iou"]:
        if key in metrics and isinstance(metrics[key], dict):
            d = metrics[key]
            break
    else:
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            return extract_class_iou(metrics["metrics"])
        raise KeyError("Could not find per-class IoU dict in metrics.json")

    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue

    # Convert to percent if values look like fractions
    if out and max(out.values()) <= 1.0:
        out = {k: 100.0 * v for k, v in out.items()}
    return out


def normalise_keys(d: Dict[str, float]) -> Dict[str, float]:
    """Normalise class-name keys so we can match our CLASSES list robustly."""
    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "-").replace(" ", "-")

    return {norm(k): float(v) for k, v in d.items()}


def build_from_json() -> Dict[str, Dict[str, float]]:
    """
    Returns:
      stage_label -> {class_name -> iou_percent}
    """
    stage_to_iou: Dict[str, Dict[str, float]] = {}

    for stage_label, folder in STAGES:
        m = read_metrics_json(folder)
        iou_raw = extract_class_iou(m)
        iou = normalise_keys(iou_raw)

        # Map our exact plotted names -> possible keys in metrics.json
        # (This keeps colours mapped correctly because we always return keys from CLASSES.)
        key_candidates = {
            "Forest land": ["forest-land", "forest", "forestland"],
            "Grassland":   ["grassland"],
            "Cropland":    ["cropland", "agricultural-land", "agriculture", "agri"],
            "Settlement":  ["settlement", "built-up", "builtup"],
            "Semi-nat.":   ["semi-nat.", "semi-nat", "semi-natural", "semi-natural-grassland"],
        }

        out: Dict[str, float] = {}
        for cls in CLASSES:
            found = None
            for cand in key_candidates[cls]:
                if cand in iou:
                    found = iou[cand]
                    break
            if found is None:
                raise KeyError(
                    f"Could not match class '{cls}' in {folder}/metrics.json keys. "
                    f"Example keys: {list(iou.keys())[:20]}"
                )
            out[cls] = float(found)

        stage_to_iou[stage_label] = out

    return stage_to_iou



def plot_iou_trends(stage_to_iou: Dict[str, Dict[str, float]]) -> None:
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
            color=CLASS_COLORS[cls],     # <-- exact palette mapping
            linestyle=LINESTYLE[cls],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Stage", fontsize=16)
    ax.set_ylabel("IoU (%)", fontsize=16)

    # Horizontal gridlines only
    ax.yaxis.grid(True, linewidth=0.7, alpha=0.35)
    ax.xaxis.grid(False)

    # Y limits (since 3a removed)
    ax.set_ylim(55, 100)

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
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("Saved:", OUT_PDF)
    print("Saved:", OUT_PNG)


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    stage_to_iou = build_from_json()
    print("Loaded IoU from metrics.json.")
    plot_iou_trends(stage_to_iou)
