#!/usr/bin/env python3
"""
scripts/figures/Figure11.py

Pixel-level GT-conditioned transitions from the baseline to the full model, per class,
aggregated over the 10-seed campaign (validation set).

For each foreground class c (1..5), on GT pixels of that class (gt==c, excluding gt==0),
and for each seed s = 42..51 independently:

  FIX(c,s)   = 100 * (baseline wrong  & full correct) / GT_pixels(c)   -> recovered
  BREAK(c,s) = 100 * (baseline correct & full wrong)  / GT_pixels(c)   -> lost
  NET(c,s)   = FIX(c,s) - BREAK(c,s)

Each is reported as the per-seed mean with a 95 % paired-t confidence interval over the
ten seeds (replicates), matching the seed convention used throughout the manuscript.

This figure is recall / GT-conditioned: it tells you, of the pixels that truly belong to a
class, what fraction the full model recovers vs newly loses. It says nothing about
precision (false positives elsewhere) -- that asymmetry lives in the confusion matrices
(Figure 9).

Inputs (all local, no checkpoints / no inference):
  sonic/results/seed{42..51}/analysis/seed_softmax/{stage1_baseline,stage3_clsbal}/seed{seed}/<tile>.npy
  data/biodiversity_split/val/masks/<tile>.png

Writes:
  figures/Figure11.pdf

Run:
  python scripts/figures/Figure11.py
"""

from __future__ import annotations

import sys
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats


# -----------------------------------------------------------------------------
# Repo discovery
# -----------------------------------------------------------------------------
def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not find repo root")


repo_root = find_repo_root()

SEEDS = list(range(42, 52))                  # 42..51, 10 seeds
CELL_BASELINE = "stage1_baseline"
CELL_FULL = "stage3_clsbal"
SOFTMAX_ROOT = repo_root / "sonic/results"
MASK_DIR = repo_root / "data/biodiversity_split/val/masks"

OUT_PDF = repo_root / "figures/Figure11.pdf"

# Canonical class order (geoseg/datasets/biodiversity_dataset.py): foreground 1..5.
FOREGROUND_IDS = [1, 2, 3, 4, 5]
CLASS_LABELS = {1: "Forest", 2: "Grassland", 3: "Cropland", 4: "Settlement", 5: "Semi-nat."}
RARE = {"Settlement", "Semi-nat."}


# -----------------------------------------------------------------------------
# Plot style (match the other figure scripts)
# -----------------------------------------------------------------------------
def set_plot_style(use_tex: bool = True) -> None:
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{lmodern}"
    else:
        rc["text.usetex"] = False
    plt.rcParams.update(rc)


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------
def softmax_dir(cell: str, seed: int) -> Path:
    return SOFTMAX_ROOT / f"seed{seed}" / "analysis" / "seed_softmax" / cell / f"seed{seed}"


def argmax_pred(npy_path: Path) -> np.ndarray:
    """Load a (6,H,W) softmax and return the (H,W) argmax label map."""
    return np.load(npy_path).astype(np.float32).argmax(axis=0).astype(np.uint8)


def fix_break_per_seed(seed: int):
    """Return {class_name: (fix_pct, break_pct)} for one seed over all val tiles."""
    base_dir = softmax_dir(CELL_BASELINE, seed)
    full_dir = softmax_dir(CELL_FULL, seed)

    gt_counts = {c: 0 for c in FOREGROUND_IDS}
    fix_counts = {c: 0 for c in FOREGROUND_IDS}
    break_counts = {c: 0 for c in FOREGROUND_IDS}

    tiles = sorted(p.stem for p in base_dir.glob("*.npy"))
    for tile in tiles:
        gt = imageio.imread(MASK_DIR / f"{tile}.png").astype(np.uint8)
        pred_b = argmax_pred(base_dir / f"{tile}.npy")
        pred_f = argmax_pred(full_dir / f"{tile}.npy")

        b_correct = pred_b == gt
        f_correct = pred_f == gt
        for c in FOREGROUND_IDS:
            gt_c = gt == c
            n = int(gt_c.sum())
            if n == 0:
                continue
            gt_counts[c] += n
            fix_counts[c] += int((~b_correct & f_correct & gt_c).sum())
            break_counts[c] += int((b_correct & ~f_correct & gt_c).sum())

    out = {}
    for c in FOREGROUND_IDS:
        denom = max(1, gt_counts[c])
        out[CLASS_LABELS[c]] = (
            100.0 * fix_counts[c] / denom,
            100.0 * break_counts[c] / denom,
        )
    return out


def paired_ci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    n = x.size
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, h


def collect():
    """Per class: arrays of length len(SEEDS) for fix, break, net."""
    fix = defaultdict(list)
    brk = defaultdict(list)
    for seed in SEEDS:
        per = fix_break_per_seed(seed)
        for cls, (f, b) in per.items():
            fix[cls].append(f)
            brk[cls].append(b)
        print(f"  seed {seed}: done")
    stats_out = {}
    for cls in (CLASS_LABELS[c] for c in FOREGROUND_IDS):
        f = np.array(fix[cls])
        b = np.array(brk[cls])
        net = f - b
        stats_out[cls] = {
            "fix": paired_ci(f),
            "break": paired_ci(b),
            "net": paired_ci(net),
        }
    return stats_out


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
FIX_COLOR = "#2166AC"      # recovered  (steel blue, solid)
BREAK_COLOR = "#D6604D"    # lost       (brick red, hatched -> greyscale-safe)


def plot(stats_out, out_pdf: Path):
    labels = [CLASS_LABELS[c] for c in FOREGROUND_IDS]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.6, 5.0))

    fix_m = np.array([stats_out[l]["fix"][0] for l in labels])
    fix_e = np.array([stats_out[l]["fix"][1] for l in labels])
    brk_m = np.array([stats_out[l]["break"][0] for l in labels])
    brk_e = np.array([stats_out[l]["break"][1] for l in labels])
    net_m = np.array([stats_out[l]["net"][0] for l in labels])
    net_e = np.array([stats_out[l]["net"][1] for l in labels])

    ax.bar(x - width / 2, fix_m, width, yerr=fix_e, capsize=4,
           color=FIX_COLOR, edgecolor="black", linewidth=0.7,
           error_kw=dict(elinewidth=1.2),
           label="Recovered (baseline wrong $\\rightarrow$ full correct)")
    ax.bar(x + width / 2, brk_m, width, yerr=brk_e, capsize=4,
           color="white", edgecolor=BREAK_COLOR, linewidth=1.1, hatch="////",
           error_kw=dict(elinewidth=1.2, ecolor="black"),
           label="Lost (baseline correct $\\rightarrow$ full wrong)")

    # net effect as a black diamond with its own paired CI, offset above the pair
    ax.errorbar(x, net_m, yerr=net_e, fmt="D", color="black", markersize=6.5,
                capsize=4, elinewidth=1.2, zorder=5,
                label="Net change (recovered $-$ lost)")

    ax.axhline(0.0, color="black", lw=1.0, zorder=1)

    ax.set_xticks(x)
    xlabels = [(r"\textbf{%s}" % l if (plt.rcParams["text.usetex"] and l in RARE) else l)
               for l in labels]
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("GT class pixels (\\%)")

    ax.yaxis.grid(True, linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, loc="upper left", ncol=1)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved:", out_pdf)

    # also dump the numbers for caption/audit
    print("\nPer-class transitions (mean over 10 seeds, +/- 95% paired-t CI):")
    for l in labels:
        fm, fe = stats_out[l]["fix"]
        bm, be = stats_out[l]["break"]
        nm, ne = stats_out[l]["net"]
        print(f"  {l:11s}  fix {fm:5.2f}+/-{fe:4.2f}   break {bm:5.2f}+/-{be:4.2f}   net {nm:+5.2f}+/-{ne:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()

    print("Computing 10-seed FIX/BREAK from stored softmax...")
    stats_out = collect()

    try:
        set_plot_style(use_tex=not args.no_tex)
        plot(stats_out, OUT_PDF)
    except Exception as e:
        if args.no_tex:
            raise
        print(f"usetex failed ({e}); retrying with mathtext")
        set_plot_style(use_tex=False)
        plot(stats_out, OUT_PDF)


if __name__ == "__main__":
    main()
