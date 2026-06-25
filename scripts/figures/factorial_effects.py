#!/usr/bin/env python3
"""
scripts/figures/factorial_effects.py

Per-class main-effects forest plot for the 2x2 factorial (validation set).

For each metric (overall mIoU + the five foreground classes) the figure shows the three
factorial contrasts as point estimates with 95 % paired-t confidence intervals over the ten
seeds, on a shared signed percentage-point axis with a bold zero reference line:

  * OEM transfer main effect       = mean[ (transfer-only - baseline) + (full - sampler-only) ] / 2
  * clsbal sampler main effect     = mean[ (sampler-only - baseline) + (full - transfer-only) ] / 2
  * transfer x sampler interaction = (full - sampler-only) - (transfer-only - baseline)

This is the faithful encoding of the factorial design: the result IS the paired contrasts
(transfer dominant and positive on every class; sampler a rare-class-for-cropland trade-off;
interaction CIs straddling zero = additive), not cell levels on a forced axis. Cell levels
live in Table 2.

Data: analysis/seed_aggregate/per_seed_metrics.csv (all 4 cells x 6 metrics x 10 seeds, val).
Writes: figures/factorial_effects.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "analysis").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not find repo root")


# metric label -> (csv 'metric', csv 'class')
METRICS = [
    ("mIoU", "mIoU", ""),
    ("Forest", "IoU", "Forest"),
    ("Grassland", "IoU", "Grassland"),
    ("Cropland", "IoU", "Cropland"),
    ("Settlement", "IoU", "Settlement"),
    ("Semi-natural", "IoU", "Semi-natural"),
]
RARE = {"Settlement", "Semi-natural"}
CELLS = ["baseline", "transfer-only", "sampler-only", "full"]

# effect label, colour, marker
EFFECTS = [
    ("OEM transfer", "#2166AC", "o"),
    ("clsbal sampler", "#D6604D", "s"),
    ("Transfer $\\times$ sampler", "#777777", "D"),
]


def setup_font(use_tex: bool):
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 13,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def load_per_seed(csv_path: Path, split: str):
    """(metric, class, cell) -> {seed: value}."""
    D = defaultdict(dict)
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != split:
                continue
            D[(r["metric"], r["class"], r["cell_label"])][int(r["seed"])] = float(r["value_pct"])
    return D


def paired_ci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    n = x.size
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def factorial_effects(D, metric_csv, cls):
    cells = {c: D[(metric_csv, cls, c)] for c in CELLS}
    seeds = sorted(set.intersection(*[set(v) for v in cells.values()]))
    if not seeds:
        raise ValueError(f"no shared seeds for {metric_csv}/{cls}")
    b = np.array([cells["baseline"][s] for s in seeds])
    t = np.array([cells["transfer-only"][s] for s in seeds])
    sa = np.array([cells["sampler-only"][s] for s in seeds])
    f = np.array([cells["full"][s] for s in seeds])
    return {
        "OEM transfer": paired_ci(((t - b) + (f - sa)) / 2),
        "clsbal sampler": paired_ci(((sa - b) + (f - t)) / 2),
        "Transfer $\\times$ sampler": paired_ci((f - sa) - (t - b)),
    }, len(seeds)


def render(csv_path: Path, split: str, out_dir: Path, use_tex: bool):
    setup_font(use_tex)
    D = load_per_seed(csv_path, split)

    n_metrics = len(METRICS)
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)

    # one band per metric, mIoU at the top -> descending y
    band_y = {lab: (n_metrics - 1 - i) for i, (lab, _, _) in enumerate(METRICS)}
    offsets = {EFFECTS[0][0]: 0.26, EFFECTS[1][0]: 0.0, EFFECTS[2][0]: -0.26}

    for lab, _, _ in METRICS:                       # shade rare-class bands
        if lab in RARE:
            y = band_y[lab]
            ax.axhspan(y - 0.45, y + 0.45, color="0.93", zorder=0)
    ax.axvline(0.0, color="black", lw=1.2, zorder=1)

    n_seeds = None
    for lab, mcsv, cls in METRICS:
        eff, n_seeds = factorial_effects(D, mcsv, cls)
        y0 = band_y[lab]
        for ename, color, marker in EFFECTS:
            m, lo, hi = eff[ename]
            ax.errorbar(m, y0 + offsets[ename], xerr=[[m - lo], [hi - m]],
                        fmt=marker, color=color, ecolor=color, elinewidth=1.8,
                        capsize=3.5, markersize=6.5, zorder=3)

    ax.set_yticks([band_y[lab] for lab, _, _ in METRICS])
    ylabels = [(r"\textbf{%s}" % lab if (use_tex and lab in RARE) else lab)
               for lab, _, _ in METRICS]
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.6, n_metrics - 0.4)
    ax.set_xlabel("Factorial effect on IoU (percentage points)")

    ax.grid(axis="x", color="0.85", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)

    handles = [Patch(facecolor=c, edgecolor="none", label=lab) for lab, c, _ in EFFECTS]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="0.8")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "factorial_effects.pdf"
    fig.savefig(pdf, bbox_inches="tight")   # PDF only: figures/ is PDF-only by repo convention
    plt.close(fig)
    print(f"[factorial_effects] {split}: {n_seeds} seeds; wrote {pdf}")
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/seed_aggregate/per_seed_metrics.csv")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    csv_path = (root / args.csv).resolve()
    out_dir = (root / args.out_dir).resolve()
    use_tex = not args.no_tex
    try:
        render(csv_path, args.split, out_dir, use_tex)
    except Exception as e:
        if not use_tex:
            raise
        print(f"[factorial_effects] usetex failed ({e}); retrying with mathtext")
        render(csv_path, args.split, out_dir, use_tex=False)


if __name__ == "__main__":
    main()
