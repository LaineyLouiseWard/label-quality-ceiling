#!/usr/bin/env python3
"""
scripts/figures/frequency_vs_difficulty.py

N1 -- the hook for the label-ceiling narrative: per-class segmentation
difficulty is NOT explained by class rarity.

One point per foreground class:
  x = class frequency (% of foreground pixels in the training set, log scale)
  y = per-class IoU at the pretrained baseline, mean +/- 1 SD over the 10-seed
      campaign (seeds 42-51).

The seed rule: IoU is a *performance* metric,
so it is computed per seed and then averaged across seeds -- NOT pooled from an
ensemble. The per-class baseline IoU comes from the canonical 219-tile Irish
evaluation (eval_on_dumps_219.py: per-seed per-class IoU averaged over the 10
seeds, +/-1 SD bars) -- the single source shared with Fig 8 and N2, free of the
foreign-tile contamination that was in final_results.

The frequency axis is recomputed from the training masks so nothing is
hard-coded; it reproduces the locked figures (Grassland 68.4, Forest 14.6,
Semi-nat. 8.4, Cropland 5.2, Settlement 3.5 %).

A faint OLS line (IoU ~ log10 freq) shows how weak the rarity trend is: Forest
(common but hard) and Settlement (rare and hard) sit well off it, while Cropland
(rare but easy) sits above it -- killing the monotone-rarity read on sight.

All five foreground classes are shown; the per-class lens is motivated up front
by ODOS's operational need for usable accuracy on every land-cover type. We do
NOT add a defensive caption disclaimer about class selection -- that would invite
the HARKing read it tries to deflect.

Data:
  data/biodiversity_split/train/masks/*.png       (frequency axis)
  analysis/eval_219/per_class_iou.json             (per-class IoU axis, canonical)

Output:
  figures/frequency_vs_difficulty.pdf  (+ .png for quick QC)

  NB this is a NEW-narrative (label-ceiling) figure; it is intentionally NOT yet
  wired into build_all_figures.py, which still builds the pre-reframe set.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Canonical class order: Background=0, Forest=1, Grassland=2, Cropland=3,
# Settlement=4, Seminatural=5 (geoseg/datasets/biodiversity_dataset.py).
CLASS_IDS = [1, 2, 3, 4, 5]
# id -> (display name, analysis-JSON key, palette RGB) -- palette matches
# class_distributions.py / oem_mapping.tex so the figures read as a set. The JSON key is
# the STUDENT_CLASSES name used by the boundary analysis (shared 219-tile evaluation).
CLASS_INFO = {
    1: ("Forest", "Forest", (250, 62, 119)),
    2: ("Grassland", "Grassland", (168, 232, 84)),
    3: ("Cropland", "Cropland", (242, 180, 92)),
    4: ("Settlement", "Settlement", (59, 141, 247)),
    5: ("Semi-natural", "Seminatural", (255, 214, 33)),
}

# Per-class label placement (points, dx/dy from the marker) + horizontal alignment.
LABEL_OFFSET = {
    1: (8, 6),     # Forest
    2: (-8, 8),    # Grassland (rightmost point -> label sits to its left, inside the axes)
    3: (8, 6),     # Cropland
    4: (9, 9),     # Settlement
    5: (8, 6),     # Semi-natural
}
LABEL_HA = {2: "right"}  # Grassland anchored on the right so text runs leftward


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "data").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not find repo root")


def setup_font(use_tex: bool):
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 13,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def class_frequencies(root: Path) -> dict[int, float]:
    """% of foreground pixels per class over the training masks."""
    counts = np.zeros(6, dtype=np.int64)
    files = sorted(glob.glob(str(root / "data/biodiversity_split/train/masks/*.png")))
    if not files:
        raise FileNotFoundError("no training masks under data/biodiversity_split/train/masks")
    for f in files:
        a = np.asarray(Image.open(f))
        counts += np.bincount(a.ravel(), minlength=6)[:6]
    fg = counts[1:].sum()
    return {c: 100.0 * counts[c] / fg for c in CLASS_IDS}


def per_seed_iou(root: Path) -> dict[int, tuple[float, float, int]]:
    """class id -> (per-seed-mean IoU, across-seed SD, n_seeds) at the pretrained baseline.

    Reads the CANONICAL 219-tile evaluation (eval_on_dumps_219.py), the single source of
    truth for per-class IoU shared with Fig 8 (confusion) and N2 (it matches the boundary
    standard_iou exactly). Irish-only, contamination-free."""
    p = root / "analysis/eval_219/per_class_iou.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run scripts/analysis/eval_on_dumps_219.py first")
    ev = json.load(open(p))["stage1_baseline"]
    n = ev["n_seeds"]
    out = {}
    for c in CLASS_IDS:
        key = CLASS_INFO[c][1]
        out[c] = (ev["per_class_iou_mean"][key], ev["per_class_iou_std"][key], n)
    return out


def render(root: Path, out_dir: Path, use_tex: bool):
    setup_font(use_tex)
    freq = class_frequencies(root)
    iou = per_seed_iou(root)
    n_seeds = next(iter(iou.values()))[2]

    x = np.array([freq[c] for c in CLASS_IDS])
    y = np.array([iou[c][0] for c in CLASS_IDS])
    yerr = np.array([iou[c][1] for c in CLASS_IDS])
    logx = np.log10(x)

    # Weak-trend guide: OLS of IoU on log10(frequency). n=5, descriptive only.
    slope, intercept = np.polyfit(logx, y, 1)
    yhat = slope * logx + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    xx = np.linspace(logx.min() - 0.08, logx.max() + 0.08, 100)
    ax.plot(10 ** xx, slope * xx + intercept, ls="--", lw=1.1, color="#888888", zorder=1)

    for c in CLASS_IDS:
        name, _, rgb = CLASS_INFO[c]
        col = np.array(rgb) / 255.0
        ax.errorbar(freq[c], iou[c][0], yerr=iou[c][1], fmt="o", ms=9,
                    color=col, mec="black", mew=0.7, ecolor="black",
                    elinewidth=0.9, capsize=2.5, zorder=3)
        dx, dy = LABEL_OFFSET[c]
        ax.annotate(name, (freq[c], iou[c][0]), textcoords="offset points",
                    xytext=(dx, dy), ha=LABEL_HA.get(c, "left"), fontsize=10.5, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks([3, 5, 10, 20, 50])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(2.7, 90)
    ax.set_ylim(0.76, 1.0)
    ax.set_xlabel(r"Class frequency (\% of foreground pixels, log scale)"
                  if use_tex else "Class frequency (% of foreground pixels, log scale)")
    ax.set_ylabel("Baseline per-class IoU")
    ax.grid(True, which="both", ls=":", lw=0.5, color="#cccccc", zorder=0)

    ax.text(0.015, -0.155, "rare", transform=ax.transAxes, fontsize=12, color="#333333")
    ax.text(0.93, -0.155, "common", transform=ax.transAxes, fontsize=12, color="#333333")

    r2_str = (rf"$R^2 = {r2:.2f}$" if use_tex else f"R^2 = {r2:.2f}")
    ax.text(0.97, 0.05, r2_str, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="#555555")
    fig.tight_layout()
    fig.canvas.draw()  # finalise layout so the fit-line angle is measured in display space
    _xa, _xb = xx[25], xx[72]
    _pa = ax.transData.transform((10 ** _xa, slope * _xa + intercept))
    _pb = ax.transData.transform((10 ** _xb, slope * _xb + intercept))
    _ang = float(np.degrees(np.arctan2(_pb[1] - _pa[1], _pb[0] - _pa[0])))
    _xm = xx[70]
    ax.text(10 ** _xm, slope * _xm + intercept, "OLS fit", rotation=_ang,
            rotation_mode="anchor", ha="center", va="center", fontsize=10,
            color="#888888", zorder=2,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "frequency_vs_difficulty.pdf"
    png = out_dir / "frequency_vs_difficulty.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[frequency_vs_difficulty] {n_seeds} seeds; OLS slope={slope:+.3f} R^2={r2:.3f}")
    for c in CLASS_IDS:
        m, sd, _ = iou[c]
        print(f"  {CLASS_INFO[c][0]:<13} freq={freq[c]:5.2f}%  IoU={m:.3f}+-{sd:.3f}")
    print(f"  -> {pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()

    root = find_repo_root()
    out_dir = (root / args.out_dir).resolve()
    use_tex = not args.no_tex
    try:
        render(root, out_dir, use_tex)
    except Exception as e:
        if not use_tex:
            raise
        print(f"[frequency_vs_difficulty] usetex failed ({e}); retrying with mathtext")
        render(root, out_dir, use_tex=False)


if __name__ == "__main__":
    main()
