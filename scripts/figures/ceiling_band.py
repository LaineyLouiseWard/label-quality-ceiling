#!/usr/bin/env python3
"""
N5 -- the closing synthesis figure. Per class, the curation levers (baseline -> full model)
are shown against a boundary-tolerant IoU band: the score the FULL model reaches if
near-boundary label disagreement (1-2 px = 0.5-1 m) is forgiven.

Reading:
  o  open marker  = baseline IoU (stage1_baseline)
  *  filled marker = full-model IoU (stage3_clsbal); the arrow is the curation gain
  shaded band     = full model's IoU at 1-2 px boundary tolerance (from the N2 trimap curve)

The gap between the full marker and the band is the BOUNDARY-LOCALISED residual -- error the
levers cannot remove because it sits in the label-ambiguous boundary shell. Grassland/Cropland
already sit in their band (near the achievable score); Forest/Settlement show a clear boundary
residual; Semi-natural's band is itself low (its N2c interior floor), so even boundary tolerance
cannot lift it far -- its wall is interior, not boundary.

FRAMING DISCIPLINE (PLOT_PLAN N5 / D.5). The band is labelled descriptively as a boundary-
tolerant IoU range, NOT "the ceiling", on the figure. The interpretation of the boundary-tolerant
score as an empirical PROXY for the label-ambiguity ceiling -- explicitly NOT a measured
inter-annotator bound, which we lack the multi-annotator data for -- belongs in the Discussion
caption only. The band is drawn as a 1-2 px RANGE so its width signals it is a tolerance choice,
not a hard line.

Data (no new computation; the same 219-tile trimap curves as N2):
  analysis/label_ceiling/boundary_trimap_stage1_baseline.json   (baseline IoU)
  analysis/label_ceiling/boundary_trimap_stage3_clsbal.json     (full IoU + 1-2 px band)

Output:
  figures/ceiling_band.pdf  (+ .png for QC)
  New-narrative figure; NOT in build_all_figures.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RGB = {
    "Forest": (250, 62, 119),
    "Grassland": (168, 232, 84),
    "Cropland": (242, 180, 92),
    "Settlement": (59, 141, 247),
    "Seminatural": (255, 214, 33),
}
SHORT = {"Forest": "Forest", "Grassland": "Grassland", "Cropland": "Cropland",
         "Settlement": "Settlement", "Seminatural": "Semi-natural"}
CLASSES = ["Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]


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
        "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 13, "font.size": 12, "legend.fontsize": 9.5,
        "xtick.labelsize": 11, "ytick.labelsize": 12, "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def col(name):
    return np.array(RGB[name]) / 255.0


def load(root):
    b = json.load(open(root / "analysis/label_ceiling/boundary_trimap_stage1_baseline.json"))["recovery_trimap"]
    f = json.load(open(root / "analysis/label_ceiling/boundary_trimap_stage3_clsbal.json"))["recovery_trimap"]
    rp = b["radii_px"]
    i0, i1px, i2px = rp.index(-1), rp.index(1), rp.index(2)
    rows = {}
    for c in CLASSES:
        rows[c] = dict(
            base=b["per_seed_class_iou_mean"][c][i0],
            full=f["per_seed_class_iou_mean"][c][i0],
            tol1=f["per_seed_class_iou_mean"][c][i1px],
            tol2=f["per_seed_class_iou_mean"][c][i2px],
        )
    return rows


def render(root, out_dir, use_tex):
    setup_font(use_tex)
    rows = load(root)
    order = sorted(CLASSES, key=lambda c: rows[c]["full"])  # ascending -> best on top
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    band_label_done = False
    for yi, c in zip(y, order):
        d = rows[c]
        cc = col(c)
        lo, hi = sorted((d["tol1"], d["tol2"]))
        ax.add_patch(plt.Rectangle((lo, yi - 0.3), hi - lo, 0.6, facecolor=cc, alpha=0.28,
                                   edgecolor=cc, lw=0.8, zorder=1,
                                   label=None if band_label_done else "_band"))
        band_label_done = True
        # baseline -> full arrow
        ax.annotate("", xy=(d["full"], yi), xytext=(d["base"], yi),
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4, shrinkA=3, shrinkB=3),
                    zorder=2)
        ax.plot(d["base"], yi, "o", ms=8, mfc="white", mec=cc, mew=1.6, zorder=3)
        ax.plot(d["full"], yi, "o", ms=9, mfc=cc, mec="black", mew=0.7, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[c] for c in order])
    ax.set_xlabel("per-class IoU")
    ax.set_xlim(0.50, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(True, axis="x", ls=":", lw=0.5, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    tol = (r"boundary-tolerant IoU (1--2\,px)" if use_tex
           else "boundary-tolerant IoU (1-2 px)")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", mew=1.6,
                   ms=8, label="baseline"),
        plt.Line2D([], [], marker="o", ls="none", mfc="#777777", mec="black", mew=0.7,
                   ms=9, label="full model"),
        Patch(facecolor="#999999", alpha=0.28, edgecolor="#999999", label=tol),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.4)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / "ceiling_band.pdf", out_dir / "ceiling_band.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[ceiling_band] -> {pdf}")
    for c in order:
        d = rows[c]
        print(f"  {SHORT[c]:13s} base={d['base']:.3f} full={d['full']:.3f} "
              f"band=[{min(d['tol1'],d['tol2']):.3f},{max(d['tol1'],d['tol2']):.3f}]  "
              f"boundary residual={min(d['tol1'],d['tol2'])-d['full']:+.3f}")


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
        print(f"[ceiling_band] usetex failed ({e}); retrying with mathtext")
        render(root, out_dir, use_tex=False)


if __name__ == "__main__":
    main()
