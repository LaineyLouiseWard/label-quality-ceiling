#!/usr/bin/env python
"""
N3 -- reliability diagram + ECE. Honest calibration characterisation: the model's confidences
are meaningful overall, and the boundary uncertainty is real but UNDER-captured by softmax alone
-- which is exactly what motivates the deep-ensemble measures in N4.

Two panels, ensemble mean softmax on the 219-tile Irish val set (seed rule D.2: calibration is
an ENSEMBLE quantity -> mean over the 10 seeds' softmax, then argmax/confidence):

  (a) Reliability diagram, baseline vs full model. Both hug the diagonal (ECE ~0.02) -> the model
      is well calibrated in aggregate; the residual error is NOT a gross calibration artefact.
      Foreground pixels only (ignore_index=0).
  (b) Calibration stratified by distance-to-GT-boundary (near <=1.5 m vs interior >8 m), baseline.
      The interior is on the diagonal (ECE ~0.02); near-boundary pixels are LOWER confidence
      (mean ~0.76 vs ~0.94 -- the model does flag boundaries as harder) but sit BELOW the diagonal
      (ECE ~0.09): the single/ensemble softmax stays MILDLY OVERCONFIDENT at boundaries. So
      max-softmax confidence under-states boundary ambiguity -> use the ensemble entropy/MI
      decomposition (N4) as the proper uncertainty signal there. Non-circular (GT boundaries only).

Precedent: Guo et al. (2017) reliability/ECE; Naeini et al. (2015) binned calibration error.

Data: per-seed softmax dumps via list_val_tiles (219 tiles).
Output: figures/reliability_ece.pdf (+ .png)
  New-narrative figure; NOT in build_all_figures.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.seed_disagreement import (  # noqa: E402
    list_val_tiles, load_mask, load_seed_stack, boundary_distance, GSD_M,
)

N_BINS = 15
EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
NEAR_M, FAR_M = 1.5, 8.0   # boundary band vs interior floor (matches N2)


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(10):
        if (p / "data").is_dir() and (p / "scripts").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    return Path(__file__).resolve().parents[2]


def setup_font(use_tex: bool):
    rc = {"font.family": "serif", "font.serif": ["Computer Modern Roman"],
          "axes.labelsize": 13, "font.size": 12, "axes.titlesize": 14,
          "legend.fontsize": 11.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
          "axes.axisbelow": True, "figure.dpi": 150}
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


class CalAccum:
    """Streaming reliability accumulators: per bin sum-confidence, sum-correct, count."""
    def __init__(self):
        self.sconf = np.zeros(N_BINS)
        self.scorr = np.zeros(N_BINS)
        self.n = np.zeros(N_BINS, dtype=np.int64)

    def add(self, conf, correct):
        b = np.clip(np.digitize(conf, EDGES[1:-1]), 0, N_BINS - 1)
        self.sconf += np.bincount(b, weights=conf, minlength=N_BINS)
        self.scorr += np.bincount(b, weights=correct.astype(float), minlength=N_BINS)
        self.n += np.bincount(b, minlength=N_BINS)

    def curve(self):
        n = np.where(self.n > 0, self.n, 1)
        return self.sconf / n, self.scorr / n, self.n

    def ece(self):
        conf, acc, n = self.curve()
        N = self.n.sum()
        w = self.n / (N if N else 1)
        return float(np.sum(w * np.abs(acc - conf)))


def run_cell(root, softmax_root, mask_dir, cell, seeds, stratify=False):
    ids, dropped = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    overall = CalAccum()
    near = CalAccum() if stratify else None
    far = CalAccum() if stratify else None
    for iid in ids:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)   # (N,C,H,W)
        mean_p = stack.mean(axis=0)                               # (C,H,W)
        conf = mean_p.max(axis=0)                                 # (H,W)
        pred = mean_p.argmax(axis=0)
        mask = load_mask(mask_dir, iid)
        fg = mask != 0
        correct = (pred == mask)
        overall.add(conf[fg], correct[fg])
        if stratify:
            dist = boundary_distance(mask) * GSD_M                # metres
            nb = fg & (dist <= NEAR_M)
            fr = fg & (dist > FAR_M)
            near.add(conf[nb], correct[nb])
            far.add(conf[fr], correct[fr])
    return overall, near, far, len(ids)


def render(root, out_dir, seeds, use_tex):
    setup_font(use_tex)
    sr, md = "sonic/results", "data/biodiversity_split/val/masks"
    base, _, _, n = run_cell(root, sr, md, "stage1_baseline", seeds, stratify=False)
    full, b_near, b_far, _ = run_cell(root, sr, md, "stage3_clsbal", seeds, stratify=True)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 4.4))

    # (a) baseline vs full reliability
    axA.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=1)
    for accum, name, c in [(base, "baseline", "#2166ac"), (full, "full model", "#b2182b")]:
        conf, acc, nb = accum.curve()
        m = nb > 0
        axA.plot(conf[m], acc[m], "-o", ms=4, lw=1.5, color=c,
                 label=f"{name} (ECE={accum.ece():.3f})", zorder=3)
    axA.set_xlim(0.45, 1.0); axA.set_ylim(0.45, 1.0)
    axA.set_xlabel("predicted confidence"); axA.set_ylabel("empirical accuracy")
    axA.set_title("(a)")
    axA.legend(loc="upper left", frameon=False)
    axA.grid(True, ls=":", lw=0.5, color="#cccccc"); axA.set_axisbelow(True)

    # (b) baseline stratified by distance-to-boundary
    axB.plot([0, 1], [0, 1], ls="--", lw=1, color="#999999", zorder=1)
    near_lab = (rf"near boundary ($\leq{NEAR_M:g}$\,m)" if use_tex else f"near boundary (<={NEAR_M:g} m)")
    far_lab = (rf"interior ($>{FAR_M:g}$\,m)" if use_tex else f"interior (>{FAR_M:g} m)")
    for accum, name, c in [(b_near, near_lab, "#b2182b"), (b_far, far_lab, "#2166ac")]:
        conf, acc, nb = accum.curve()
        m = nb > 0
        axB.plot(conf[m], acc[m], "-o", ms=4, lw=1.5, color=c,
                 label=f"{name}, ECE={accum.ece():.3f}", zorder=3)
    axB.set_xlim(0.45, 1.0); axB.set_ylim(0.45, 1.0)
    axB.set_xlabel("predicted confidence"); axB.set_ylabel("empirical accuracy")
    axB.set_title("(b)")
    axB.legend(loc="upper left", frameon=False)
    axB.grid(True, ls=":", lw=0.5, color="#cccccc"); axB.set_axisbelow(True)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / "reliability_ece.pdf", out_dir / "reliability_ece.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[reliability_ece] {n} tiles. baseline ECE={base.ece():.4f} full ECE={full.ece():.4f}")
    print(f"  full-model near-boundary ECE={b_near.ece():.4f} (mean conf {b_near.sconf.sum()/max(b_near.n.sum(),1):.3f}), "
          f"interior ECE={b_far.ece():.4f} (mean conf {b_far.sconf.sum()/max(b_far.n.sum(),1):.3f})")
    print(f"  -> {pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--no-tex", action="store_true")
    args = ap.parse_args()
    root = find_repo_root()
    out_dir = (root / args.out_dir).resolve()
    use_tex = not args.no_tex
    try:
        render(root, out_dir, args.seeds, use_tex)
    except Exception as e:
        if not use_tex:
            raise
        print(f"[reliability_ece] usetex failed ({e}); retrying mathtext")
        render(root, out_dir, args.seeds, use_tex=False)


if __name__ == "__main__":
    main()
