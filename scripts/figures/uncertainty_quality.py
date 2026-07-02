#!/usr/bin/env python
"""
N4 -- the uncertainty is INFORMATIVE and DATA-INTRINSIC (the principled "labels, not capacity"
evidence). Two panels, baseline model, 219-tile Irish val set, deep ensemble of 10 seeds.

  (a) Sparsification / error-retention curve. Rank foreground pixels by predicted total entropy
      H[mean_p] and progressively discard the most-uncertain; plot the misclassification rate of
      what remains. A steep drop (close to the oracle that discards true errors first, far from
      the random-removal flat line) means the uncertainty RANKS error -- it is a usable signal,
      not noise. AUSE = area between the uncertainty curve and the oracle. (Ilg et al. 2018.)
  (b) Aleatoric vs epistemic vs distance-to-boundary. Total entropy decomposed into expected
      entropy (aleatoric / data uncertainty) + mutual information (epistemic / model uncertainty;
      Depeweg 2018, Houlsby 2011 BALD, deep-ensemble Lakshminarayanan 2017). MI stays a thin
      sliver at every distance -> the boundary uncertainty is DATA-INTRINSIC (the seeds agree it
      is uncertain and that does not shrink under model resampling), i.e. consistent with
      irreducible label ambiguity rather than fixable model uncertainty. INFERENCE, not a proof
      (the split is ensemble-relative). From the regenerated 219 stats.

Data:
  per-seed softmax dumps via list_val_tiles                 (panel a)
  analysis/label_ceiling/stats_stage1_baseline.json         (panel b, regenerated on 219)
Output: figures/uncertainty_quality.pdf (+ .png). NOT in build_all_figures.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.seed_disagreement import (  # noqa: E402
    list_val_tiles, load_mask, load_seed_stack, tile_uncertainty,
)

NB = 256
ENT_EDGES = np.linspace(0.0, 1.0, NB + 1)


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
          "legend.fontsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
          "axes.axisbelow": True, "figure.dpi": 150}
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def sparsification(root, softmax_root, mask_dir, cell, seeds):
    """Entropy-histogram sparsification: n and errors per total-entropy bin over fg pixels."""
    ids, _ = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    n_hist = np.zeros(NB, dtype=np.int64)
    e_hist = np.zeros(NB, dtype=np.int64)
    for iid in ids:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)
        total, _, _ = tile_uncertainty(stack)            # H[mean_p] in [0,1]
        mean_p = stack.mean(axis=0)
        pred = mean_p.argmax(axis=0)
        mask = load_mask(mask_dir, iid)
        fg = mask != 0
        err = fg & (pred != mask)
        b_all = np.clip(np.digitize(total[fg], ENT_EDGES[1:-1]), 0, NB - 1)
        b_err = np.clip(np.digitize(total[err], ENT_EDGES[1:-1]), 0, NB - 1)
        n_hist += np.bincount(b_all, minlength=NB)
        e_hist += np.bincount(b_err, minlength=NB)
    return n_hist, e_hist, len(ids)


def curves(n_hist, e_hist, n_pts=101):
    """Build error-vs-fraction-removed for uncertainty / oracle / random."""
    N, E = int(n_hist.sum()), int(e_hist.sum())
    # remove highest-entropy bins first -> cumulate from the top
    n_top = np.cumsum(n_hist[::-1])           # pixels removed after k top bins
    e_top = np.cumsum(e_hist[::-1])
    frac_removed_bins = n_top / N
    err_remaining_bins = np.where(N - n_top > 0, (E - e_top) / np.maximum(N - n_top, 1), np.nan)
    # resample onto a regular removal grid
    f = np.linspace(0, 0.95, n_pts)
    unc = np.interp(f, np.concatenate([[0], frac_removed_bins]),
                    np.concatenate([[E / N], err_remaining_bins]))
    # oracle: remove true errors first
    removed = f * N
    oracle = np.where(removed < E, (E - removed) / np.maximum(N - removed, 1), 0.0)
    random = np.full_like(f, E / N)
    d = unc - oracle
    ause = float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(f)))   # trapezoid, version-safe
    return f, unc, oracle, random, ause, E / N


def render(root, out_dir, seeds, use_tex):
    setup_font(use_tex)
    sr, md, cell = "sonic/results", "data/biodiversity_split/val/masks", "stage3_clsbal"
    n_hist, e_hist, n = sparsification(root, sr, md, cell, seeds)
    f, unc, oracle, random, ause, base_err = curves(n_hist, e_hist)
    print(f"[uncertainty_quality] cell={cell}  AUSE={ause:.4f}")

    st = json.load(open(root / "analysis/label_ceiling/stats_stage3_clsbal.json"))
    dc = st["distance_curve_foreground"]
    em = [e for e in dc["edges_m"] if np.isfinite(e)]
    cen = [(a + b) / 2 for a, b in zip(em[:-1], em[1:])] + [em[-1] * 1.4]
    tot = np.array(dc["mean_total"]); exp = np.array(dc["mean_expected"]); mi = np.array(dc["mean_mi"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 4.4))

    # (a) sparsification
    axA.plot(f, unc, "-", lw=1.8, color="#b2182b", label="rank by entropy", zorder=3)
    axA.plot(f, oracle, "--", lw=1.3, color="#2166ac", label="oracle (errors first)", zorder=2)
    axA.plot(f, random, ":", lw=1.3, color="#777777", label="random removal", zorder=1)
    axA.fill_between(f, oracle, unc, color="#b2182b", alpha=0.10, zorder=0)
    axA.set_xlabel("fraction of most-uncertain pixels removed")
    axA.set_ylabel("misclassification rate of remainder")
    axA.set_title("(a)")
    txt = (rf"AUSE $=$ {ause:.4f}" if use_tex else f"AUSE = {ause:.4f}")
    axA.text(0.96, 0.92, txt, transform=axA.transAxes, ha="right", va="top", fontsize=12, color="#333333")
    axA.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.85))
    axA.grid(True, ls=":", lw=0.5, color="#cccccc"); axA.set_axisbelow(True)
    axA.set_xlim(0, 0.95); axA.set_ylim(0, base_err * 1.05)

    # (b) aleatoric (expected) + epistemic (MI) vs distance, stacked to show total
    axB.fill_between(cen, 0, exp, color="#2166ac", alpha=0.5, label="aleatoric (expected $H$)" if use_tex else "aleatoric (expected H)")
    axB.fill_between(cen, exp, exp + mi, color="#b2182b", alpha=0.6, label="epistemic (MI)")
    axB.plot(cen, tot, "-o", ms=3, lw=1.2, color="black", label="total $H[\\bar{p}]$" if use_tex else "total H")
    axB.set_xscale("log")
    axB.set_xlabel("distance to GT boundary (m, log)")
    axB.set_ylabel(r"uncertainty / $\log 6$" if use_tex else "uncertainty / log6")
    axB.set_title("(b)")
    axB.legend(loc="upper right", frameon=False)
    axB.grid(True, ls=":", lw=0.5, color="#cccccc"); axB.set_axisbelow(True)
    epi_frac = float(mi.sum() / tot.sum())
    epi_band = float(mi[0] / tot[0])  # boundary-band share; matches caption/body (5.7%)
    axB.text(0.04, 0.05, (rf"epistemic share $\approx$ {epi_band * 100:.1f}\%" if use_tex
                          else f"epistemic share ~ {epi_band:.1%}"),
             transform=axB.transAxes, fontsize=11, color="#555555")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / "uncertainty_quality.pdf", out_dir / "uncertainty_quality.png"
    fig.savefig(pdf, bbox_inches="tight"); fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[uncertainty_quality] {n} tiles. AUSE={ause:.4f}, base err={base_err:.4f}, "
          f"epistemic share={epi_frac:.1%}")
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
        print(f"[uncertainty_quality] usetex failed ({e}); retrying mathtext")
        render(root, out_dir, args.seeds, use_tex=False)


if __name__ == "__main__":
    main()
