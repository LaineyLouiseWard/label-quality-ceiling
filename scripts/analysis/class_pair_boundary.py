#!/usr/bin/env python
"""
Class-pair boundary uncertainty (label-ceiling, non-circular).

Over the 10-seed stage3_clsbal validation ensemble, look at every GROUND-TRUTH boundary
pixel -- a pixel that has at least one 4-neighbour of a DIFFERENT gt class. For each such
pixel, the unordered pair of GT classes meeting there is recorded, and we accumulate:

  (a) boundary-pixel COUNT per pair          -> prevalence (how common that boundary is)
  (b) mean ensemble TOTAL entropy per pair    -> how uncertain the ensemble is there
  (c) mean ensemble MUTUAL INFORMATION per pair (epistemic / BALD)

Uncertainty (total entropy, MI) is the deep-ensemble decomposition used everywhere else in
the label-ceiling analysis (see seed_disagreement.py): for the N-seed softmax stack,

    total = H[mean_p],  expected = mean_i H[p_i],  MI = total - expected,

all normalised by log(C) so they live in [0, 1].

GROUPING IS STRICTLY BY GROUND TRUTH, never by predictions -> the analysis is non-circular.
A boundary pixel can touch more than one differing class across its 4-neighbourhood; the
pixel is counted ONCE for each distinct unordered GT pair it participates in (so a triple
junction contributes to all three pairs). Counts and uncertainty sums are accumulated per
pair, then reduced to symmetric 5x5 FOREGROUND (classes 1..5) matrices.

Outputs:
  analysis/label_ceiling/class_pair_boundary.json   (full numbers + top lists)
  analysis/label_ceiling/Npair_class_pair_matrix.{pdf,png}   (draft heatmaps)

Usage (defaults match the on-disk layout):
    PYTHONPATH=. python scripts/analysis/class_pair_boundary.py \
        --softmax-root sonic/results \
        --mask-dir data/biodiversity_split/val/masks \
        --cell stage3_clsbal \
        --seeds 42 43 44 45 46 47 48 49 50 51 \
        --out-dir analysis/label_ceiling
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoseg.taxonomy import STUDENT_CLASSES  # noqa: E402

# Reuse the canonical uncertainty + IO machinery so the decomposition is identical.
from scripts.analysis.seed_disagreement import (  # noqa: E402
    C,
    tile_uncertainty,
    load_mask,
    list_tiles,
    load_seed_stack,
)

FOREGROUND = list(range(1, C))  # 1..5


# ---------------------------------------------------------------------------
# Per-tile: which unordered GT pairs meet at each boundary pixel
# ---------------------------------------------------------------------------
def pair_membership(mask: np.ndarray):
    """Yield (pair, sel) for each unordered GT class pair present at a 4-neighbour boundary.

    `pair` is (a, b) with a < b; `sel` is a boolean (H, W) mask of pixels that participate in
    that boundary -- i.e. pixels of class a OR b that have a 4-neighbour of the OTHER class.
    Both sides of a boundary are counted (the pixel-pair on each side of the edge).

    Uses ground truth only.
    """
    m = mask
    H, W = m.shape
    # For each of the 4 directed neighbour offsets, mark pixels whose neighbour differs.
    # We build, per unordered pair, the set of pixels (on EITHER side) adjacent across that pair.
    # Collect directed adjacencies: (this_class, neighbour_class) at each pixel.
    out = {}

    # Vertical neighbours (up/down)
    diff_v = m[:-1, :] != m[1:, :]            # edge between row i and i+1
    # Horizontal neighbours (left/right)
    diff_h = m[:, :-1] != m[:, 1:]            # edge between col j and j+1

    # Build a dict: pair -> boolean (H,W) of participating pixels.
    def _add(pair, sel):
        if pair not in out:
            out[pair] = np.zeros((H, W), dtype=bool)
        out[pair] |= sel

    # Vertical edges: top pixel (rows 0..H-2) and bottom pixel (rows 1..H-1)
    if diff_v.any():
        lo = np.minimum(m[:-1, :], m[1:, :])[diff_v]
        hi = np.maximum(m[:-1, :], m[1:, :])[diff_v]
        for a, b in set(zip(lo.tolist(), hi.tolist())):
            em = diff_v & (np.minimum(m[:-1, :], m[1:, :]) == a) & (np.maximum(m[:-1, :], m[1:, :]) == b)
            sel = np.zeros((H, W), dtype=bool)
            sel[:-1, :] |= em      # top pixel
            sel[1:, :] |= em       # bottom pixel
            _add((int(a), int(b)), sel)

    # Horizontal edges: left pixel (cols 0..W-2) and right pixel (cols 1..W-1)
    if diff_h.any():
        for a, b in set(
            zip(
                np.minimum(m[:, :-1], m[:, 1:])[diff_h].tolist(),
                np.maximum(m[:, :-1], m[:, 1:])[diff_h].tolist(),
            )
        ):
            em = diff_h & (np.minimum(m[:, :-1], m[:, 1:]) == a) & (np.maximum(m[:, :-1], m[:, 1:]) == b)
            sel = np.zeros((H, W), dtype=bool)
            sel[:, :-1] |= em      # left pixel
            sel[:, 1:] |= em       # right pixel
            _add((int(a), int(b)), sel)

    return out


# ---------------------------------------------------------------------------
# Accumulator: symmetric pair sums over all tiles
# ---------------------------------------------------------------------------
class PairAccumulator:
    def __init__(self):
        # full CxC so background pairs are tracked too; foreground sub-matrix reported.
        self.count = np.zeros((C, C), dtype=np.int64)
        self.sum_total = np.zeros((C, C))
        self.sum_mi = np.zeros((C, C))

    def add(self, mask, total, mi):
        for (a, b), sel in pair_membership(mask).items():
            n = int(sel.sum())
            if n == 0:
                continue
            st = float(total[sel].sum())
            sm = float(mi[sel].sum())
            self.count[a, b] += n
            self.count[b, a] += n
            self.sum_total[a, b] += st
            self.sum_total[b, a] += st
            self.sum_mi[a, b] += sm
            self.sum_mi[b, a] += sm

    def matrices(self):
        n = np.where(self.count > 0, self.count, 1)
        mean_total = self.sum_total / n
        mean_mi = self.sum_mi / n
        mean_total[self.count == 0] = np.nan
        mean_mi[self.count == 0] = np.nan
        return self.count.copy(), mean_total, mean_mi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(softmax_root, mask_dir, cell, seeds, out_dir):
    img_ids = list_tiles(softmax_root, seeds, cell)
    acc = PairAccumulator()
    for iid in img_ids:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)   # (N, C, H, W)
        total, _expected, mi = tile_uncertainty(stack)
        mask = load_mask(mask_dir, iid)
        if mask.shape != total.shape:
            raise ValueError(f"shape mismatch {iid}: mask {mask.shape} vs softmax {total.shape}")
        acc.add(mask, total, mi)

    count, mean_total, mean_mi = acc.matrices()

    # ---- foreground sub-matrices + ranked pair lists ----
    fg = FOREGROUND
    names = STUDENT_CLASSES

    pairs = []  # (a, b) foreground pairs a<b
    for a, b in combinations(fg, 2):
        if count[a, b] == 0:
            continue
        pairs.append((a, b))

    def _rank(metric, reverse=True):
        rows = []
        for a, b in pairs:
            rows.append(
                {
                    "pair": f"{names[a]}<->{names[b]}",
                    "i": a,
                    "j": b,
                    "count": int(count[a, b]),
                    "mean_total_entropy": float(mean_total[a, b]),
                    "mean_mi": float(mean_mi[a, b]),
                }
            )
        return sorted(rows, key=lambda r: r[metric], reverse=reverse)

    by_prevalence = _rank("count")
    by_uncertainty = _rank("mean_total_entropy")
    by_mi = _rank("mean_mi")

    def _fg_mat(M):
        return [[(None if (isinstance(M[a, b], float) and np.isnan(M[a, b])) else float(M[a, b]))
                 for b in fg] for a in fg]

    result = {
        "cell": cell,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "n_tiles": len(img_ids),
        "grouping": "strictly by GROUND-TRUTH class pair at 4-neighbour boundaries (non-circular)",
        "normalisation": "entropy / log(6); values in [0,1]",
        "pixel_counting": "boundary pixel counted once per distinct unordered GT pair it touches; "
                          "both sides of each edge counted",
        "foreground_classes": [names[k] for k in fg],
        "foreground_indices": list(fg),
        "matrices_foreground": {
            "prevalence_count": _fg_mat(count.astype(float)),
            "mean_total_entropy": _fg_mat(mean_total),
            "mean_mi": _fg_mat(mean_mi),
        },
        "ranked_by_prevalence": by_prevalence,
        "ranked_by_mean_total_entropy": by_uncertainty,
        "ranked_by_mean_mi": by_mi,
        "top_prevalent_pairs": by_prevalence[:5],
        "top_uncertain_pairs": by_uncertainty[:5],
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_pair_boundary.json").write_text(json.dumps(result, indent=2))
    return result, (count, mean_total, mean_mi)


# ---------------------------------------------------------------------------
# Draft heatmap figure
# ---------------------------------------------------------------------------
def render_figure(count, mean_total, mean_mi, out_dir, cell, use_tex=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 11.5,
        "font.size": 11.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 12.5,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)

    fg = FOREGROUND
    labels = [STUDENT_CLASSES[k].replace("Seminatural", "Semi-natural") for k in fg]
    cnt = count[np.ix_(fg, fg)].astype(float)
    H = mean_total[np.ix_(fg, fg)]
    I = mean_mi[np.ix_(fg, fg)]

    # Mask the diagonal (no within-class boundary) for the uncertainty panels.
    diag = np.eye(len(fg), dtype=bool)
    Hm = np.ma.masked_where(diag | np.isnan(H), H)
    Im = np.ma.masked_where(diag | np.isnan(I), I)
    cnt_disp = np.ma.masked_where(diag | (cnt == 0), cnt)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)

    def _annotate(ax, M, fmt, txtcond):
        for i in range(len(fg)):
            for j in range(len(fg)):
                if i == j or (np.ma.is_masked(M[i, j])):
                    continue
                val = M[i, j]
                ax.text(j, i, fmt(val), ha="center", va="center",
                        fontsize=9.5, color=txtcond(val))

    # Panel 1: prevalence (log scale for dynamic range)
    cmap_c = plt.cm.viridis.copy()
    cmap_c.set_bad("0.9")
    pos = cnt_disp.compressed()
    vmax = pos.max() if pos.size else 1
    im0 = axes[0].imshow(np.ma.log10(cnt_disp + 1), cmap=cmap_c, aspect="equal")
    axes[0].set_title("Boundary prevalence\n(pixel count, log scale)")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(r"$\log_{10}(\mathrm{count}+1)$")
    _annotate(axes[0], cnt_disp,
              lambda v: f"{int(v/1000)}k" if v >= 1000 else f"{int(v)}",
              lambda v: "white" if np.log10(v + 1) < 0.6 * np.log10(vmax + 1) else "black")

    # Panel 2: mean total entropy
    cmap_u = plt.cm.magma.copy()
    cmap_u.set_bad("0.9")
    im1 = axes[1].imshow(Hm, cmap=cmap_u, aspect="equal")
    axes[1].set_title("Mean ensemble total entropy\nat shared boundary")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label(r"$H[\bar p]\ /\ \log 6$")
    hv = Hm.compressed()
    hmid = (hv.min() + hv.max()) / 2 if hv.size else 0.5
    _annotate(axes[1], Hm, lambda v: f"{v:.2f}",
              lambda v: "white" if v < hmid else "black")

    # Panel 3: mean MI
    cmap_m = plt.cm.cividis.copy()
    cmap_m.set_bad("0.9")
    im2 = axes[2].imshow(Im, cmap=cmap_m, aspect="equal")
    axes[2].set_title("Mean ensemble mutual information\n(epistemic) at shared boundary")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label(r"$I\ /\ \log 6$")
    iv = Im.compressed()
    imid = (iv.min() + iv.max()) / 2 if iv.size else 0.5
    _annotate(axes[2], Im, lambda v: f"{v:.2f}",
              lambda v: "white" if v < imid else "black")

    for ax in axes:
        ax.set_xticks(range(len(fg)))
        ax.set_yticks(range(len(fg)))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(-0.5, len(fg), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(fg), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)

    # No baked-in title: this is a paper figure; its caption lives in the LaTeX \caption.

    out_dir = Path(out_dir)
    pdf = out_dir / "Npair_class_pair_matrix.pdf"
    png = out_dir / "Npair_class_pair_matrix.png"
    # dpi=300 so the flat-colour heatmap cells clear MDPI's 600 dpi floor as printed (the cells
    # are uniform colour so this is cosmetic, but it avoids a blanket pdfimages resolution flag).
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return pdf, png


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default="data/biodiversity_split/val/masks")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    ap.add_argument("--no-tex", action="store_true", help="disable LaTeX fonts (fallback)")
    args = ap.parse_args()

    print(f"[class_pair_boundary] cell={args.cell} seeds={args.seeds}")
    result, (count, mean_total, mean_mi) = run(
        args.softmax_root, args.mask_dir, args.cell, args.seeds, args.out_dir
    )

    use_tex = not args.no_tex
    try:
        pdf, png = render_figure(count, mean_total, mean_mi, args.out_dir, args.cell, use_tex)
    except Exception as e:  # LaTeX may be missing on some machines
        print(f"[class_pair_boundary] LaTeX render failed ({e}); retrying with mathtext")
        pdf, png = render_figure(count, mean_total, mean_mi, args.out_dir, args.cell, use_tex=False)

    print("\nTop PREVALENT GT class-pair boundaries (by pixel count):")
    for r in result["top_prevalent_pairs"]:
        print(f"  {r['pair']:28s} n={r['count']:>9,d}  H={r['mean_total_entropy']:.3f}  MI={r['mean_mi']:.3f}")
    print("\nTop UNCERTAIN GT class-pair boundaries (by mean total entropy):")
    for r in result["top_uncertain_pairs"]:
        print(f"  {r['pair']:28s} H={r['mean_total_entropy']:.3f}  MI={r['mean_mi']:.3f}  n={r['count']:>9,d}")
    print(f"\n[class_pair_boundary] wrote {args.out_dir}/class_pair_boundary.json")
    print(f"[class_pair_boundary] wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
