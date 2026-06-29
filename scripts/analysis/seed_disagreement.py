#!/usr/bin/env python
"""
Seed-ensemble disagreement -> label/boundary ambiguity (the label-ceiling analysis).

Implements docs/DISAGREEMENT_ANALYSIS_METHOD_2026-06-22.md. For a deep ensemble of N
same-architecture seeds, per validation pixel and over the per-seed softmax p^(i)_c:

    mean softmax        p_bar_c = (1/N) sum_i p^(i)_c
    total entropy       H[p_bar]                       (aleatoric-TYPE upper term; normalised by log C)
    expected entropy    E_i[H]  = (1/N) sum_i H[p^(i)]  (aleatoric-TYPE / data-uncertainty proxy)
    mutual information   I       = H[p_bar] - E_i[H]     (epistemic-type / BALD; >=0, LOWER BOUND at N)

All quantities are normalised by log(C) so they live in [0, 1].

Two non-circular stratifiers (both use GROUND TRUTH only, never predictions):
  1. distance to the nearest GT class boundary (scipy.ndimage distance transform on labels)
  2. the pixel's GROUND-TRUTH class

Reports foreground classes (1..5) only in class summaries, matching the rest of the paper.

This module does the COMPUTATION and writes per-pixel aggregates + a summary JSON. Figure
rendering lives in figure_label_ceiling.py so the numbers are reusable headless.

Usage (defaults match the on-disk layout):
    PYTHONPATH=. python scripts/analysis/seed_disagreement.py \
        --softmax-root sonic/results \
        --mask-dir data/biodiversity_split/val/masks \
        --cell stage3_clsbal --cell stage1_baseline \
        --seeds 42 43 44 45 46 47 48 49 50 51 \
        --out-dir analysis/label_ceiling
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# ---------------------------------------------------------------------------
# Taxonomy (single source of truth)
# ---------------------------------------------------------------------------
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoseg.taxonomy import STUDENT_CLASSES  # noqa: E402

C = len(STUDENT_CLASSES)          # 6
LOG_C = np.log(C)
BACKGROUND_INDEX = 0
FOREGROUND = list(range(1, C))    # 1..5
EPS = 1e-12

# Distance bins in PIXELS. GSD = 0.5 m/px, so metres = pixels * 0.5.
GSD_M = 0.5
# Edges chosen to resolve the boundary band finely then widen; last bin is open-ended.
DIST_BIN_EDGES_PX = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, np.inf])


# ---------------------------------------------------------------------------
# Core per-tile computation
# ---------------------------------------------------------------------------
def entropy(p: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shannon entropy along `axis`, normalised by log C -> [0, 1]."""
    return -(p * np.log(np.clip(p, EPS, 1.0))).sum(axis=axis) / LOG_C


def tile_uncertainty(seed_probs: np.ndarray):
    """seed_probs: (N, C, H, W) float. Returns (total_H, expected_H, MI) each (H, W), in [0,1].

    total      = H[mean_p]                 (total uncertainty)
    expected   = mean_i H[p_i]             (aleatoric-type / data-uncertainty proxy)
    MI         = total - expected          (epistemic-type; clip tiny negatives from fp error)
    """
    mean_p = seed_probs.mean(axis=0)                      # (C, H, W)
    total = entropy(mean_p, axis=0)                       # (H, W)
    per_seed_H = entropy(seed_probs, axis=1)              # (N, H, W)
    expected = per_seed_H.mean(axis=0)                    # (H, W)
    mi = total - expected
    mi = np.clip(mi, 0.0, None)                           # MI >= 0 by construction
    return total, expected, mi


def boundary_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from each pixel to the nearest GT class boundary.

    A boundary pixel is one whose label differs from any 4-neighbour. Distance is the EDT
    of the complement of the boundary set, so boundary pixels get distance 0. Uses GT only.
    """
    m = mask
    bnd = np.zeros(m.shape, dtype=bool)
    bnd[:-1, :] |= m[:-1, :] != m[1:, :]
    bnd[1:, :] |= m[:-1, :] != m[1:, :]
    bnd[:, :-1] |= m[:, :-1] != m[:, 1:]
    bnd[:, 1:] |= m[:, :-1] != m[:, 1:]
    if not bnd.any():
        # single-class tile: every pixel is infinitely far from a boundary
        return np.full(m.shape, np.inf, dtype=np.float32)
    # EDT of the "background" (non-boundary); boundary pixels -> 0.
    return ndimage.distance_transform_edt(~bnd).astype(np.float32)


# ---------------------------------------------------------------------------
# Aggregators (streaming over tiles to keep memory flat)
# ---------------------------------------------------------------------------
class Accumulator:
    """Streaming sums for the two stratifications, foreground-aware."""

    def __init__(self):
        nb = len(DIST_BIN_EDGES_PX) - 1
        # distance-binned: sums of total/expected/MI + counts, over FOREGROUND GT pixels only
        self.d_n = np.zeros(nb, dtype=np.int64)
        self.d_tot = np.zeros(nb)
        self.d_exp = np.zeros(nb)
        self.d_mi = np.zeros(nb)
        # also the all-pixel (incl. background) version for an appendix/robustness note
        self.d_n_all = np.zeros(nb, dtype=np.int64)
        self.d_tot_all = np.zeros(nb)
        self.d_mi_all = np.zeros(nb)
        # per-GT-class: sums over all C classes (report foreground only later)
        self.c_n = np.zeros(C, dtype=np.int64)
        self.c_tot = np.zeros(C)
        self.c_exp = np.zeros(C)
        self.c_mi = np.zeros(C)
        # per (seed-fold) per-class means, for error bars: handled separately

    def add(self, total, expected, mi, dist, mask):
        fg = mask != BACKGROUND_INDEX
        # --- distance stratification ---
        bidx = np.digitize(dist, DIST_BIN_EDGES_PX[1:-1])  # 0..nb-1
        # all-pixel
        for b in range(len(self.d_n_all)):
            sel = bidx == b
            if not sel.any():
                continue
            self.d_n_all[b] += int(sel.sum())
            self.d_tot_all[b] += float(total[sel].sum())
            self.d_mi_all[b] += float(mi[sel].sum())
            # foreground-only
            selfg = sel & fg
            if selfg.any():
                self.d_n[b] += int(selfg.sum())
                self.d_tot[b] += float(total[selfg].sum())
                self.d_exp[b] += float(expected[selfg].sum())
                self.d_mi[b] += float(mi[selfg].sum())
        # --- per-GT-class stratification (all classes; FG reported later) ---
        for k in range(C):
            sel = mask == k
            if not sel.any():
                continue
            self.c_n[k] += int(sel.sum())
            self.c_tot[k] += float(total[sel].sum())
            self.c_exp[k] += float(expected[sel].sum())
            self.c_mi[k] += float(mi[sel].sum())

    # ---- reductions ----
    def dist_curve(self):
        n = np.where(self.d_n > 0, self.d_n, 1)
        return {
            "edges_px": DIST_BIN_EDGES_PX.tolist(),
            "edges_m": (DIST_BIN_EDGES_PX * GSD_M).tolist(),
            "n": self.d_n.tolist(),
            "mean_total": (self.d_tot / n).tolist(),
            "mean_expected": (self.d_exp / n).tolist(),
            "mean_mi": (self.d_mi / n).tolist(),
        }

    def dist_curve_all(self):
        n = np.where(self.d_n_all > 0, self.d_n_all, 1)
        return {
            "edges_px": DIST_BIN_EDGES_PX.tolist(),
            "n": self.d_n_all.tolist(),
            "mean_total": (self.d_tot_all / n).tolist(),
            "mean_mi": (self.d_mi_all / n).tolist(),
        }

    def class_table(self):
        n = np.where(self.c_n > 0, self.c_n, 1)
        mt = self.c_tot / n
        me = self.c_exp / n
        mm = self.c_mi / n
        out = {}
        for k in range(C):
            out[STUDENT_CLASSES[k]] = {
                "index": k,
                "n_pixels": int(self.c_n[k]),
                "mean_total_entropy": float(mt[k]),
                "mean_expected_entropy": float(me[k]),
                "mean_mi": float(mm[k]),
                # fractions of total: how much of the total uncertainty is each component
                "frac_aleatoric_type": float(me[k] / mt[k]) if mt[k] > 0 else float("nan"),
                "frac_epistemic": float(mm[k] / mt[k]) if mt[k] > 0 else float("nan"),
            }
        return out


def per_seed_class_means(seed_probs_dir, seeds, cell, mask_dir, img_ids):
    """Per-seed (leave-N: actually per-individual-seed) per-GT-class TOTAL entropy of that
    single seed's softmax, to give an across-seed spread (error bars) on the class bars.

    NB: for a single seed there is no ensemble, so 'total' here is just that seed's own
    softmax entropy. This is only used to size error bars on N7; the headline bar height is
    the ensemble value from Accumulator.class_table().
    Returns array (n_seeds, C) of mean entropy per class (NaN where class absent).
    """
    out = np.full((len(seeds), C), np.nan)
    masks = {iid: load_mask(mask_dir, iid) for iid in img_ids}
    for si, s in enumerate(seeds):
        sums = np.zeros(C)
        cnts = np.zeros(C, dtype=np.int64)
        d = Path(seed_probs_dir) / f"seed{s}" / "analysis" / "seed_softmax" / cell / f"seed{s}"
        for iid in img_ids:
            p = np.load(d / f"{iid}.npy").astype(np.float32)
            h = entropy(p, axis=0)
            m = masks[iid]
            for k in range(C):
                sel = m == k
                if sel.any():
                    sums[k] += float(h[sel].sum())
                    cnts[k] += int(sel.sum())
        out[si] = np.where(cnts > 0, sums / np.where(cnts > 0, cnts, 1), np.nan)
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_mask(mask_dir, img_id) -> np.ndarray:
    m = np.array(Image.open(Path(mask_dir) / f"{img_id}.png").convert("L"))
    return m


def seed_dir(softmax_root, seed, cell):
    return Path(softmax_root) / f"seed{seed}" / "analysis" / "seed_softmax" / cell / f"seed{seed}"


def list_tiles(softmax_root, seeds, cell):
    d0 = seed_dir(softmax_root, seeds[0], cell)
    return sorted(p.stem for p in d0.glob("*.npy"))


def list_val_tiles(softmax_root, seeds, cell, mask_dir):
    """Tiles present in BOTH the softmax dumps and the mask dir, plus the dropped set.

    The dumps span a superset of tiles (an earlier broad eval); the evaluation set is
    whatever has a ground-truth mask. The mask dir is the canonical Irish val split, so
    foreign tiles (e.g. col1_/den*_) with no Irish mask are dropped rather than scored on.
    Returns (kept_ids, dropped_ids)."""
    ids = list_tiles(softmax_root, seeds, cell)
    kept = [i for i in ids if (Path(mask_dir) / f"{i}.png").exists()]
    return kept, sorted(set(ids) - set(kept))


def load_seed_stack(softmax_root, seeds, cell, img_id) -> np.ndarray:
    """(N, C, H, W) float32 stack of the N seeds' softmax for one tile."""
    arrs = []
    for s in seeds:
        a = np.load(seed_dir(softmax_root, s, cell) / f"{img_id}.npy").astype(np.float32)
        arrs.append(a)
    return np.stack(arrs, axis=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_cell(softmax_root, mask_dir, cell, seeds, out_dir, save_maps_for=None):
    img_ids, dropped = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    if dropped:
        print(f"[{cell}] scoring {len(img_ids)} val tiles with masks; "
              f"dropped {len(dropped)} dump tiles without an Irish mask")
    acc = Accumulator()
    saved_maps = {}
    save_maps_for = set(save_maps_for or [])

    for iid in img_ids:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)   # (N,C,H,W)
        total, expected, mi = tile_uncertainty(stack)
        mask = load_mask(mask_dir, iid)
        if mask.shape != total.shape:
            raise ValueError(f"shape mismatch {iid}: mask {mask.shape} vs softmax {total.shape}")
        dist = boundary_distance(mask)
        acc.add(total, expected, mi, dist, mask)
        if iid in save_maps_for:
            saved_maps[iid] = {
                "total": total.astype(np.float32),
                "expected": expected.astype(np.float32),
                "mi": mi.astype(np.float32),
                "mask": mask.astype(np.uint8),
                "mean_pred": stack.mean(axis=0).argmax(axis=0).astype(np.uint8),
            }

    # per-seed class spread for error bars
    seed_class = per_seed_class_means(softmax_root, seeds, cell, mask_dir, img_ids)

    result = {
        "cell": cell,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "n_tiles": len(img_ids),
        "normalisation": "entropy / log(6); values in [0,1]",
        "distance_curve_foreground": acc.dist_curve(),
        "distance_curve_allpixels": acc.dist_curve_all(),
        "per_gt_class": acc.class_table(),
        "per_seed_class_total_entropy_mean": np.nanmean(seed_class, axis=0).tolist(),
        "per_seed_class_total_entropy_std": np.nanstd(seed_class, axis=0).tolist(),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"stats_{cell}.json").write_text(json.dumps(result, indent=2))

    if saved_maps:
        np.savez_compressed(out_dir / f"maps_{cell}.npz",
                            **{f"{iid}__{k}": v for iid, d in saved_maps.items()
                               for k, v in d.items()})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default="data/biodiversity_split/val/masks")
    ap.add_argument("--cell", action="append", dest="cells",
                    default=None, help="repeatable; default: stage3_clsbal + stage1_baseline")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(range(42, 52)))
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    ap.add_argument("--save-map-tiles", nargs="*", default=None,
                    help="img_ids to dump full entropy/MI maps for (for N5/N6). "
                         "Default: auto-pick rare-class tiles.")
    args = ap.parse_args()

    cells = args.cells or ["stage3_clsbal", "stage1_baseline"]

    # auto-pick representative rare-class tiles if none given (Irish val tiles only)
    if args.save_map_tiles is None:
        val_ids, _ = list_val_tiles(args.softmax_root, args.seeds, cells[0], args.mask_dir)
        save_tiles = pick_rare_class_tiles(args.mask_dir, val_ids)
    else:
        save_tiles = args.save_map_tiles

    for cell in cells:
        print(f"[seed_disagreement] cell={cell}  seeds={args.seeds}")
        res = run_cell(args.softmax_root, args.mask_dir, cell, args.seeds,
                       args.out_dir, save_maps_for=save_tiles)
        ct = res["per_gt_class"]
        print(f"  per-GT-class mean total entropy (foreground):")
        for name in STUDENT_CLASSES[1:]:
            d = ct[name]
            print(f"    {name:12s} H={d['mean_total_entropy']:.4f} "
                  f"MI={d['mean_mi']:.4f} (epi frac {d['frac_epistemic']:.2f})  n={d['n_pixels']}")
    print(f"[seed_disagreement] wrote stats + maps to {args.out_dir}")
    print(f"  map tiles: {save_tiles}")


def pick_rare_class_tiles(mask_dir, img_ids, k_settlement=2, k_semi=2):
    """Pick tiles richest in Settlement(4) and Seminatural(5) for the qualitative maps."""
    sett, semi = [], []
    for iid in img_ids:
        m = load_mask(mask_dir, iid)
        sett.append((iid, int((m == 4).sum())))
        semi.append((iid, int((m == 5).sum())))
    sett.sort(key=lambda x: -x[1])
    semi.sort(key=lambda x: -x[1])
    picks = []
    for iid, _ in sett[:k_settlement]:
        if iid not in picks:
            picks.append(iid)
    for iid, _ in semi[:k_semi]:
        if iid not in picks:
            picks.append(iid)
    return picks


if __name__ == "__main__":
    main()
