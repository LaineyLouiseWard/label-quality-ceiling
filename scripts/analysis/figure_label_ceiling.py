#!/usr/bin/env python
"""
Render the label-ceiling figures from seed_disagreement.py outputs.

  N4  entropy & MI vs distance-to-GT-boundary  (the LOAD-BEARING non-circular curve)
  N7  per-GT-class total-entropy + MI bars      (grouped strictly by ground truth)
  N5  total-entropy maps on rare-class tiles     (RGB | GT | total entropy | MI)
  N6  decomposition maps on rare-class tiles      (expected entropy | MI)   [optional pair]

Fonts: Computer Modern via text.usetex (matches the paper). If LaTeX is unavailable the
caller should fall back to mathtext and FLAG it for the 5b font pass.
Class panels use STUDENT_PALETTE.

Usage:
    PYTHONPATH=. python scripts/analysis/figure_label_ceiling.py \
        --stats-dir analysis/label_ceiling --out-dir analysis/label_ceiling \
        --img-dir data/biodiversity_split/val/images
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoseg.taxonomy import STUDENT_CLASSES, STUDENT_PALETTE  # noqa: E402

C = len(STUDENT_CLASSES)
GSD_M = 0.5
FG = list(range(1, C))
PAL = np.array(STUDENT_PALETTE) / 255.0
CLASS_CMAP = ListedColormap(PAL)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, C + 0.5, 1), C)

CELL_LABEL = {"stage3_clsbal": "Full (transfer + sampler)",
              "stage1_baseline": "Baseline"}
CELL_COLOR = {"stage3_clsbal": "#2166ac", "stage1_baseline": "#b2182b"}


def setup_font(use_tex: bool):
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def _bin_centres_m(edges_px):
    e = np.array(edges_px, dtype=float)
    # represent the open-ended last bin by its lower edge + a nominal half-width
    centres = np.empty(len(e) - 1)
    for i in range(len(e) - 1):
        lo, hi = e[i], e[i + 1]
        centres[i] = (lo + (lo + (e[i] - e[i - 1] if i > 0 else 2))) / 2 if np.isinf(hi) else (lo + hi) / 2
    return centres * GSD_M


# ---------------------------------------------------------------------------
# N4 — distance-to-GT-boundary (LOAD BEARING)
# ---------------------------------------------------------------------------
def fig_n4(stats, out, use_tex):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    ax_h, ax_mi = axes
    for cell, st in stats.items():
        c = st["distance_curve_foreground"]
        x = _bin_centres_m(c["edges_px"])
        H = np.array(c["mean_total"])
        MI = np.array(c["mean_mi"])
        col = CELL_COLOR[cell]
        lab = CELL_LABEL[cell]
        ax_h.plot(x, H, "-o", ms=3.2, lw=1.4, color=col, label=lab)
        ax_mi.plot(x, MI, "-o", ms=3.2, lw=1.4, color=col, label=lab)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"Distance to nearest GT class boundary (m)")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    ax_h.set_ylabel(r"Mean total entropy $H[\bar{p}]\,/\,\log 6$")
    ax_mi.set_ylabel(r"Mean mutual information $I\,/\,\log 6$")
    ax_h.set_title(r"(a) Total uncertainty")
    ax_mi.set_title(r"(b) Epistemic (MI, lower bound)")
    ax_h.legend(frameon=False, loc="upper right")
    fig.savefig(out / "N4_distance_to_boundary.pdf", bbox_inches="tight")
    fig.savefig(out / "N4_distance_to_boundary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# N7 — per-GT-class bars (grouped strictly by ground truth)
# ---------------------------------------------------------------------------
def fig_n7(stats, out, use_tex):
    names = list(STUDENT_CLASSES[1:])  # foreground only
    x = np.arange(len(names))
    cells = list(stats.keys())
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    ax_h, ax_mi = axes

    for j, cell in enumerate(cells):
        st = stats[cell]
        H = [st["per_gt_class"][n]["mean_total_entropy"] for n in names]
        MI = [st["per_gt_class"][n]["mean_mi"] for n in names]
        # across-seed std (per-seed single-model entropy) for an error indication on H
        sd_all = np.array(st["per_seed_class_total_entropy_std"])
        sdH = [sd_all[STUDENT_CLASSES.index(n)] for n in names]
        off = (j - (len(cells) - 1) / 2) * w
        ax_h.bar(x + off, H, w, yerr=sdH, capsize=2.5, color=CELL_COLOR[cell],
                 alpha=0.9, label=CELL_LABEL[cell], error_kw=dict(lw=0.8))
        ax_mi.bar(x + off, MI, w, color=CELL_COLOR[cell], alpha=0.9,
                  label=CELL_LABEL[cell])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("Seminatural", "Semi-natural") for n in names],
                           rotation=20, ha="right")
        ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.5)
    ax_h.set_ylabel(r"Mean total entropy $H[\bar{p}]\,/\,\log 6$")
    ax_mi.set_ylabel(r"Mean mutual information $I\,/\,\log 6$")
    ax_h.set_title(r"(a) Total uncertainty by GT class")
    ax_mi.set_title(r"(b) Epistemic (MI) by GT class")
    ax_h.legend(frameon=False, loc="upper left")
    # colour the x tick labels by class palette for a visual link
    for ax in axes:
        for tick, n in zip(ax.get_xticklabels(), names):
            tick.set_color(PAL[STUDENT_CLASSES.index(n)] * 0.75)
    fig.savefig(out / "N7_per_gt_class.pdf", bbox_inches="tight")
    fig.savefig(out / "N7_per_gt_class.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# N5 / N6 — maps on rare-class tiles
# ---------------------------------------------------------------------------
def _load_maps(npz_path):
    z = np.load(npz_path)
    out = {}
    for k in z.files:
        iid, field = k.rsplit("__", 1)
        out.setdefault(iid, {})[field] = z[k]
    return out


def _rgb_for(img_dir, iid):
    """Canonical RGB read matching the dataset: per-band 2-98 percentile stretch, first 3 bands.

    Inlined from biodiversity_dataset._read_tif_as_rgb_uint8 (importing that module pulls in
    albumentations, which need not be installed in an analysis env).
    """
    try:
        import rasterio
        with rasterio.open(Path(img_dir) / f"{iid}.tif") as src:
            data = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)  # (H,W,C)
        data = np.where(np.isnan(data), 0, data)
        out = np.zeros_like(data)
        for c in range(data.shape[2]):
            band = data[:, :, c]
            valid = band[(band != 0) & ~np.isnan(band)]
            if valid.size:
                p2, p98 = np.percentile(valid, (2, 98))
                if p98 > p2:
                    band = (np.clip(band, p2, p98) - p2) / (p98 - p2)
            out[:, :, c] = band
        out = (out * 255).clip(0, 255).astype(np.uint8)
        out = out[:, :, :3] if out.shape[2] >= 3 else np.repeat(out, 3, axis=2)
        if out.shape[:2] != (512, 512):
            from PIL import Image as _I
            out = np.array(_I.fromarray(out).resize((512, 512)))
        return out
    except Exception as e:  # pragma: no cover
        print(f"  [warn] RGB read failed for {iid}: {e}; using grey placeholder")
        return np.full((512, 512, 3), 200, np.uint8)


def fig_maps(cell, maps, img_dir, out, use_tex, tiles, vmax_total, vmax_mi):
    n = len(tiles)
    fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.45 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    for r, iid in enumerate(tiles):
        d = maps[iid]
        rgb = _rgb_for(img_dir, iid)
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(d["mask"], cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        im2 = axes[r, 2].imshow(d["total"], cmap="viridis", vmin=0, vmax=vmax_total)
        im3 = axes[r, 3].imshow(d["mi"], cmap="magma", vmin=0, vmax=vmax_mi)
        axes[r, 0].set_ylabel(iid.replace("_", r"\_") if use_tex else iid, fontsize=10)
        if r == 0:
            axes[r, 0].set_title("RGB")
            axes[r, 1].set_title("Ground truth")
            axes[r, 2].set_title(r"Total entropy $H[\bar{p}]$")
            axes[r, 3].set_title(r"Mutual information $I$")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im2, ax=axes[:, 2].tolist(), fraction=0.046, pad=0.02, aspect=40)
    fig.colorbar(im3, ax=axes[:, 3].tolist(), fraction=0.046, pad=0.02, aspect=40)
    # class legend below
    handles = [Patch(facecolor=PAL[k], edgecolor="0.3",
                     label=STUDENT_CLASSES[k].replace("Seminatural", "Semi-natural"))
               for k in range(C)]
    fig.legend(handles=handles, loc="lower center", ncol=C, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)
    tag = "N5_entropy_maps" if cell == "stage3_clsbal" else f"N5_entropy_maps_{cell}"
    fig.savefig(out / f"{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_decomp(cell, maps, out, use_tex, tiles, vmax_exp, vmax_mi):
    """N6: expected entropy (aleatoric-type) | MI (epistemic) side by side per tile."""
    n = len(tiles)
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.55 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    for r, iid in enumerate(tiles):
        d = maps[iid]
        im0 = axes[r, 0].imshow(d["expected"], cmap="viridis", vmin=0, vmax=vmax_exp)
        im1 = axes[r, 1].imshow(d["mi"], cmap="magma", vmin=0, vmax=vmax_mi)
        axes[r, 0].set_ylabel(iid.replace("_", r"\_") if use_tex else iid, fontsize=10)
        if r == 0:
            axes[r, 0].set_title(r"Aleatoric-type $\mathbb{E}_i[H]$")
            axes[r, 1].set_title(r"Epistemic (MI) $I$")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im0, ax=axes[:, 0].tolist(), fraction=0.046, pad=0.02, aspect=40)
    fig.colorbar(im1, ax=axes[:, 1].tolist(), fraction=0.046, pad=0.02, aspect=40)
    tag = "N6_decomposition_maps" if cell == "stage3_clsbal" else f"N6_decomposition_maps_{cell}"
    fig.savefig(out / f"{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", default="analysis/label_ceiling")
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    ap.add_argument("--img-dir", default="data/biodiversity_split/val/images")
    ap.add_argument("--cells", nargs="+", default=["stage3_clsbal", "stage1_baseline"])
    ap.add_argument("--no-tex", action="store_true", help="force mathtext fallback")
    args = ap.parse_args()

    use_tex = not args.no_tex
    # probe usetex once; if it fails, fall back and flag
    setup_font(use_tex)
    if use_tex:
        try:
            fig = plt.figure(); fig.text(0.5, 0.5, r"$H[\bar{p}]\;\mathbb{E}_i[H]$"); fig.canvas.draw()
            plt.close(fig)
        except Exception as e:
            print(f"[figure_label_ceiling] usetex probe FAILED -> mathtext fallback. FLAG for 5b. ({e})")
            use_tex = False
            setup_font(False)

    stats = {c: json.load(open(Path(args.stats_dir) / f"stats_{c}.json"))
             for c in args.cells}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fig_n4(stats, out, use_tex)
    fig_n7(stats, out, use_tex)

    # shared colour scales across cells for fair map comparison
    for cell in args.cells:
        npz = Path(args.stats_dir) / f"maps_{cell}.npz"
        if not npz.exists():
            print(f"  [skip maps] {npz} not found")
            continue
        maps = _load_maps(npz)
        tiles = list(maps.keys())
        # robust vmax from the saved tiles (95th pct) so the colourbar isn't blown out
        all_tot = np.concatenate([maps[t]["total"].ravel() for t in tiles])
        all_exp = np.concatenate([maps[t]["expected"].ravel() for t in tiles])
        all_mi = np.concatenate([maps[t]["mi"].ravel() for t in tiles])
        vmax_total = float(np.percentile(all_tot, 99))
        vmax_exp = float(np.percentile(all_exp, 99))
        vmax_mi = float(np.percentile(all_mi, 99))
        fig_maps(cell, maps, args.img_dir, out, use_tex, tiles, vmax_total, vmax_mi)
        fig_decomp(cell, maps, out, use_tex, tiles, vmax_exp, vmax_mi)

    print(f"[figure_label_ceiling] wrote N4/N5/N6/N7 to {out} (usetex={use_tex})")
    if not use_tex:
        print("  ** Computer Modern via mathtext fallback — FLAG for the 5b font pass. **")


if __name__ == "__main__":
    main()
