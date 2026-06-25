#!/usr/bin/env python
"""
DRAFT qualitative boundary-uncertainty overlay (the "money shot" companion to N4).

Per tile, a horizontal panel row (the segmentation-uncertainty lit-standard layout,
cf. Kendall & Gal 2017 Fig. 1; Kahl et al. 2024 / ValUES):

    RGB | Ground truth | Ensemble prediction | Prediction error (grey/red) | Total entropy + GT boundary CONTOURS

Ground truth and prediction share STUDENT_PALETTE so the label and the ensemble argmax are
directly comparable; the prediction is the ensemble mean argmax (argmax of the mean softmax),
i.e. the exact prediction whose uncertainty H[mean_p] the final panel shows.

The final panel renders the 10-seed ensemble TOTAL entropy H[mean_p] as a perceptually-uniform
heatmap and draws the GROUND-TRUTH inter-class boundaries as thin contour lines on top, so the
"uncertainty hugs the class boundaries" claim is made WITHIN one panel (no mental registration).
This is the qualitative complement to N4 (the quantitative entropy-vs-distance-to-boundary curve)
and N7 (per-GT-class bars); it visualises the same non-circular, ground-truth-grouped signal.

Uncertainty quantity = ensemble total entropy, identical decomposition to seed_disagreement.py
(H[mean_p], normalised by log 6 -> [0,1]). Boundaries are the project's pure-numpy 4-neighbour
GT boundary (matches boundary_distance / class_pair_boundary), NOT skimage, so no extra deps.

Status: DRAFT for researcher approval -- titled "DRAFT", with a methods strip in the caption.

Usage:
    PYTHONPATH=. python scripts/analysis/draft_boundary_overlay.py
    # optional: --tiles biodiversity_1969 biodiversity_1403  --no-tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geoseg.taxonomy import STUDENT_CLASSES, STUDENT_PALETTE  # noqa: E402

# Reuse the canonical uncertainty + IO machinery so the decomposition is identical.
from scripts.analysis.seed_disagreement import (  # noqa: E402
    C,
    tile_uncertainty,
    load_mask,
    load_seed_stack,
)

PAL = np.array(STUDENT_PALETTE) / 255.0
CLASS_CMAP = ListedColormap(PAL)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, C + 0.5, 1), C)
FG = list(range(1, C))

# The two boundary-rich draft tiles (see the boundary-rich-tile analysis):
#   biodiversity_1969 -- BEST OVERALL: 9 distinct fg-fg pairs, both rare classes meet Forest,
#                        Grassland and each other; visually even (top pair only 22%).
#   biodiversity_1403 -- BEST Settlement<->Seminatural contact (27%): the two rare classes meet
#                        each other at length -> directly illustrates the top-uncertain pair.
DEFAULT_TILES = ["biodiversity_1969", "biodiversity_1403"]

# Contrast boxes (x0, y0, x1, y1) drawn on the GT and entropy panels of a tile to show, in
# space, the class-pair decoupling that Figure 13 reports dataset-wide: a confident COMMON
# boundary stays dark while an uncertain RARE-class contact lights up. Window locations and
# their local mean entropy were measured from the data (see the box-finding analysis).
CONTRAST_BOXES = {
    "biodiversity_1403": [
        (18, 56, 168, 206),    # settlement<->semi-natural contact  (local mean H ~0.47, bright)
        (349, 93, 499, 243),   # forest<->grassland boundary        (local mean H ~0.19, dark)
    ],
}


def setup_font(use_tex: bool):
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 16,
        "font.size": 16,
        "legend.fontsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titlesize": 18,
        "figure.dpi": 150,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
    plt.rcParams.update(rc)


def gt_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """4-neighbour GT class boundary (matches seed_disagreement.boundary_distance). Pure numpy."""
    m = mask
    bnd = np.zeros(m.shape, dtype=bool)
    bnd[:-1, :] |= m[:-1, :] != m[1:, :]
    bnd[1:, :] |= m[:-1, :] != m[1:, :]
    bnd[:, :-1] |= m[:, :-1] != m[:, 1:]
    bnd[:, 1:] |= m[:, :-1] != m[:, 1:]
    return bnd


def _rgb_for(img_dir, iid):
    """Canonical RGB read: per-band 2-98 percentile stretch, first 3 bands (matches the dataset)."""
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
        return out
    except Exception as e:  # pragma: no cover
        print(f"  [warn] RGB read failed for {iid}: {e}; using grey placeholder")
        return np.full((512, 512, 3), 200, np.uint8)


def render(tiles, softmax_root, mask_dir, img_dir, cell, seeds, out_dir, use_tex, error_tiles=()):
    setup_font(use_tex)
    error_tiles = set(error_tiles or ())

    # Compute ensemble entropy + GT boundary for each tile first, to set a shared vmax.
    panels = []
    for iid in tiles:
        stack = load_seed_stack(softmax_root, seeds, cell, iid)   # (N, C, H, W)
        total, _expected, mi = tile_uncertainty(stack)            # (H, W) in [0,1]
        mask = load_mask(mask_dir, iid)
        if mask.shape != total.shape:
            raise ValueError(f"shape mismatch {iid}: mask {mask.shape} vs softmax {total.shape}")
        rgb = _rgb_for(img_dir, iid)
        # Ensemble mean argmax: the prediction whose uncertainty H[mean_p] the last panel shows.
        pred = stack.mean(axis=0).argmax(axis=0).astype(np.uint8)
        err = (pred != mask) & (mask != 0)   # prediction error inside the annotated area
        panels.append({"iid": iid, "rgb": rgb, "mask": mask, "pred": pred, "err": err,
                       "total": total, "mi": mi, "bnd": gt_boundary_mask(mask)})

    vmax = max(p["total"].max() for p in panels)   # entropy colour scale (shared across tiles)
    vmax = float(np.ceil(vmax * 20) / 20)          # round up to a clean 0.05 step
    # The epistemic (MI) term is ~10x smaller than the total entropy; on the entropy scale it
    # renders near-black and hides its structure. Give panel (f) its OWN, smaller scale (its
    # own colourbar) so the structure is visible, while the two colourbar maxima make the
    # magnitude gap explicit.
    vmax_mi = max(p["mi"].max() for p in panels)
    vmax_mi = float(np.ceil(vmax_mi * 20) / 20)

    n = len(panels)
    ERR_RGB = np.array([0.84, 0.13, 0.16])      # crimson = prediction error
    OK_RGB = np.array([0.85, 0.85, 0.85])       # grey = correct (annotated area)

    # GridSpec layout: five square image columns, a spacer, then a colourbar column that
    # spans BOTH rows (gs[:, 6]) so the colourbar is exactly the height of the two-row image
    # stack. Explicit margins reserve a band at the bottom for the legend (no overlap) and a
    # gap on the right (colourbar not touching the last image). figsize chosen so cells are
    # square (images fill them; no internal whitespace).
    fig = plt.figure(figsize=(14.6, 2.62 * n))
    gs = fig.add_gridspec(n, 10, width_ratios=[1, 1, 1, 1, 1, 1, 0.12, 0.06, 0.22, 0.06],
                          wspace=0.05, hspace=0.05,
                          left=0.012, right=0.96, top=0.90, bottom=0.17)
    axes = np.empty((n, 6), dtype=object)
    for r in range(n):
        for c in range(6):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax_e = fig.add_subplot(gs[:, 7])   # entropy colourbar
    cax_f = fig.add_subplot(gs[:, 9])   # MI colourbar (own, smaller scale)

    im_ent = None
    im_mi = None
    for r, p in enumerate(panels):
        axes[r, 0].imshow(p["rgb"])
        axes[r, 1].imshow(p["mask"], cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        axes[r, 2].imshow(p["pred"], cmap=CLASS_CMAP, norm=CLASS_NORM, interpolation="nearest")
        # two-tone prediction-error map: white outside annotation, grey correct, crimson error
        canvas = np.ones((*p["err"].shape, 3))
        canvas[p["mask"] != 0] = OK_RGB
        canvas[p["err"]] = ERR_RGB
        axes[r, 3].imshow(canvas, interpolation="nearest")
        im_ent = axes[r, 4].imshow(p["total"], cmap="magma", vmin=0, vmax=vmax,
                                   interpolation="nearest")
        # GT class boundaries on the entropy heatmap, white core over a black casing so they
        # stay legible where they cross bright (near-white) magma.
        bf = p["bnd"].astype(float)
        axes[r, 4].contour(bf, levels=[0.5], colors="black", linewidths=1.4, alpha=0.85)
        axes[r, 4].contour(bf, levels=[0.5], colors="white", linewidths=0.6, alpha=0.95)

        # (f) epistemic term (mutual information) on its OWN, much smaller colour scale (its
        # max is several times below the entropy's, shown by the two colourbars). The
        # epistemic uncertainty is small in magnitude and, like the entropy, what little there
        # is hugs the class boundaries -> the residual uncertainty is aleatoric (label
        # ambiguity), not epistemic (model capacity). Same GT contours overlaid.
        im_mi = axes[r, 5].imshow(p["mi"], cmap="magma", vmin=0, vmax=vmax_mi,
                                  interpolation="nearest")
        axes[r, 5].contour(bf, levels=[0.5], colors="black", linewidths=1.4, alpha=0.85)
        axes[r, 5].contour(bf, levels=[0.5], colors="white", linewidths=0.6, alpha=0.95)

        # Contrast boxes on the GT (b), entropy (e) and MI (f) panels: confident common
        # boundary vs uncertain rare-class contact (spatial counterpart of the class-pair figure).
        for (bx0, by0, bx1, by1) in CONTRAST_BOXES.get(p["iid"], []):
            for cc in (1, 4, 5):
                axes[r, cc].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                                fill=False, edgecolor="cyan", linewidth=2.0,
                                                zorder=10))   # above the GT contour lines

        if r == 0:
            # Pad single-line headers to two lines so all align with the 2-line (e) title.
            pad2 = lambda s: s + "\n" + r"$\vphantom{Ag}$"
            axes[r, 0].set_title(pad2("(a) RGB"))
            axes[r, 1].set_title(pad2("(b) Ground truth"))
            axes[r, 2].set_title(pad2("(c) Prediction"))
            axes[r, 3].set_title(pad2("(d) Error"))
            axes[r, 4].set_title("\n".join([r"(e) Entropy $H[\bar{p}]$",
                                            r"+ GT boundaries"]))
            axes[r, 5].set_title("\n".join([r"(f) Mutual info $I$",
                                            r"+ GT boundaries"]))
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])

    cb_e = fig.colorbar(im_ent, cax=cax_e)
    cb_e.set_label(r"$H[\bar{p}]\ /\ \log 6$" if use_tex else "entropy / log 6", fontsize=17)
    cb_e.ax.tick_params(labelsize=14)
    cb_f = fig.colorbar(im_mi, cax=cax_f)
    cb_f.set_label(r"$I\ /\ \log 6$" if use_tex else "MI / log 6", fontsize=17)
    cb_f.ax.tick_params(labelsize=14)

    # class legend in the reserved bottom band (cannot overlap the images)
    handles = [Patch(facecolor=PAL[k], edgecolor="0.3",
                     label=STUDENT_CLASSES[k].replace("Seminatural", "Semi-natural"))
               for k in range(C)]
    handles.append(Patch(facecolor=ERR_RGB, edgecolor="0.3", label="Prediction error"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=16, columnspacing=1.3,
               handlelength=1.5, handletextpad=0.5)

    # No baked-in title: this is a paper figure; its caption lives in the LaTeX \caption.

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "draft_boundary_overlay.pdf"
    png = out_dir / "draft_boundary_overlay.png"
    # dpi=300 so the embedded raster panels clear MDPI's 600 dpi floor once the wide figure
    # is scaled down to \linewidth (~2.7x), giving ~800 ppi as printed.
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.15, dpi=300)
    fig.savefig(png, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return pdf, png


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default="data/biodiversity_split/val/masks")
    ap.add_argument("--img-dir", default="data/biodiversity_split/val/images")
    ap.add_argument("--cell", default="stage3_clsbal")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--error-tiles", nargs="*",
                    default=["biodiversity_1969", "biodiversity_1403"],
                    help="tiles whose panel (c) shows the prediction-ERROR map (grey/red) "
                         "instead of the class-coloured prediction; default = all shown tiles")
    ap.add_argument("--out-dir", default="analysis/label_ceiling")
    ap.add_argument("--no-tex", action="store_true", help="disable LaTeX fonts (fallback)")
    args = ap.parse_args()

    print(f"[draft_boundary_overlay] tiles={args.tiles} cell={args.cell} seeds={args.seeds}")
    use_tex = not args.no_tex
    try:
        pdf, png = render(args.tiles, args.softmax_root, args.mask_dir, args.img_dir,
                          args.cell, args.seeds, args.out_dir, use_tex,
                          error_tiles=args.error_tiles)
    except Exception as e:
        if not use_tex:
            raise
        print(f"[draft_boundary_overlay] usetex render failed ({e}); retrying with mathtext")
        pdf, png = render(args.tiles, args.softmax_root, args.mask_dir, args.img_dir,
                          args.cell, args.seeds, args.out_dir, use_tex=False,
                          error_tiles=args.error_tiles)
    print(f"[draft_boundary_overlay] wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
