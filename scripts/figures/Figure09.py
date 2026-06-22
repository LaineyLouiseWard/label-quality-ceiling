"""Fig 9: Foreground-only confusion matrices for Stages 1 and 3
(baseline, class-balanced sampler), plus a diverging Stage 3 - Stage 1 delta panel.

Each stage panel drops Background and row-normalises over the 5 foreground
classes (Blues, shared 0-1 scale). Panel (c) shows the Stage 3 - Stage 1
difference on a diverging RdBu scale centred at 0, highlighting where
confusion was corrected. Near-zero off-diagonal annotations are blanked;
the diagonal is always annotated.

Writes:
  figures/Figure09.pdf

Run:
  python scripts/figures/Figure09.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "geoseg").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")

repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

CM_A = repo_root / "evaluation/evaluation_results/val/stage1_baseline/confusion_matrix.npy"
CM_B = repo_root / "evaluation/evaluation_results/val/stage3_clsbal/confusion_matrix.npy"

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "Figure09.pdf"

assert CM_A.exists(), f"Missing: {CM_A}"
assert CM_B.exists(), f"Missing: {CM_B}"

# ---- style: match your other figs ----
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# Foreground-only class names (Background dropped — matches foreground-only mIoU
# convention and the 5-class appendix confusion table).
CLASS_NAMES_FG = [
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Semi-nat.",
]

# knobs
CMAP = "Blues"
DELTA_CMAP = "RdBu"          # diverging colormap for the Stage 3 - Stage 1 delta panel
ANNOT_FONTSIZE = 12
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 16
V_MIN, V_MAX = 0.0, 1.0
DELTA_ABS = 1.0              # delta panel colour scale: symmetric [-1, +1] centred at 0
ANNOT_EPS = 0.005           # blank off-diagonal annotations below this magnitude

def foreground_row_normalize(cm: np.ndarray) -> np.ndarray:
    """Drop Background (index 0), then row-normalise over the 5 foreground classes."""
    cm = cm.astype(np.float64)[1:, 1:]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums

def _set_cm_ticks(ax):
    ax.set_xticks(range(len(CLASS_NAMES_FG)))
    ax.set_yticks(range(len(CLASS_NAMES_FG)))
    ax.set_xticklabels(CLASS_NAMES_FG, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_FG)

def plot_cm(ax, cm_pct):
    im = ax.imshow(cm_pct, interpolation="nearest", vmin=V_MIN, vmax=V_MAX, cmap=CMAP)

    ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True", fontsize=LABEL_FONTSIZE)
    _set_cm_ticks(ax)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            # Always annotate the diagonal; blank near-zero off-diagonal cells.
            if i != j and abs(val) < ANNOT_EPS:
                continue
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=ANNOT_FONTSIZE,
                color="white" if val > 0.6 else "black"
            )

    return im

def plot_delta(ax, delta):
    """Diverging (Stage 3 - Stage 1) panel, centred at 0."""
    im = ax.imshow(delta, interpolation="nearest", vmin=-DELTA_ABS, vmax=DELTA_ABS, cmap=DELTA_CMAP)

    ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True", fontsize=LABEL_FONTSIZE)
    _set_cm_ticks(ax)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta[i, j]
            if i != j and abs(val) < ANNOT_EPS:
                continue
            ax.text(
                j, i, f"{val:+.2f}",
                ha="center", va="center",
                fontsize=ANNOT_FONTSIZE,
                color="white" if abs(val) > 0.6 else "black"
            )

    return im

def main():
    cm_a = np.load(CM_A)
    cm_b = np.load(CM_B)

    # Foreground-only (5x5), row-normalised over foreground classes.
    cm_a_pct = foreground_row_normalize(cm_a)
    cm_b_pct = foreground_row_normalize(cm_b)

    # Diverging delta panel: Stage 3 - Stage 1 (where confusion was corrected).
    cm_delta = cm_b_pct - cm_a_pct

    # 5 columns: 2 Blues CMs + shared Blues colorbar, then the RdBu delta + its colorbar.
    fig = plt.figure(figsize=(22, 9), dpi=300)
    gs = fig.add_gridspec(
        1, 5, width_ratios=[1.1, 1.1, 0.05, 1.1, 0.05], wspace=0.55
    )

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = plot_cm(ax1, cm_a_pct)
    ax1.set_title("(a) Stage 1: Baseline", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = plot_cm(ax2, cm_b_pct)
    ax2.set_title("(b) Stage 3: +Class-balanced sampler (clsbal)", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    # shared Blues colorbar for the two stage panels
    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # diverging delta panel
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = plot_delta(ax4, cm_delta)
    ax4.set_title(r"(c) Stage 3 $-$ Stage 1", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    # RdBu colorbar for the delta panel
    cax2 = fig.add_subplot(gs[0, 4])
    cbar2 = fig.colorbar(im4, cax=cax2)
    cbar2.ax.tick_params(labelsize=12)

    # --- shrink colorbar heights (same style as the original fig) ---
    for cx in (cax, cax2):
        pos = cx.get_position()
        cx.set_position([
            pos.x0 - 0.02,                  # nudge left a bit
            pos.y0 + pos.height * 0.25,     # move up
            pos.width,
            pos.height * 0.5                # shorten
        ])

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
