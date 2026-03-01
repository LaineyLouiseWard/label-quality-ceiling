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
CM_B = repo_root / "evaluation/evaluation_results/val/stage4_sampling/confusion_matrix.npy"

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig_confusion_matrix_stage1_vs_stage4.pdf"

assert CM_A.exists(), f"Missing: {CM_A}"
assert CM_B.exists(), f"Missing: {CM_B}"

# ---- style: match your other figs ----
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

CLASS_NAMES_6 = [
    "Background",
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Seminat.",
]

# knobs
CMAP = "Blues"
ANNOT_FONTSIZE = 12
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 16

V_MIN, V_MAX = 0.0, 1.0

def row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums

def plot_cm(ax, cm_pct):
    im = ax.imshow(cm_pct, interpolation="nearest", vmin=V_MIN, vmax=V_MAX, cmap=CMAP)

    ax.set_xlabel("Predicted", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True", fontsize=LABEL_FONTSIZE)

    ax.set_xticks(range(len(CLASS_NAMES_6)))
    ax.set_yticks(range(len(CLASS_NAMES_6)))
    ax.set_xticklabels(CLASS_NAMES_6, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_6)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=ANNOT_FONTSIZE,
                color="white" if val > 0.6 else "black"
            )

    return im

def main():
    cm_a = np.load(CM_A)
    cm_b = np.load(CM_B)

    cm_a_pct = row_normalize(cm_a)
    cm_b_pct = row_normalize(cm_b)

    # 3 columns: left CM, right CM, colorbar (same height as plots)
    fig = plt.figure(figsize=(15, 9), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.04], wspace=0.9)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = plot_cm(ax1, cm_a_pct)
    ax1.set_title("(a) Stage 1: Baseline", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = plot_cm(ax2, cm_b_pct)
    ax2.set_title("(b) Stage 4: Hard × Minority", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=10)

    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # --- shrink colorbar height ---
    pos = cax.get_position()
    cax.set_position([
        pos.x0 - 0.13,
        pos.y0 + pos.height * 0.25,   # move up
        pos.width,
        pos.height * 0.5              # shorten
    ])

    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_PDF)

if __name__ == "__main__":
    main()
