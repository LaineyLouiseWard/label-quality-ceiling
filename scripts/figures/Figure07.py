"""Fig 7: Example low- and high-weight Biodiversity tiles under the Stage 3
class-balanced (clsbal) frequency-only sampler (Kang 2020), annotated with the
resulting sampling weight.

The shipped Stage 3 sampler is FREQUENCY-ONLY: a tile's weight is the inverse
tile-PRESENCE frequency of the rarest minority class (Settlement=4, Semi-nat.=5)
it contains; tiles with no minority get the uniform baseline 1.0. There is no
hardness term and no continuous pixel-richness multiplier (those belonged to the
retired A0 hardness x richness sampler). The weight therefore takes only a few
discrete values, so we no longer decompose it into hard/minority components.

FLAG (post-run visual review): clsbal weights are discrete (baseline 1.0 vs the
inverse-frequency value of whichever minority class is present), so the
"low-weight vs high-weight tile" contrast here is really "no-minority baseline
tile" vs "rarest-minority tile" rather than a continuous easy/hard spread. Confirm
the two selected example tiles read sensibly once the final clsbal weights exist.

Writes:
  figures/Figure07.pdf

Run:
  python scripts/figures/Figure07.py
"""

import sys
from pathlib import Path

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "geoseg").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")

repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# -----------------------
# Paths (NEW REPO)
# -----------------------
SPLIT_ROOT = repo_root / "data" / "biodiversity_split" / "train"
MSK_DIR = SPLIT_ROOT / "masks"

# Stage 3 ships the class-balanced (clsbal) frequency-only sampler.
WEIGHTS_CANDIDATES = [
    repo_root / "artifacts" / "sampler_weights_clsbal.tsv",
]

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "Figure07.pdf"

print("Repo root:", repo_root)
print("Masks dir:", MSK_DIR)
print("Output:", OUT_PDF)

assert MSK_DIR.exists(), f"Missing masks dir: {MSK_DIR}"

WEIGHTS_TSV = next((p for p in WEIGHTS_CANDIDATES if p.exists()), None)
assert WEIGHTS_TSV is not None, (
    "Missing clsbal sampler weights TSV. Looked for:\n"
    + "\n".join([f"  - {p}" for p in WEIGHTS_CANDIDATES])
    + "\nGenerate with:\n"
    "  PYTHONPATH=. python scripts/data_prep/build_clsbal_sampler.py"
)

print("Weights TSV:", WEIGHTS_TSV)

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}",
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# %%
# --- palette + class names ---
COLOR_MAP = {
    0: [0, 0, 0],
    1: [250, 62, 119],
    2: [168, 232, 84],
    3: [242, 180, 92],
    4: [59, 141, 247],
    5: [255, 214, 33],
}

CLASS_NAMES = [
    "Background", "Forest land", "Grassland", "Cropland",
    "Settlement", "Semi-nat."
]

colors = np.array([np.array(COLOR_MAP[i]) / 255.0 for i in range(6)])
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

# %%
# clsbal is FREQUENCY-ONLY (Kang 2020): the per-tile weight is the inverse
# tile-presence frequency of the rarest minority class it contains (or 1.0 if
# none). It is read directly from the TSV; there is no hardness/richness
# decomposition to recompute here (that was the retired A0 sampler).

def load_sampler_weight_map(weights_tsv: Path) -> dict[str, float]:
    m = {}
    with open(weights_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, w = line.split("\t")
            m[k] = float(w)
    return m

w_map = load_sampler_weight_map(WEIGHTS_TSV)
w_vals = np.array(list(w_map.values()), dtype=np.float32)
print("Loaded sampler weights:", len(w_vals), "min/med/max:", float(w_vals.min()), float(np.median(w_vals)), float(w_vals.max()))

# Pick a representative low-weight and high-weight tile from the clsbal weights.
# clsbal weights are discrete (baseline 1.0 vs the inverse-frequency value of the
# rarest minority class present), so q=0.10 lands on a no-minority baseline tile
# and q=0.90 lands on a rarest-minority tile.
# FLAG (post-run visual review): confirm these two tiles read as a sensible
# low/high contrast once the final clsbal weights exist.
q_lo, q_hi = 0.10, 0.90
w_lo = np.quantile(w_vals, q_lo)
w_hi = np.quantile(w_vals, q_hi)

keys = list(w_map.keys())
weights_arr = np.array([w_map[k] for k in keys], dtype=np.float32)

idx_easy = int(np.argmin(np.abs(weights_arr - w_lo)))
idx_hard = int(np.argmin(np.abs(weights_arr - w_hi)))

easy_id = keys[idx_easy]
hard_id = keys[idx_hard]

print("Low-weight tile:", easy_id, "weight:", float(w_map[easy_id]))
print("High-weight tile:", hard_id, "weight:", float(w_map[hard_id]))

# %%
def mask_path_from_id(msk_dir: Path, img_id: str) -> Path:
    p = msk_dir / f"{img_id}.png"
    if not p.exists():
        hits = list(msk_dir.glob(f"{img_id}*.png"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Mask not found for {img_id}: {p}")
    return p

easy_mask_path = mask_path_from_id(MSK_DIR, easy_id)
hard_mask_path = mask_path_from_id(MSK_DIR, hard_id)

easy_mask = np.array(Image.open(easy_mask_path))
hard_mask = np.array(Image.open(hard_mask_path))

print("Low-weight mask:", easy_mask_path.name, easy_mask.shape, easy_mask.dtype)
print("High-weight mask:", hard_mask_path.name, hard_mask.shape, hard_mask.dtype)

# %%
def add_panel_label_above_center(ax, label, fontsize=32):
    ax.text(
        0.5, 1.02, label,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold"
    )

def add_weights_text(ax, text, fontsize=26):
    # Put inside the axis at the bottom so it won't get cropped by savefig.
    ax.text(
        0.5, -0.08, text,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize,
        fontweight="normal",
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2.0),
    )

# %%
def minority_classes_present(mask_np: np.ndarray) -> list[int]:
    # minority = Settlement(4) or Semi-natural(5); clsbal keys a tile's weight on
    # the rarest minority class present (any-pixel presence).
    return [c for c in (4, 5) if bool((mask_np == c).any())]

def minority_label(mask_np: np.ndarray) -> str:
    present = minority_classes_present(mask_np)
    if not present:
        return "no minority"
    # CLASS_NAMES index 4 = Settlement, 5 = Semi-nat.
    return ", ".join(CLASS_NAMES[c] for c in present)

# %%
# clsbal is frequency-only: the per-tile weight is read straight from the TSV.
easy_final = float(w_map[easy_id])
hard_final = float(w_map[hard_id])

easy_minority = minority_label(easy_mask)
hard_minority = minority_label(hard_mask)

print(f"Low-weight tile:  minority={easy_minority}, weight={easy_final:.2f}")
print(f"High-weight tile: minority={hard_minority}, weight={hard_final:.2f}")

# %%
def plot_figure07_clsbal_sampling(
    easy_mask, hard_mask,
    easy_minority, easy_final,
    hard_minority, hard_final,
    out_pdf: Path
):
    fig = plt.figure(figsize=(8*3, 4.2*3), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.18], hspace=0.15, wspace=-0.15)

    # (a) low-weight tile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(easy_mask, cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_axis_off()
    add_panel_label_above_center(ax1, "(a)")
    add_weights_text(ax1, f"{easy_minority}, weight={easy_final:.2f}", fontsize=28)

    # (b) high-weight tile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hard_mask, cmap=cmap, norm=norm, interpolation="nearest")
    ax2.set_axis_off()
    add_panel_label_above_center(ax2, "(b)")
    add_weights_text(ax2, f"{hard_minority}, weight={hard_final:.2f}", fontsize=28)

    # legend row
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    handles = [Patch(facecolor=colors[i], label=CLASS_NAMES[i]) for i in range(6)]
    legend_ax.legend(handles=handles, loc="center",bbox_to_anchor=(0.5, 0.40), ncol=3, frameon=False, fontsize=32)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print("Saved:", out_pdf)

plot_figure07_clsbal_sampling(
    easy_mask, hard_mask,
    easy_minority, easy_final,
    hard_minority, hard_final,
    OUT_PDF
)
