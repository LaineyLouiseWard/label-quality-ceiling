# Auto-exported from fig7_minority_cropping_schematic.ipynb
# Random vs minority-aware cropping schematic (current repo paths; figure layout unchanged)

import sys
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# -----------------------------------------------------------------------------
# Repo discovery
# -----------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "geoseg").is_dir() and (p / "config").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")

repo_root = find_repo_root(Path.cwd())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# -----------------------------------------------------------------------------
# Paths (schematic tile from biodiversity_raw to keep the same example as original)
# -----------------------------------------------------------------------------
DATA_ROOT = repo_root / "data" / "biodiversity_raw"
MSK_DIR = DATA_ROOT / "masks"

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "fig7_minority_cropping_schematic.pdf"

print("Repo root:", repo_root)
print("Masks dir:", MSK_DIR)
print("Output:", OUT_PDF)
assert MSK_DIR.exists(), f"Missing: {MSK_DIR}"

# -----------------------------------------------------------------------------
# Matplotlib style
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# -----------------------------------------------------------------------------
# Palette + classes
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Settings (match current repo intent)
# -----------------------------------------------------------------------------
CROP_SIZE = 512
IGNORE_INDEX = 0
NUM_CLASSES = 6

# Random crop uses dominance rejection only (V1-like)
MAX_RATIO = 0.75
SEED_RANDOM = 3

# Minority-aware crop targets minority classes only (V2-like)
CLASS_INTEREST = [4, 5]         # Settlement, Semi-natural
#CLASS_RATIO = [0.01, 0.01]      # minimum fractions
CLASS_RATIO = [0.01, 0.01]
SEED_MINORITY = SEED_RANDOM

# Deterministic schematic: pick one allowed scale
SCALE = 1.5

# -----------------------------------------------------------------------------
# FORCE TILE (same tile id as original code)
# -----------------------------------------------------------------------------
mask_path = MSK_DIR / "biodiversity_0003.png"
assert mask_path.exists(), f"Missing tile: {mask_path}"
print("Selected tile:", mask_path.stem)

mask = np.array(Image.open(mask_path))
print("Mask shape:", mask.shape, "dtype:", mask.dtype)
print("Contains settlement (4):", (mask == 4).any(), "semi-natural (5):", (mask == 5).any())

# -----------------------------------------------------------------------------
# Deterministic scale-up (nearest neighbour)
# -----------------------------------------------------------------------------
H, W = mask.shape[:2]
new_h, new_w = int(H * SCALE), int(W * SCALE)

mask_s = resize(
    mask,
    (new_h, new_w),
    order=0,
    preserve_range=True,
    anti_aliasing=False
).astype(mask.dtype)

print("Scaled mask shape:", mask_s.shape)

# -----------------------------------------------------------------------------
# SmartCropV1-like (random crop with dominance rejection)
# -----------------------------------------------------------------------------
def smartcrop_v1_bbox(mask, crop_size=512, max_ratio=0.75, ignore_index=0, seed=0, max_tries=50):
    """
    Random crop with dominance rejection (V1-like).
    Returns (crop_mask, bbox) where bbox=(x1,y1,x2,y2) in scaled-mask coords.
    """
    rng = random.Random(seed)
    h, w = mask.shape[:2]
    tw = th = crop_size

    if w == tw and h == th:
        return mask.copy(), (0, 0, w, h)

    last_bbox = (0, 0, min(tw, w), min(th, h))
    for _ in range(max_tries):
        x1 = rng.randint(0, max(0, w - tw))
        y1 = rng.randint(0, max(0, h - th))
        x2, y2 = x1 + tw, y1 + th
        last_bbox = (x1, y1, x2, y2)

        m = mask[y1:y2, x1:x2]
        valid = m[m != ignore_index]
        if valid.size == 0:
            continue

        counts = np.bincount(valid.flatten(), minlength=NUM_CLASSES)
        total = max(1, counts.sum())
        dom = counts.max() / total
        if dom <= max_ratio:
            return m.copy(), last_bbox

    x1, y1, x2, y2 = last_bbox
    return mask[y1:y2, x1:x2].copy(), last_bbox

# -----------------------------------------------------------------------------
# SmartCropV2-like (minority-aware crop with min presence constraints)
# -----------------------------------------------------------------------------
def smartcrop_v2_bbox(
    mask,
    crop_size=512,
    num_classes=6,
    class_interest=None,
    class_ratio=None,
    max_ratio=0.75,
    ignore_index=0,
    seed=0,
    max_tries=80,
):
    """
    Minority-aware crop (V2-like): reject overly-dominant crops and enforce minimum
    presence of interest classes. Returns (crop_mask, bbox).
    """
    rng = random.Random(seed)
    h, w = mask.shape[:2]
    tw = th = crop_size

    if class_interest is None:
        class_interest = list(range(num_classes))
    if class_ratio is None:
        class_ratio = [0.0] * len(class_interest)

    if w == tw and h == th:
        return mask.copy(), (0, 0, w, h)

    last_bbox = (0, 0, min(tw, w), min(th, h))
    for _ in range(max_tries):
        x1 = rng.randint(0, max(0, w - tw))
        y1 = rng.randint(0, max(0, h - th))
        x2, y2 = x1 + tw, y1 + th
        last_bbox = (x1, y1, x2, y2)

        m = mask[y1:y2, x1:x2]
        valid = m[m != ignore_index]
        if valid.size == 0:
            continue

        counts = np.bincount(valid.flatten(), minlength=num_classes)
        total = max(1, counts.sum())

        dom = counts.max() / total
        if dom > max_ratio:
            continue

        ok = True
        for cls, thr in zip(class_interest, class_ratio):
            if counts[cls] / total < thr:
                ok = False
                break
        if not ok:
            continue

        return m.copy(), last_bbox

    x1, y1, x2, y2 = last_bbox
    return mask[y1:y2, x1:x2].copy(), last_bbox

# -----------------------------------------------------------------------------
# Compute crops
# -----------------------------------------------------------------------------
msk_r, bbox_r = smartcrop_v1_bbox(
    mask_s,
    crop_size=CROP_SIZE,
    max_ratio=MAX_RATIO,
    ignore_index=IGNORE_INDEX,
    seed=SEED_RANDOM,
)
print("Random bbox:", bbox_r)
print("Random crop contains settlement:", (msk_r == 4).any(), "semi-natural:", (msk_r == 5).any())

msk_m, bbox_m = smartcrop_v2_bbox(
    mask_s,
    crop_size=CROP_SIZE,
    num_classes=NUM_CLASSES,
    class_interest=CLASS_INTEREST,
    class_ratio=CLASS_RATIO,
    max_ratio=MAX_RATIO,
    ignore_index=IGNORE_INDEX,
    seed=SEED_MINORITY,
)
print("Minority bbox:", bbox_m)
print("Minority crop contains settlement:", (msk_m == 4).any(), "semi-natural:", (msk_m == 5).any())
print("BBoxes equal?:", bbox_r == bbox_m)

# -----------------------------------------------------------------------------
# Plot (boxes in BOTH panels; matches caption)
# -----------------------------------------------------------------------------
def add_panel_label_above_center(ax, label, fontsize=32):
    ax.text(
        0.5, 1.02, label,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold"
    )

colors = np.array([np.array(COLOR_MAP[i]) / 255.0 for i in range(6)])
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)

def plot_fig7_boxes_both(full_mask, bbox_random, bbox_minority, out_pdf: Path):
    def draw_bbox(ax, bbox):
        x1, y1, x2, y2 = bbox
        ax.imshow(full_mask, cmap=cmap, norm=norm, interpolation="nearest")
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="white",
                linewidth=4,
            )
        )
        ax.set_axis_off()

    fig = plt.figure(figsize=(8*3, 4.2*3), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.18], hspace=0.15, wspace=-0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_bbox(ax1, bbox_random)
    add_panel_label_above_center(ax1, "(a)")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_bbox(ax2, bbox_minority)
    add_panel_label_above_center(ax2, "(b)")

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    handles = [Patch(facecolor=colors[i], label=CLASS_NAMES[i]) for i in range(6)]
    legend_ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=32)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print("Saved:", out_pdf)

plot_fig7_boxes_both(mask_s, bbox_r, bbox_m, OUT_PDF)
