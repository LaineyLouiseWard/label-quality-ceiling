"""Fig 6: Example low- and high-weight Biodiversity tiles under hard x minority-aware
sampling (Stage 4), annotated with tile difficulty, minority richness, and resulting weight.

Writes:
  figures/Figure06.pdf

Run:
  python scripts/figures/Figure06.py
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
SPLIT_ROOT = repo_root / "data" / "biodiversity_split" / "train_rep"
MSK_DIR = SPLIT_ROOT / "masks"

WEIGHTS_CANDIDATES = [
    repo_root / "artifacts" / "stage4_sampling_weights.tsv",
]

OUT_DIR = repo_root / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "Figure06.pdf"

# Stage 3b checkpoint used to compute hardness (pixel error mass)
STAGE3B_CKPT = repo_root / "model_weights" / "biodiversity" / "stage3b_finetune" / "stage3b_finetune.ckpt"

print("Repo root:", repo_root)
print("Masks dir:", MSK_DIR)
print("Stage3b ckpt:", STAGE3B_CKPT)
print("Output:", OUT_PDF)

assert MSK_DIR.exists(), f"Missing masks dir: {MSK_DIR}"
assert STAGE3B_CKPT.exists(), (
    f"Missing Stage 3b checkpoint: {STAGE3B_CKPT}\n"
    "Hardness is computed from this checkpoint (as in scripts/data_prep/build_stage4_weights.py)."
)

WEIGHTS_TSV = next((p for p in WEIGHTS_CANDIDATES if p.exists()), None)
assert WEIGHTS_TSV is not None, (
    "Missing Stage 4 weights TSV. Looked for:\n"
    + "\n".join([f"  - {p}" for p in WEIGHTS_CANDIDATES])
    + "\nGenerate with:\n"
    "  python scripts/data_prep/build_stage4_weights.py --ckpt model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt"
)

print("Weights TSV:", WEIGHTS_TSV)

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

import torch
from torch.utils.data import DataLoader

from geoseg.models.ftunetformer import ft_unetformer
from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug

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
# Stage 4 weighting hyperparams (MUST match scripts/data_prep/build_stage4_weights.py defaults)
EPS = 1e-6
BETA_TEMPER = 0.5      # (hardness + eps)^beta
GAMMA_RICH = 1.0       # (richness + eps)^gamma

# Note: final weight also includes clipping/normalisation/mixing; we read that from the TSV.

def _norm_id(x: str) -> str:
    """Strip _repN suffix so replicas share the same base weight (matches build_stage4_weights.py)."""
    if "_rep" in x:
        base, rep = x.rsplit("_rep", 1)
        if rep.isdigit():
            return base
    return x

def load_stage4_weight_map(weights_tsv: Path) -> dict[str, float]:
    m = {}
    with open(weights_tsv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, w = line.split("\t")
            m[k] = float(w)
    return m

w_map = load_stage4_weight_map(WEIGHTS_TSV)
w_vals = np.array(list(w_map.values()), dtype=np.float32)
print("Loaded Stage4 weights:", len(w_vals), "min/med/max:", float(w_vals.min()), float(np.median(w_vals)), float(w_vals.max()))

# pick representative "easy" and "hard" examples from final Stage4 weights
q_lo, q_hi = 0.10, 0.90
w_lo = np.quantile(w_vals, q_lo)
w_hi = np.quantile(w_vals, q_hi)

keys = list(w_map.keys())
weights_arr = np.array([w_map[k] for k in keys], dtype=np.float32)

idx_easy = int(np.argmin(np.abs(weights_arr - w_lo)))
idx_hard = int(np.argmin(np.abs(weights_arr - w_hi)))

easy_id = keys[idx_easy]
hard_id = keys[idx_hard]

print("Easy:", easy_id, "final_w:", float(w_map[easy_id]))
print("Hard:", hard_id, "final_w:", float(w_map[hard_id]))

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

print("Easy mask:", easy_mask_path.name, easy_mask.shape, easy_mask.dtype)
print("Hard mask:", hard_mask_path.name, hard_mask.shape, hard_mask.dtype)

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
def load_student_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Invalid Lightning checkpoint: {ckpt_path}")
    sd = ckpt["state_dict"]
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    if not net_sd:
        raise ValueError("Could not locate model weights in checkpoint")
    net.load_state_dict(net_sd, strict=False)
    return net

@torch.no_grad()
def compute_hardness_error_fraction(img_id: str, data_root: Path, ckpt_path: Path, device: str = "cuda") -> float:
    """
    Hardness proxy used by build_stage4_weights.py:
      err = mean(pred != gt) over pixels.
    We compute it for the single tile (val_aug).
    """
    ds = BiodiversityTrainDataset(data_root=str(data_root), transform=val_aug)

    # Find index by base id (train_rep includes reps; keys are base ids)
    target_idx = None
    for i, rid in enumerate(ds.img_ids):
        if _norm_id(rid) == img_id:
            target_idx = i
            break
    if target_idx is None:
        raise FileNotFoundError(f"Could not locate {img_id} in dataset ids under {data_root}")

    sample = ds[target_idx]
    img = sample["img"].unsqueeze(0)
    gt = sample["gt_semantic_seg"].unsqueeze(0)

    net = ft_unetformer(pretrained=False, weight_path=None, num_classes=6, decoder_channels=256)
    net = load_student_from_lightning_ckpt(net, ckpt_path)
    net.eval()

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    net.to(dev)
    img = img.to(dev)
    gt = gt.to(dev)

    pred = torch.argmax(net(img), dim=1)
    err = (pred != gt).float().mean().item()
    return float(err)

def compute_minority_richness_fraction(mask_np: np.ndarray) -> float:
    # minority = Settlement(4) or SemiNatural(5)
    m = (mask_np == 4) | (mask_np == 5)
    # exclude ignore index (0) from denom? build_stage4_weights.py uses mean over all pixels, incl bg
    # so we match it exactly: mean over full tile.
    return float(m.mean())

def component_weights_from_h_r(h: float, r: float) -> tuple[float, float]:
    hard_w = (h + EPS) ** BETA_TEMPER
    min_w = (r + EPS) ** GAMMA_RICH
    return float(hard_w), float(min_w)

# %%
# Compute components for the two shown tiles
easy_h = compute_hardness_error_fraction(easy_id, SPLIT_ROOT, STAGE3B_CKPT)
hard_h = compute_hardness_error_fraction(hard_id, SPLIT_ROOT, STAGE3B_CKPT)

easy_r = compute_minority_richness_fraction(easy_mask)
hard_r = compute_minority_richness_fraction(hard_mask)

easy_hw, easy_mw = component_weights_from_h_r(easy_h, easy_r)
hard_hw, hard_mw = component_weights_from_h_r(hard_h, hard_r)

easy_final = float(w_map[easy_id])
hard_final = float(w_map[hard_id])

print(f"Easy components: h={easy_h:.4f}, r={easy_r:.4f}, hard_w={easy_hw:.4f}, min_w={easy_mw:.4f}, final={easy_final:.4f}")
print(f"Hard components: h={hard_h:.4f}, r={hard_r:.4f}, hard_w={hard_hw:.4f}, min_w={hard_mw:.4f}, final={hard_final:.4f}")

# %%
def plot_figure06_hard_x_minority_sampling(
    easy_mask, hard_mask,
    easy_hw, easy_mw, easy_final,
    hard_hw, hard_mw, hard_final,
    out_pdf: Path
):
    fig = plt.figure(figsize=(8*3, 4.2*3), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.18], hspace=0.15, wspace=-0.15)

    # (a) easy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(easy_mask, cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_axis_off()
    add_panel_label_above_center(ax1, "(a)")
    add_weights_text(ax1, f"hard={easy_hw:.2f}, minority={easy_mw:.2f}, final={easy_final:.2f}", fontsize=28)

    # (b) hard
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(hard_mask, cmap=cmap, norm=norm, interpolation="nearest")
    ax2.set_axis_off()
    add_panel_label_above_center(ax2, "(b)")
    add_weights_text(ax2, f"hard={hard_hw:.2f}, minority={hard_mw:.2f}, final={hard_final:.2f}", fontsize=28)

    # legend row
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    handles = [Patch(facecolor=colors[i], label=CLASS_NAMES[i]) for i in range(6)]
    legend_ax.legend(handles=handles, loc="center",bbox_to_anchor=(0.5, 0.40), ncol=3, frameon=False, fontsize=32)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, pad_inches=0.02)
    plt.close(fig)
    print("Saved:", out_pdf)

plot_figure06_hard_x_minority_sampling(
    easy_mask, hard_mask,
    easy_hw, easy_mw, easy_final,
    hard_hw, hard_mw, hard_final,
    OUT_PDF
)
