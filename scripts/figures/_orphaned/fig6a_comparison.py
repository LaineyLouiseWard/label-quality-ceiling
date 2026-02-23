#!/usr/bin/env python3
"""
fig6a_comparison.py

Single-sample qualitative comparison for ONE tile ID.

Default output panels (1x4):
(a) RGB image
(b) Original GT mask (from split masks/)
(c) Stage 1 prediction
(d) Stage 6 prediction

Optional:
If you pass --corrected-mask, we output 1x5 and put corrected GT as (e).

Key visual fix:
- Many tiles are clipped to AOIs, so the RGB image has black "no-data" regions.
- For *visualisation*, we force GT + predictions to class 0 (black) wherever RGB is near-black.

Example (no corrected mask):
python -m scripts.figures.fig6a_comparison \
  --id den5_0003 \
  --split test \
  --baseline-ckpt model_weights/biodiversity/stage1_baseline_ftunetformer \
  --stage6-ckpt   model_weights/biodiversity/stage6_final_kd_ftunetformer \
  --out-pdf figures/fig6a_comparison.pdf

Example (with corrected mask on far right):
python -m scripts.figures.fig6a_comparison \
  --id den5_0003 \
  --split test \
  --corrected-mask scripts/figures/den5_0003.png \
  --out-pdf figures/fig6a_comparison.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from PIL import Image

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
)
from geoseg.models.ftunetformer import ft_unetformer


# ---------- palette (match your taxonomy) ----------
PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # Background / void
    1: (250, 62, 119),    # Forest
    2: (168, 232, 84),    # Grassland
    3: (242, 180, 92),    # Cropland
    4: (116, 116, 116),   # Settlement
    5: (255, 214, 33),    # Seminatural
}
CLASS_NAMES = ["Background", "Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]
MASK_CMAP = ListedColormap([tuple(np.array(PALETTE[i]) / 255.0) for i in range(6)])

# Albumentations Normalize() defaults
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(12):
        if (p / "scripts").is_dir() and (p / "config").is_dir() and (p / "geoseg").is_dir():
            return p
        p = p.parent
    raise RuntimeError("Could not find repo root. Run from inside the repo.")


def resolve_ckpt(path_like: str, pattern: str = "*.ckpt") -> Path:
    """
    Accept either:
      - a direct .ckpt path
      - a directory -> pick most recent .ckpt in it (recursive)
    """
    p = Path(path_like).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {p}")
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    ckpts = list(p.rglob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {p}")

    ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpts[0]


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    """
    Load Lightning .ckpt into raw nn.Module.
    Handles 'net.' or 'model.' prefixes.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("net."):
            cleaned[k.replace("net.", "", 1)] = v
        elif k.startswith("model."):
            cleaned[k.replace("model.", "", 1)] = v
        else:
            cleaned[k] = v

    net.load_state_dict(cleaned, strict=False)
    return net


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (C,H,W) float tensor AFTER albu.Normalize(mean/std).
    returns: (H,W,3) uint8 RGB for plotting.
    """
    img = img_t.detach().cpu().numpy().astype(np.float32)
    if img.shape[0] < 3:
        img = np.repeat(img, 3, axis=0)
    img = img[:3]
    img = (img.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def make_valid_mask_from_rgb(img_rgb: np.ndarray, thresh: int = 5) -> np.ndarray:
    """
    img_rgb: (H,W,3) uint8
    Returns (H,W) bool where True means "valid imagery".
    Treat near-black as nodata/outside AOI.
    """
    return (img_rgb[..., 0] > thresh) | (img_rgb[..., 1] > thresh) | (img_rgb[..., 2] > thresh)


@torch.no_grad()
def predict(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    x = img_t.unsqueeze(0).to(device)
    logits = net(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def load_corrected_mask(path: Path) -> np.ndarray:
    m = Image.open(path).convert("L")  # expect single-channel label indices
    return np.array(m).astype(np.uint8)


def find_index_by_id(ds, target_id: str) -> int:
    for i in range(len(ds)):
        s = ds[i]
        if s["img_id"] == target_id:
            return i
    raise ValueError(f"ID not found in dataset split: {target_id}")


def plot_row(
    img_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_base: np.ndarray,
    pred_s6: np.ndarray,
    out_pdf: Path,
    tile_id: str,
    corrected_mask: np.ndarray | None = None,
) -> None:
    if corrected_mask is None:
        ncols = 4
        titles = ["(a) Image", "(b) Ground truth", "(c) Stage 1", "(d) Stage 6"]
    else:
        ncols = 5
        titles = ["(a) Image", "(b) Ground truth", "(c) Stage 1", "(d) Stage 6", "(e) Corrected GT"]

    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), constrained_layout=True)

    axes[0].imshow(img_rgb)
    axes[0].set_title(titles[0])
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap=MASK_CMAP, vmin=0, vmax=5, interpolation="nearest")
    axes[1].set_title(titles[1])
    axes[1].axis("off")

    axes[2].imshow(pred_base, cmap=MASK_CMAP, vmin=0, vmax=5, interpolation="nearest")
    axes[2].set_title(titles[2])
    axes[2].axis("off")

    axes[3].imshow(pred_s6, cmap=MASK_CMAP, vmin=0, vmax=5, interpolation="nearest")
    axes[3].set_title(titles[3])
    axes[3].axis("off")

    if corrected_mask is not None:
        axes[4].imshow(corrected_mask, cmap=MASK_CMAP, vmin=0, vmax=5, interpolation="nearest")
        axes[4].set_title(titles[4])
        axes[4].axis("off")

    fig.suptitle(tile_id, y=1.02, fontsize=12)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_pdf)) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a single qualitative comparison PDF for one tile.")
    ap.add_argument("--id", required=True, help="Tile id stem (e.g., den5_0003)")
    ap.add_argument("--split", choices=["val", "test"], default="test", help="Which split to load the GT from.")
    ap.add_argument("--data-root", type=str, default="data/biodiversity_split", help="Root containing val/ and test/")
    ap.add_argument(
        "--baseline-ckpt",
        type=str,
        default="model_weights/biodiversity/stage1_baseline_ftunetformer",
        help="Stage 1 checkpoint OR folder containing .ckpt",
    )
    ap.add_argument(
        "--stage6-ckpt",
        type=str,
        default="model_weights/biodiversity/stage6_final_kd_ftunetformer",
        help="Stage 6 checkpoint OR folder containing .ckpt",
    )
    ap.add_argument("--corrected-mask", type=str, default=None, help="Optional corrected GT PNG path.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--out-pdf", type=str, default="figures/fig6a_comparison.pdf", help="Output PDF path.")
    args = ap.parse_args()

    repo_root = find_repo_root()

    data_root = (repo_root / args.data_root).resolve()
    split_root = data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    # dataset provides image tensor + gt mask tensor
    if args.split == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
    else:
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))

    idx = find_index_by_id(ds, args.id)
    sample = ds[idx]

    img_t = sample["img"]                       # (C,H,W) normalized float
    gt_t = sample["gt_semantic_seg"]            # (H,W)
    tile_id = sample["img_id"]

    img_rgb = denormalize_to_uint8(img_t)
    gt_mask = gt_t.detach().cpu().numpy().astype(np.uint8)

    # valid imagery mask from RGB (outside AOI is near-black)
    valid = make_valid_mask_from_rgb(img_rgb)

    # For visuals: force nodata/outside AOI to background (0)
    gt_mask_vis = gt_mask.copy()
    gt_mask_vis[~valid] = 0

    # load checkpoints
    base_ckpt = resolve_ckpt(str((repo_root / args.baseline_ckpt).resolve()))
    s6_ckpt = resolve_ckpt(str((repo_root / args.stage6_ckpt).resolve()))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    base_net = ft_unetformer(pretrained=False, weight_path=None, num_classes=6, decoder_channels=256)
    s6_net = ft_unetformer(pretrained=False, weight_path=None, num_classes=6, decoder_channels=256)

    base_net = load_net_from_lightning_ckpt(base_net, base_ckpt).to(device)
    s6_net = load_net_from_lightning_ckpt(s6_net, s6_ckpt).to(device)

    # predictions
    pred_base = predict(base_net, img_t, device)
    pred_s6 = predict(s6_net, img_t, device)

    # For visuals: force preds to background where image is nodata
    pred_base_vis = pred_base.copy()
    pred_s6_vis = pred_s6.copy()
    pred_base_vis[~valid] = 0
    pred_s6_vis[~valid] = 0

    corrected_vis = None
    if args.corrected_mask is not None:
        corrected_path = (repo_root / args.corrected_mask).expanduser().resolve()
        if not corrected_path.exists():
            raise FileNotFoundError(f"Corrected mask not found: {corrected_path}")

        corrected = load_corrected_mask(corrected_path)

        # resize to match GT if needed (nearest-neighbour for labels)
        if corrected.shape != gt_mask.shape:
            corrected = np.array(
                Image.fromarray(corrected).resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST),
                dtype=np.uint8,
            )

        corrected_vis = corrected.copy()
        corrected_vis[~valid] = 0

    out_pdf = (repo_root / args.out_pdf).resolve()
    plot_row(
        img_rgb=img_rgb,
        gt_mask=gt_mask_vis,
        pred_base=pred_base_vis,
        pred_s6=pred_s6_vis,
        corrected_mask=corrected_vis,
        out_pdf=out_pdf,
        tile_id=tile_id,
    )

    print("Saved:", out_pdf)
    print("Baseline ckpt:", base_ckpt)
    print("Stage6 ckpt:", s6_ckpt)
    print("Split:", args.split, "| Data:", split_root)


if __name__ == "__main__":
    main()
