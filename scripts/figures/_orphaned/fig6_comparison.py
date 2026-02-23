#!/usr/bin/env python3
"""
Fig 6: 4-panel qualitative comparison (random samples)

Panels (1x4):
(a) RGB image
(b) "Actual" mask (val mask; note: not available for test split)
(c) Baseline prediction (Stage 1)
(d) Final prediction (Stage 6)

Defaults:
- samples 50 random tiles from data/biodiversity_split/val
- runs inference directly from Lightning .ckpt checkpoints
- writes a single multipage PDF to: figures/fig6_comparison/fig6_comparison_random50.pdf

Usage:
python -m scripts.figures.fig6_comparison --n 50 --seed 0 --save-individual

Optional:
  --n 12
  --seed 7
  --baseline-ckpt model_weights/.../something.ckpt   (or a directory containing .ckpt)
  --stage6-ckpt   model_weights/.../something.ckpt   (or a directory containing .ckpt)
  --save-individual   (also saves one PDF per sample)
  --ids biodiversity_1446 biodiversity_2031          (plot only specific IDs; overrides random sampling)

Usage:
50 random samples combined:
python -m scripts.figures.fig6_comparison \
  --n 50 \
  --seed 0 \
  --save-individual

One specific sample:
python -m scripts.figures.fig6_comparison \
  --ids biodiversity_1446 \
  --save-individual

  """

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from geoseg.datasets.biodiversity_dataset import BiodiversityValDataset
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------
# Style (match your figs)
# -----------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)

COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (250, 62, 119),   # Forest
    2: (168, 232, 84),   # Grassland
    3: (242, 180, 92),   # Cropland
    4: (59, 141, 247),   # Settlement
    5: (255, 214, 33),   # Semi-Natural Grassland
}

# Albumentations Normalize() defaults
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(12):
        if (p / "scripts").is_dir() and (p / "config").is_dir() and (p / "geoseg").is_dir():
            return p
        p = p.parent
    raise RuntimeError(
        "Could not find repo root (expected scripts/, config/, geoseg/). Run from inside the repo."
    )


def resolve_ckpt(path_like: str, pattern: str = "*.ckpt") -> Path:
    """
    Accept either:
      - a direct .ckpt path
      - a directory, in which case we pick the most recently modified .ckpt inside (recursive)
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
        raise FileNotFoundError(f"No checkpoints found under directory: {p}")

    ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpts[0]


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Invalid Lightning checkpoint: {ckpt_path}")

    sd = ckpt["state_dict"]

    # Extract underlying segmentation network weights
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}

    if not net_sd:
        raise ValueError(
            "Could not locate model weights in checkpoint (expected keys starting with 'net.' or 'model.')."
        )

    net.load_state_dict(net_sd, strict=False)
    return net


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: (C,H,W) float tensor after albu.Normalize (ImageNet mean/std).
    returns: (H,W,3) uint8 for plotting
    """
    img = img_t.detach().cpu().numpy().astype(np.float32)
    if img.shape[0] < 3:
        img = np.repeat(img, 3, axis=0)
    img = img[:3]  # (3,H,W)
    img = (img.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) int
    returns RGB uint8 (H,W,3)
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in COLOR_MAP.items():
        out[mask == k] = np.array(rgb, dtype=np.uint8)
    return out


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    x = img_t.unsqueeze(0).to(device)
    logits = net(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def make_row_figure(
    img_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    base_rgb: np.ndarray,
    s6_rgb: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(10, 2.6),
        constrained_layout=True
    )

    labels = ["(a)", "(b)", "(c)", "(d)"]
    panels = [img_rgb, mask_rgb, base_rgb, s6_rgb]

    for ax, lab, panel in zip(axes, labels, panels):
        ax.imshow(panel)
        ax.text(
            0.5, 1.05, lab,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )
        ax.set_aspect("equal")
        ax.axis("off")

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate Fig6 4-panel qualitative comparisons (random samples or specific IDs)."
    )
    ap.add_argument("--n", type=int, default=50, help="Number of random samples to plot.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    ap.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Specific image IDs to plot (e.g., biodiversity_1446). Overrides random sampling.",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/val",
        help="Val split root containing images/ and masks/.",
    )
    ap.add_argument(
        "--baseline-ckpt",
        type=str,
        default="model_weights/biodiversity/stage1_baseline_ftunetformer",
        help="Stage 1 baseline .ckpt OR a directory containing .ckpt.",
    )
    ap.add_argument(
        "--stage6-ckpt",
        type=str,
        default="model_weights/biodiversity/stage6_final_kd_ftunetformer",
        help="Stage 6 final .ckpt OR a directory containing .ckpt.",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="figures/fig6_comparison",
        help="Repo-level output folder for PDFs.",
    )
    ap.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save one PDF per sample in out-dir/individual/",
    )
    args = ap.parse_args()

    repo_root = find_repo_root()

    data_root = (repo_root / args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")

    base_ckpt = resolve_ckpt(str(repo_root / args.baseline_ckpt))
    s6_ckpt = resolve_ckpt(str(repo_root / args.stage6_ckpt))

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # dataset uses val_aug by default (resize->512 + Normalize)
    ds = BiodiversityValDataset(data_root=str(data_root))

    # Select indices
    if args.ids is not None:
        id_to_idx = {ds[i]["img_id"]: i for i in range(len(ds))}
        missing = [k for k in args.ids if k not in id_to_idx]
        if missing:
            raise ValueError(f"IDs not found in dataset: {missing}")
        idxs = [id_to_idx[k] for k in args.ids]
    else:
        n = min(args.n, len(ds))
        rng = random.Random(args.seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        idxs = idxs[:n]

    # Build nets and load weights
    num_classes = 6
    base_net = ft_unetformer(pretrained=False, weight_path=None, num_classes=num_classes, decoder_channels=256)
    s6_net = ft_unetformer(pretrained=False, weight_path=None, num_classes=num_classes, decoder_channels=256)

    base_net = load_net_from_lightning_ckpt(base_net, base_ckpt).to(device)
    s6_net = load_net_from_lightning_ckpt(s6_net, s6_ckpt).to(device)

    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids:
        pdf_name = "fig6_comparison_" + "_".join(args.ids) + ".pdf"
    else:
        pdf_name = "fig6_comparison_random50.pdf"

    pdf_path = out_dir / pdf_name
    indiv_dir = out_dir / "individual"
    if args.save_individual:
        indiv_dir.mkdir(parents=True, exist_ok=True)

    print("Repo root:", repo_root)
    print("Data root:", data_root)
    print("Baseline ckpt:", base_ckpt)
    print("Stage6 ckpt:", s6_ckpt)
    print("Device:", device)
    print("Saving:", pdf_path)

    with PdfPages(str(pdf_path)) as pdf:
        for i in idxs:
            sample = ds[i]
            img_t = sample["img"]
            mask_t = sample["gt_semantic_seg"]
            img_id = sample["img_id"]

            # panel (a): RGB image
            img_rgb = denormalize_to_uint8(img_t)

            # panel (b): actual val mask
            mask_np = mask_t.detach().cpu().numpy().astype(np.uint8)
            mask_rgb = colorize_mask(mask_np)

            # panel (c)/(d): predictions
            base_pred = predict_mask(base_net, img_t, device)
            s6_pred = predict_mask(s6_net, img_t, device)
            base_rgb = colorize_mask(base_pred)
            s6_rgb = colorize_mask(s6_pred)

            fig = make_row_figure(img_rgb, mask_rgb, base_rgb, s6_rgb)

            # only add ID title for the multipage PDF (not individual figures)
            if not args.save_individual:
                fig.suptitle(img_id, y=0.98, fontsize=10)


            pdf.savefig(fig, bbox_inches="tight")
            if args.save_individual:
                fig.savefig(indiv_dir / f"{img_id}.pdf", bbox_inches="tight")
            plt.close(fig)

    print("Saved:", pdf_path)


if __name__ == "__main__":
    main()
