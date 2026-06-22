#!/usr/bin/env python3
"""
scripts/figures/graphical_abstract.py

Save GA-ready prediction masks (as separate PNGs) for one chosen tile:
- Stage 1 (baseline)
- Stage 3 (hard x minority sampler)
- Stage 4 (KD)

NOTE: the companion graphical_abstract_tikz.tex is now a 4-stage layout; its Key-Result
numbers and the three stage PNGs still need to be regenerated from the median-seed checkpoints.

Outputs:
  figures/graphical_abstract/<img_id>_stage1_baseline.png
  figures/graphical_abstract/<img_id>_stage3_sampler.png
  figures/graphical_abstract/<img_id>_stage4_kd.png

Optional:
  figures/graphical_abstract/<img_id>_rgb.png
  figures/graphical_abstract/<img_id>_gt.png

Run:
  python scripts/figures/graphical_abstract.py --img-id biodiversity_1310 --device cuda --also-save-rgb-gt
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# Repo root for imports (fixes ModuleNotFoundError: geoseg)
# -----------------------------------------------------------------------------
def find_repo_root_for_imports() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "geoseg").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("Could not find repo root for imports")

repo_root = find_repo_root_for_imports()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
    CLASSES,
    PALETTE as DATASET_PALETTE,
)
from geoseg.models.ftunetformer import ft_unetformer


# -----------------------------------------------------------------------------
# Canonical palette
# -----------------------------------------------------------------------------
PALETTE: Dict[int, Tuple[int, int, int]] = {i: tuple(rgb) for i, rgb in enumerate(DATASET_PALETTE)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def resolve_ckpt(path_like: str) -> Path:
    """
    Accept either:
      - a direct .ckpt path
      - a directory -> pick most recent .ckpt under it (recursive)
    Handles nested run folders (e.g. stage4_kd/stage4_kd/*.ckpt).
    """
    p = Path(path_like).expanduser().resolve()

    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {p}")
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    ckpts = list(p.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {p}")

    ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return ckpts[0]


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
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


def build_ftunetformer() -> torch.nn.Module:
    return ft_unetformer(num_classes=len(CLASSES), pretrained=False)


@torch.no_grad()
def predict_mask(net: torch.nn.Module, img_t: torch.Tensor, device: torch.device) -> np.ndarray:
    net.eval()
    logits = net(img_t.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def denormalize_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.detach().cpu().numpy().astype(np.float32)
    img = img[:3] if img.shape[0] >= 3 else np.repeat(img, 3, axis=0)
    img = (img.transpose(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


def colorize_mask(mask: np.ndarray, invalid: np.ndarray | None = None) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, rgb in PALETTE.items():
        out[mask == k] = rgb
    if invalid is not None:
        out[invalid] = (0, 0, 0)
    return out


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _try_load_from_split(split_root: Path, img_id: str):
    if not split_root.exists():
        return None

    if split_root.name == "val":
        ds = BiodiversityValDataset(data_root=str(split_root))
    elif split_root.name == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(split_root))
    else:
        return None

    if img_id not in ds.img_ids:
        return None

    idx = ds.img_ids.index(img_id)
    item = ds[idx]
    return item["img"], item["gt_semantic_seg"]


def load_tile_any_split(data_root: Path, img_id: str, split_order: List[str]):
    for sp in split_order:
        got = _try_load_from_split(data_root / sp, img_id)
        if got is not None:
            return got
    raise FileNotFoundError(f"{img_id} not found under {data_root} in splits {split_order}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-id", default="biodiversity_1310", help="Tile stem (no extension).")
    ap.add_argument("--data-root", default="data/biodiversity_split", help="Root containing val/ test/")
    ap.add_argument("--split-order", default="val,test", help="Search order, e.g. val,test")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # IMPORTANT: update defaults to your CURRENT folder names
    ap.add_argument("--stage1-ckpt", default="model_weights/biodiversity/stage1_baseline")
    ap.add_argument("--stage3-ckpt", default="model_weights/biodiversity/stage3_sampler")
    ap.add_argument("--stage4-ckpt", default="model_weights/biodiversity/stage4_kd")

    ap.add_argument("--out-dir", default="figures/graphical_abstract")
    ap.add_argument("--also-save-rgb-gt", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data_root = (repo_root / args.data_root).resolve()
    split_order = [s.strip() for s in args.split_order.split(",") if s.strip()]

    # load tile (val first, fallback to test)
    img_t, gt_t = load_tile_any_split(data_root, args.img_id, split_order)

    img_rgb = denormalize_to_uint8(img_t)
    gt = gt_t.detach().cpu().numpy().astype(np.uint8)

    # Use GT==0 as invalid/outside AOI for display consistency
    invalid = (gt == 0)

    # resolve checkpoints (robust to nested dirs)
    ckpt_s1 = resolve_ckpt(str((repo_root / args.stage1_ckpt).resolve()))
    ckpt_s3 = resolve_ckpt(str((repo_root / args.stage3_ckpt).resolve()))
    ckpt_s4 = resolve_ckpt(str((repo_root / args.stage4_ckpt).resolve()))

    # load nets
    net_s1 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s1).to(device)
    net_s3 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s3).to(device)
    net_s4 = load_net_from_lightning_ckpt(build_ftunetformer(), ckpt_s4).to(device)

    # predict
    p1 = predict_mask(net_s1, img_t, device)
    p3 = predict_mask(net_s3, img_t, device)
    p4 = predict_mask(net_s4, img_t, device)

    # colorize + apply invalid mask as black
    p1_rgb = colorize_mask(p1, invalid=invalid)
    p3_rgb = colorize_mask(p3, invalid=invalid)
    p4_rgb = colorize_mask(p4, invalid=invalid)

    out_dir = (repo_root / args.out_dir).resolve()
    save_png(out_dir / f"{args.img_id}_stage1_baseline.png", p1_rgb)
    save_png(out_dir / f"{args.img_id}_stage3_sampler.png", p3_rgb)
    save_png(out_dir / f"{args.img_id}_stage4_kd.png", p4_rgb)

    if args.also_save_rgb_gt:
        save_png(out_dir / f"{args.img_id}_rgb.png", img_rgb)
        save_png(out_dir / f"{args.img_id}_gt.png", colorize_mask(gt, invalid=invalid))

    print("Saved to:", out_dir)
    print("Stage1 ckpt:", ckpt_s1)
    print("Stage3 (sampler) ckpt:", ckpt_s3)
    print("Stage4 (KD) ckpt:", ckpt_s4)
    print("Device:", device)


if __name__ == "__main__":
    main()
