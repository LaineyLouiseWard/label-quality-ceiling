#!/usr/bin/env python3
"""
Dump raw per-pixel 6-class softmax probabilities for ONE seed's checkpoint over
the Biodiversity validation split — the only new data the seed-disagreement
(label-ambiguity) analysis needs.

Reuses the model-loading and whole-tile single-pass inference from
evaluation/compute_metrics.py EXACTLY (same FT-UNetFormer arch, same checkpoint
key handling, same BiodiversityValDataset preprocessing → val_aug 512x512).
NO metrics, NO TTA: this is the raw single forward-pass softmax.

For each of the 231 val tiles, writes <out-dir>/<img_id>.npy holding the full
softmax probabilities as float16 with shape [6, H, W] (class-first; H=W=512
after val_aug). Probabilities are softmax over the class dim of the logits.

Run from repo root (once per seed, ~5 min):
  PYTHONPATH=. python scripts/analysis/dump_seed_softmax.py \
      --ckpt model_weights/seed42/stage3_clsbal/last.ckpt \
      --out-dir artifacts/seed_softmax/seed42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoseg.datasets.biodiversity_dataset import BiodiversityValDataset
from evaluation.compute_metrics import build_model, load_checkpoint_into_model


@torch.no_grad()
def dump_softmax(
    ckpt_path: Path,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
) -> int:
    model = build_model(num_classes=6).to(device)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    ds = BiodiversityValDataset(data_root=str(data_root))
    loader = DataLoader(
        dataset=ds,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    softmax = nn.Softmax(dim=1)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for batch in tqdm(loader, desc=f"Dumping softmax {ckpt_path.parent.name}", leave=False):
        images = batch["img"].to(device)            # (1,3,H,W)
        img_id = batch["img_id"][0]
        probs = softmax(model(images))              # (1,6,H,W) softmax over class dim
        prob = probs[0].cpu().numpy().astype(np.float16)  # (6,H,W) class-first
        np.save(out_dir / f"{img_id}.npy", prob)
        n += 1

    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump raw per-pixel 6-class softmax over the val split for one seed's checkpoint."
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the .ckpt to load.")
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path (accepted for parity with other scripts; the arch is fixed "
             "FT-UNetFormer(num_classes=6, decoder_channels=256) via compute_metrics.build_model).",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/val",
        help="Val split root (contains images/ and masks/). 231 tiles.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to write <img_id>.npy softmax arrays (shape [6,H,W], float16).",
    )
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    ckpt_path = Path(args.ckpt).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    if args.config is not None:
        logging.info(f"--config given ({args.config}); arch is fixed FT-UNetFormer(6, decoder_channels=256).")

    logging.info(f"Checkpoint: {ckpt_path}")
    logging.info(f"Val data root: {data_root}")
    logging.info(f"Output dir: {out_dir}  (float16, shape [6,H,W] per tile)")
    logging.info(f"Device: {device}")

    n = dump_softmax(ckpt_path, data_root, out_dir, device)
    logging.info(f"Done. Wrote {n} softmax arrays to {out_dir}")


if __name__ == "__main__":
    main()
