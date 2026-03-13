#!/usr/bin/env python3
"""
Build Stage 4 sampling weights for Biodiversity train_rep.

Outputs:
  artifacts/stage4_sampling_weights.tsv

Each line:
  <img_id>\t<weight>

Hardness:
  computed from a checkpoint by running inference on train_rep (VAL-style aug),
  then measuring pixel error mass.

Minority richness:
  computed from GT masks as fraction of pixels in {Settlement(4), Seminatural(5)}.

Final weights:
  w_raw  = (hardness + eps)^beta * (richness + eps)^gamma
  w_clip = clip(w_raw, p_lo, p_hi)
  w_norm = w_clip / mean(w_clip)
  w_mix  = (1-alpha)*1 + alpha*w_norm

FAIRNESS NOTE:
- The checkpoint must be the Stage 3b finetuned student.
- Dataset MUST match Stage 4 training: biodiversity_split/train_rep
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityTrainDataset,
    val_aug,
)
from geoseg.models.ftunetformer import ft_unetformer


# ------------------
# Helpers
# ------------------
def _norm_id(x: str) -> str:
    """Strip _repN suffix so replicas share the same base weight."""
    if "_rep" in x:
        base, rep = x.rsplit("_rep", 1)
        if rep.isdigit():
            return base
    return x


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data/biodiversity_split/train_rep")
    p.add_argument("--ckpt", type=str, required=True, help="Stage 3b Lightning checkpoint")
    p.add_argument("--out", type=str, default="artifacts/stage4_weights.tsv")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--alpha_mix", type=float, default=0.5)
    p.add_argument("--beta_temper", type=float, default=0.5)
    p.add_argument("--gamma_rich", type=float, default=1.0)
    p.add_argument("--clip_lo", type=float, default=5.0)
    p.add_argument("--clip_hi", type=float, default=95.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output without prompting.")
    return p.parse_args()


def load_student(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = {
        k.replace("net.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }
    net.load_state_dict(sd, strict=False)
    print("[OK] Loaded Stage 3b student weights.")


@torch.no_grad()
def main():
    args = parse_args()
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {out_path}\n"
            "Pass --force to overwrite."
        )
    os.makedirs(out_path.parent, exist_ok=True)

    ds = BiodiversityTrainDataset(args.data_root, transform=val_aug)

    net = ft_unetformer(pretrained=False, weight_path=None, num_classes=6, decoder_channels=256)
    load_student(net, args.ckpt)
    net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    hardness_sum: dict[str, float] = {}
    hardness_cnt: dict[str, int] = {}
    richness_sum: dict[str, float] = {}
    richness_cnt: dict[str, int] = {}

    for batch in dl:
        img = batch["img"].to(device)
        gt = batch["gt_semantic_seg"].to(device)
        ids = batch["img_id"]

        pred = torch.argmax(net(img), dim=1)
        err = (pred != gt).float().mean(dim=(1, 2))
        rich = ((gt == 4) | (gt == 5)).float().mean(dim=(1, 2))

        for i, img_id in enumerate(ids):
            key = _norm_id(img_id)
            hardness_sum[key] = hardness_sum.get(key, 0.0) + err[i].item()
            hardness_cnt[key] = hardness_cnt.get(key, 0) + 1
            richness_sum[key] = richness_sum.get(key, 0.0) + rich[i].item()
            richness_cnt[key] = richness_cnt.get(key, 0) + 1

    keys = sorted(hardness_sum.keys())
    h = np.array([hardness_sum[k] / hardness_cnt[k] for k in keys])
    r = np.array([richness_sum[k] / richness_cnt[k] for k in keys])

    w_raw = (h + args.eps) ** args.beta_temper * (r + args.eps) ** args.gamma_rich
    w_clip = np.clip(
        w_raw,
        np.percentile(w_raw, args.clip_lo),
        np.percentile(w_raw, args.clip_hi),
    )
    w_norm = w_clip / (w_clip.mean() + args.eps)
    w_mix = (1 - args.alpha_mix) + args.alpha_mix * w_norm

    with open(args.out, "w") as f:
        for k, w in zip(keys, w_mix):
            f.write(f"{k}\t{w:.8f}\n")

    print(f"[OK] Wrote {len(keys)} weights -> {args.out}")


if __name__ == "__main__":
    main()
