#!/usr/bin/env python3
"""
Donor-quality scores for TARGETED Settlement copy-paste (MSAug-style confidence selection;
Gong et al. 2024, 10.1016/j.displa.2024.102779 — targeted paste beat random +1.3 vs +0.5 mIoU on
iSAID). Course-correction §15: Settlement is pixel-scarce, not confused (recall 85.2), so the lever
is to add CLEAN, learnable Settlement pixels — i.e. paste high-confidence, well-segmented donor
instances, NOT random tiles (which include noisy/ambiguous Settlement).

For each Settlement-bearing train tile, runs the frozen Stage 2b student and records:
  - confidence = mean predicted Settlement softmax prob on the GT-Settlement pixels
  - recall     = TP/(TP+FN) for Settlement on that tile
  - area       = Settlement pixel fraction
Donor sampling later weights tiles by confidence (clean instances) — see
geoseg/datasets/biodiversity_dataset.configure_settlement_copypaste(targeted=True).

Run:
  PYTHONPATH=. python scripts/data_prep/build_donor_quality.py \
    --ckpt model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug
from geoseg.models.ftunetformer import ft_unetformer

SETTLEMENT = 4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/biodiversity_split/train")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="artifacts/donor_quality_settlement.tsv")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_student(net, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    base = os.path.basename(ckpt_path)
    print(f"[G8] ckpt={base} missing={len(missing)} unexpected={len(unexpected)}")
    assert base == "stage2b_oem_finetune.ckpt" and len(missing) == 0, "wrong/partial teacher ckpt"


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and not args.force:
        raise FileExistsError(f"{out} exists; pass --force")

    ds = BiodiversityTrainDataset(args.data_root, transform=val_aug)
    net = ft_unetformer(pretrained=False, weight_path=None, num_classes=6, decoder_channels=256)
    load_student(net, args.ckpt)
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    rows = []
    for batch in dl:
        img = batch["img"].to(device)
        gt = batch["gt_semantic_seg"].to(device)
        ids = batch["img_id"]
        prob = F.softmax(net(img), dim=1)
        pred = prob.argmax(dim=1)
        for i, iid in enumerate(ids):
            gt_s = gt[i] == SETTLEMENT
            n = int(gt_s.sum().item())
            if n == 0:
                continue  # only Settlement-bearing tiles are donors
            conf = float(prob[i, SETTLEMENT][gt_s].mean().item())
            tp = int(((pred[i] == SETTLEMENT) & gt_s).sum().item())
            recall = tp / n
            area = n / gt_s.numel()
            rows.append((iid, conf, recall, area))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("img_id\tconfidence\trecall\tarea\n")
        for iid, conf, rec, area in rows:
            f.write(f"{iid}\t{conf:.6f}\t{rec:.6f}\t{area:.8f}\n")
    confs = np.array([r[1] for r in rows])
    print(f"[OK] {len(rows)} Settlement donors -> {out}  "
          f"(confidence: min {confs.min():.3f} / median {np.median(confs):.3f} / max {confs.max():.3f})")


if __name__ == "__main__":
    main()
