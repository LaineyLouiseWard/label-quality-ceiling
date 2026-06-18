#!/usr/bin/env python3
"""
Gold-standard grounding of the KD Rangeland-split prior (alpha).

The KD mapping splits the frozen teacher's OEM "Rangeland" mass to Grassland(alpha) /
Seminatural(1-alpha). The principled alpha is the *conditional* prior
    alpha = P(GT = Grassland | teacher predicts Rangeland)  renormalised over {Grassland, Seminatural}
i.e. of the pixels the teacher calls Rangeland, what fraction are truly Grassland vs Seminatural.

This is a property of the FROZEN teacher + the FIXED training masks only — it does NOT depend on
the student seed, so it is computed ONCE and reused for the whole seed campaign.

Reports two estimators:
  - hard : pixels where teacher argmax == Rangeland  (interpretable headline)
  - soft : every pixel weighted by teacher softmax P(Rangeland)  (faithful to how the mapping distributes mass)

Run from repo root:
  PYTHONPATH=. python scripts/analysis/teacher_rangeland_conditional_prior.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug
from geoseg.models.unet import TeacherUNet
from geoseg.taxonomy import OEM_NATIVE_CLASSES, STUDENT_CLASSES

OEM_RANGELAND = 2          # teacher output channel (geoseg.taxonomy.OEM_NATIVE_CLASSES)
GRASSLAND, SEMINATURAL = 2, 5
TEACHER_CKPT = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


@torch.no_grad()
def main() -> None:
    assert OEM_NATIVE_CLASSES[OEM_RANGELAND] == "Rangeland"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = TeacherUNet(num_classes=9, pretrained=False)
    teacher.load_checkpoint(TEACHER_CKPT)
    teacher.freeze()
    teacher.to(dev)

    ds = BiodiversityTrainDataset(data_root="data/biodiversity_split/train", transform=val_aug)
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False, pin_memory=True)

    hard_gt_hist = np.zeros(6, dtype=np.int64)      # GT histogram of teacher-argmax-Rangeland pixels
    soft_gt_mass = np.zeros(6, dtype=np.float64)    # GT histogram weighted by teacher P(Rangeland)
    n_tiles = 0

    for batch in dl:
        img = batch["img"].to(dev)
        gt = batch["gt_semantic_seg"].numpy()       # (B,H,W) ints 0..5
        logits = teacher(img)                        # (B,9,H,W)
        prob_r = torch.softmax(logits, dim=1)[:, OEM_RANGELAND].cpu().numpy()  # (B,H,W)
        pred = logits.argmax(dim=1).cpu().numpy()    # (B,H,W)
        is_r = pred == OEM_RANGELAND

        for c in range(6):
            gt_c = gt == c
            hard_gt_hist[c] += int((gt_c & is_r).sum())
            soft_gt_mass[c] += float(prob_r[gt_c].sum())
        n_tiles += img.shape[0]

    def split(hist):
        g, s = hist[GRASSLAND], hist[SEMINATURAL]
        return (g / (g + s)) if (g + s) > 0 else float("nan")

    print(f"\nTiles: {n_tiles}   teacher: {TEACHER_CKPT}")
    print("\n-- HARD (pixels where teacher argmax == Rangeland) --")
    tot = hard_gt_hist.sum()
    for c in range(6):
        print(f"  GT {STUDENT_CLASSES[c]:<12} {int(hard_gt_hist[c]):>12,}  ({100*hard_gt_hist[c]/max(tot,1):5.2f}% of Rangeland px)")
    print(f"  -> alpha_hard = G/(G+S) = {split(hard_gt_hist):.4f}")

    print("\n-- SOFT (GT mass weighted by teacher P(Rangeland)) --")
    for c in range(6):
        print(f"  GT {STUDENT_CLASSES[c]:<12} {soft_gt_mass[c]:>14,.0f}")
    print(f"  -> alpha_soft = G/(G+S) = {split(soft_gt_mass):.4f}")

    print(f"\n(marginal prevalence prior was 0.896; config currently 0.70)")


if __name__ == "__main__":
    main()
