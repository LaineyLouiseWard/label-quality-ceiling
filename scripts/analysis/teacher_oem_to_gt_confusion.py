#!/usr/bin/env python3
"""
Ground EVERY OEM->student KD mapping in data (generalises the Rangeland-split prior).

One forward pass of the FROZEN teacher over the training tiles produces a 9x6 matrix:
    rows  = teacher's native OEM prediction (argmax / prob mass)
    cols  = ground-truth student class
From it, every hard-coded mapping in geoseg.taxonomy.oem_to_student_kd is checkable:
"of the pixels the teacher calls OEM-class X, what are they REALLY?" — e.g. is Bareland->Seminatural(1.0)
justified, or do teacher-Bareland pixels actually fall elsewhere?

This depends only on the frozen teacher + fixed training masks, so it is computed ONCE for the campaign.
Also serves as the teacher-reliability evidence for the manuscript.

Run from repo root:
  PYTHONPATH=. python scripts/analysis/teacher_oem_to_gt_confusion.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug
from geoseg.models.unet import TeacherUNet
from geoseg.taxonomy import OEM_NATIVE_CLASSES, STUDENT_CLASSES, oem_to_student_kd

TEACHER_CKPT = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"
NS, NO = len(STUDENT_CLASSES), len(OEM_NATIVE_CLASSES)  # 6, 9


@torch.no_grad()
def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = TeacherUNet(num_classes=NO, pretrained=False)
    teacher.load_checkpoint(TEACHER_CKPT)
    teacher.freeze(); teacher.to(dev)

    ds = BiodiversityTrainDataset(data_root="data/biodiversity_split/train", transform=val_aug)
    dl = DataLoader(ds, batch_size=8, num_workers=4, shuffle=False, pin_memory=True)

    hard = np.zeros((NO, NS), dtype=np.int64)     # [teacher argmax OEM, GT student]
    soft = np.zeros((NO, NS), dtype=np.float64)   # [teacher prob mass on OEM, GT student]

    for batch in dl:
        img = batch["img"].to(dev)
        gt = batch["gt_semantic_seg"].numpy()                       # (B,H,W) 0..5
        logits = teacher(img)                                       # (B,9,H,W)
        probs = torch.softmax(logits, dim=1).cpu().numpy()          # (B,9,H,W)
        pred = logits.argmax(dim=1).cpu().numpy()                   # (B,H,W)
        for s in range(NS):
            gmask = gt == s
            if not gmask.any():
                continue
            # hard: count teacher-argmax per OEM among GT==s pixels
            po = pred[gmask]
            bc = np.bincount(po, minlength=NO)[:NO]
            hard[:, s] += bc
            # soft: teacher prob mass per OEM among GT==s pixels
            soft[:, s] += probs[:, :, :, :].transpose(0, 2, 3, 1)[gmask].sum(axis=0)

    import os
    os.makedirs("artifacts", exist_ok=True)
    np.savez("artifacts/teacher_oem_gt_confusion.npz", hard=hard, soft=soft,
             oem_classes=np.array(OEM_NATIVE_CLASSES), student_classes=np.array(STUDENT_CLASSES))
    print("saved artifacts/teacher_oem_gt_confusion.npz")

    kd_map = oem_to_student_kd(alpha=0.73)  # current grounded mapping, to annotate

    def show(mat, title):
        print(f"\n==== {title} :  rows = teacher OEM prediction, cols = GT student (row-normalised %) ====")
        head = "teacher-OEM \\ GT |" + "".join(f"{c[:9]:>11}" for c in STUDENT_CLASSES)
        print(head); print("-" * len(head))
        for o in range(NO):
            row = mat[o]
            tot = row.sum()
            pct = 100 * row / tot if tot > 0 else row
            tgt = ",".join(STUDENT_CLASSES[i] for i, _ in kd_map.get(o, []))
            line = f"{OEM_NATIVE_CLASSES[o]:<16}|" + "".join(f"{p:>10.1f}%" for p in pct)
            print(f"{line}   [KD->{tgt}]  (n={int(tot):,})" if mat is hard else f"{line}   [KD->{tgt}]")

    show(hard, "HARD (argmax)")
    show(soft, "SOFT (prob-weighted)")

    # Key derived groundings
    def split(mat, o, a, b):
        x, y = mat[o, a], mat[o, b]
        return x / (x + y) if (x + y) > 0 else float("nan")

    G, S, BG, SET = 2, 5, 0, 4
    print("\n==== Derived groundings ====")
    print(f"Rangeland split alpha=G/(G+S): hard={split(hard,2,G,S):.3f}  soft={split(soft,2,G,S):.3f}  (config=0.73)")
    print("Bareland (KD currently -> Seminatural 1.0). GT of teacher-Bareland pixels:")
    for s in range(NS):
        print(f"   hard {STUDENT_CLASSES[s]:<12} {100*hard[1,s]/max(hard[1].sum(),1):6.1f}%   "
              f"soft {100*soft[1,s]/max(soft[1].sum(),1):6.1f}%")


if __name__ == "__main__":
    main()
