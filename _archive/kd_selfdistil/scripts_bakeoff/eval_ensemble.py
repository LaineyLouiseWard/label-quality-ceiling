#!/usr/bin/env python3
"""Score the RAW uniform-softmax-mean ENSEMBLE on val (no-TTA) — the recovery-fraction denominator
for Stage-4 self-distillation (docs/MINORITY_STRATEGY_2026-06-19.md §11, Q2).

The raw ensemble (best single -> ensemble gap) is "the only valid denominator": recovery fraction
= (distilled - best_single) / (ensemble - best_single). This recomputes that ensemble number for the
SHIPPED recipe's seeds, replacing the old-pipeline job-428205 value (80.74) which must be re-derived.

Reuses evaluation/compute_metrics.py building blocks so the schema (mIoU_excluding_bg, per_class_iou)
matches the single-checkpoint metrics the readout compares against, byte-for-byte.

Run:
  PYTHONPATH=. python scripts/bakeoff/eval_ensemble.py \
    --manifest artifacts/ensemble_members.txt \
    --split val --device cuda \
    --out evaluation/evaluation_results/selfdistil/ensemble/metrics.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.compute_metrics import (
    build_model,
    load_checkpoint_into_model,
    _apply_ignore_mask,
    CLASS_NAMES_6,
)
from geoseg.datasets.biodiversity_dataset import BiodiversityValDataset
from geoseg.utils.metric import Evaluator


def read_manifest(path: Path, repo_root: Path):
    paths = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"member ckpt missing: {p}")
        paths.append(p)
    if not paths:
        raise ValueError(f"manifest {path} has no checkpoints")
    return paths


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="newline-separated member ckpt paths")
    ap.add_argument("--split", default="val", choices=["val"])  # ensemble denominator is on val
    ap.add_argument("--data-root", type=Path, default=Path("data/biodiversity_split/val"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ignore-index", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    member_paths = read_manifest(args.manifest, repo_root)
    print(f"[eval_ensemble] {len(member_paths)} members, no-TTA mean softmax, split={args.split}")

    models = []
    for p in member_paths:
        m = build_model(num_classes=6).to(device)
        m = load_checkpoint_into_model(m, p, device)
        m.eval()
        models.append(m)

    ds = BiodiversityValDataset(data_root=str(args.data_root))
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False,
                        pin_memory=(device.type == "cuda"))

    evaluator = Evaluator(num_class=6, ignore_index=args.ignore_index)
    softmax = nn.Softmax(dim=1)

    for batch in tqdm(loader, desc="ensemble", leave=False):
        images = batch["img"].to(device)
        masks = batch["gt_semantic_seg"].cpu().numpy()
        prob = None
        for m in models:
            p = softmax(m(images))
            prob = p if prob is None else prob + p
        preds = (prob / len(models)).argmax(dim=1).cpu().numpy()
        for true, pred in zip(masks, preds):
            t_flat, p_flat = _apply_ignore_mask(true, pred, args.ignore_index)
            evaluator.add_batch(t_flat, p_flat)

    iou = evaluator.Intersection_over_Union()
    f1 = evaluator.F1()
    metrics = {
        "checkpoint": f"ENSEMBLE[{len(models)}]",
        "members": [str(p) for p in member_paths],
        "split": args.split,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "ignore_index": args.ignore_index,
        "tta": False,
        "OA": float(evaluator.OA()),
        "mIoU_excluding_bg": float(np.nanmean(iou[1:])),
        "mF1_excluding_bg": float(np.nanmean(f1[1:])),
        "per_class_iou": {CLASS_NAMES_6[i]: float(iou[i]) for i in range(6)},
        "per_class_f1": {CLASS_NAMES_6[i]: float(f1[i]) for i in range(6)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"[eval_ensemble] mIoU(excl-bg)={metrics['mIoU_excluding_bg']*100:.2f}  "
          f"Settlement={metrics['per_class_iou']['Settlement']*100:.2f}  "
          f"Semi-nat={metrics['per_class_iou']['Seminatural Grassland']*100:.2f}  -> {args.out}")


if __name__ == "__main__":
    main()
