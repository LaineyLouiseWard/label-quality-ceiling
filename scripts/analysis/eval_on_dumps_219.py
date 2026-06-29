#!/usr/bin/env python
"""
Canonical evaluation on the 219-tile Irish val set, recomputed from the per-seed softmax
dumps -- the single source of truth for per-class IoU and the 6x6 confusion.

WHY THIS EXISTS. The Sonic eval artefacts (`final_results/*.json`, `seed*/val/<cell>/
confusion_matrix.npy`) were scored on 231 tiles, which include 12 foreign (`col1_`/`den*_`)
tiles that have no Irish ground-truth mask. Recomputing here from the dumps via the shared
`list_val_tiles` join (dumps ∩ Irish masks = 219) removes that contamination. Per-seed argmax
of the dumped softmax reproduces non-TTA `compute_metrics` exactly on the same tiles, so the
per-class IoU here equals the (contamination-free) reported numbers.

SCOPE. Only the cells that have softmax dumps can be recomputed: **stage1_baseline** and
**stage3_clsbal**. The other two factorial cells (stage2b_oem_finetune, stage_sampler_only)
were not dumped, so their clean recompute waits on the ADE20K run (job 452469). NB factorial
*effects* are cell differences, so the shared 12-tile bias largely cancels; only absolute
per-class IoU shifts (~1 pp on Cropland/Semi-natural).

OUTPUTS (analysis/eval_219/):
  per_class_iou.json        per-cell: per-seed-mean +/- SD per-class IoU + mIoU(fg), n_seeds, n_tiles
  confusion_<cell>.npy      summed 6x6 count confusion (rows=GT, cols=pred) over seeds (for Fig 8)

Usage:
  PYTHONPATH=. python scripts/analysis/eval_on_dumps_219.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.seed_disagreement import (  # noqa: E402
    STUDENT_CLASSES, C, list_val_tiles, load_mask, seed_dir,
)

FOREGROUND = list(range(1, C))   # 1..5


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """6x6 count confusion (rows=GT, cols=pred) over all pixels."""
    idx = C * gt.astype(np.int64).ravel() + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=C * C).reshape(C, C)


def iou_from_conf(conf: np.ndarray) -> dict:
    """Per-foreground-class IoU + macro from a (GT,pred) confusion. Background (0) excluded,
    matching compute_metrics' nanmean(iou[1:]) (ignore_index=0)."""
    out, ious = {}, []
    for k in FOREGROUND:
        tp = float(conf[k, k])
        fp = float(conf[:, k].sum() - conf[k, k])
        fn = float(conf[k, :].sum() - conf[k, k])
        denom = tp + fp + fn
        v = tp / denom if denom > 0 else float("nan")
        out[STUDENT_CLASSES[k]] = v
        ious.append(v)
    out["mIoU_fg"] = float(np.nanmean(ious))
    return out


def run_cell(softmax_root, mask_dir, cell, seeds, out_dir):
    img_ids, dropped = list_val_tiles(softmax_root, seeds, cell, mask_dir)
    if dropped:
        print(f"[{cell}] {len(img_ids)} Irish val tiles; dropped {len(dropped)} non-Irish dump tiles")

    masks = {iid: load_mask(mask_dir, iid) for iid in img_ids}
    conf_sum = np.zeros((C, C), dtype=np.int64)            # summed over seeds (for Fig 8)
    per_seed_iou = {STUDENT_CLASSES[k]: [] for k in FOREGROUND}
    per_seed_miou = []
    for s in seeds:
        d = seed_dir(softmax_root, s, cell)
        conf_s = np.zeros((C, C), dtype=np.int64)
        for iid in img_ids:
            pred = np.load(d / f"{iid}.npy").argmax(axis=0)
            m = masks[iid]
            fg = m != 0                      # ignore_index=0: exclude Background-GT pixels
            conf_s += confusion(m[fg], pred[fg])
        conf_sum += conf_s
        iou = iou_from_conf(conf_s)
        for k in FOREGROUND:
            per_seed_iou[STUDENT_CLASSES[k]].append(iou[STUDENT_CLASSES[k]])
        per_seed_miou.append(iou["mIoU_fg"])

    summary = {
        "cell": cell, "n_seeds": len(seeds), "n_tiles": len(img_ids),
        "mIoU_fg_mean": float(np.nanmean(per_seed_miou)),
        "mIoU_fg_std": float(np.nanstd(per_seed_miou, ddof=1)),
        "per_class_iou_mean": {c: float(np.nanmean(v)) for c, v in per_seed_iou.items()},
        "per_class_iou_std": {c: float(np.nanstd(v, ddof=1)) for c, v in per_seed_iou.items()},
        "per_seed_iou": per_seed_iou,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"confusion_{cell}.npy", conf_sum)
    print(f"  mIoU(fg)={summary['mIoU_fg_mean']:.4f}+-{summary['mIoU_fg_std']:.4f}  "
          + " ".join(f"{c[:4]}={summary['per_class_iou_mean'][c]:.3f}"
                     for c in summary["per_class_iou_mean"]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softmax-root", default="sonic/results")
    ap.add_argument("--mask-dir", default="data/biodiversity_split/val/masks")
    ap.add_argument("--cell", action="append", dest="cells", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--out-dir", default="analysis/eval_219")
    args = ap.parse_args()
    cells = args.cells or ["stage1_baseline", "stage3_clsbal"]
    out_dir = Path(args.out_dir)
    result = {}
    for cell in cells:
        result[cell] = run_cell(args.softmax_root, args.mask_dir, cell, args.seeds, out_dir)
    (out_dir / "per_class_iou.json").write_text(json.dumps(result, indent=2))
    print(f"[eval_on_dumps_219] wrote per_class_iou.json + confusion_*.npy to {out_dir}")


if __name__ == "__main__":
    main()
