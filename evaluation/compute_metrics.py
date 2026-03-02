#!/usr/bin/env python3
"""
Evaluate FT-UNetFormer checkpoints on the Biodiversity validation or test split.

Writes per-checkpoint outputs:
- metrics.json (OA, per-class IoU/F1, macro means excluding Background)
- confusion_matrix.png (row-normalized; ignores ignore_index pixels)
- class_iou_scores.png / class_f1_scores.png (excluding Background)
- evaluation_report.txt

Conventions:
- Biodiversity uses ignore_index=0 (Background) during evaluation.
- Macro metrics are reported excluding Background (class 0).
- Test evaluation reads masks ONLY to compute metrics (no training).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.metric import Evaluator


CLASS_NAMES_6 = [
    "Background",
    "Forest land",
    "Grassland",
    "Cropland",
    "Settlement",
    "Seminatural Grassland",
]
CLASS_NAMES_5 = CLASS_NAMES_6[1:]


def build_model(num_classes: int = 6) -> torch.nn.Module:
    """Instantiate the FT-UNetFormer architecture used in this repo."""
    return ft_unetformer(num_classes=num_classes, decoder_channels=256)

def load_checkpoint_into_model(
    model: torch.nn.Module, ckpt_path: Path, device: torch.device
) -> torch.nn.Module:
    """Load a Lightning .ckpt into the raw student nn.Module.

    KD checkpoints may also include teacher.* keys; we intentionally ignore them.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Prefer student keys stored under "net.*"
    net_sd = {k.replace("net.", "", 1): v for k, v in state_dict.items() if k.startswith("net.")}

    # Fallback: some checkpoints might store weights directly without net.*
    if not net_sd:
        # strip optional "model." prefix if present
        net_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                net_sd[k.replace("model.", "", 1)] = v
            else:
                net_sd[k] = v

    missing, unexpected = model.load_state_dict(net_sd, strict=False)

    if missing:
        logging.warning(
            f"Missing keys (non-fatal): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        logging.warning(
            f"Unexpected keys (non-fatal): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )

    # IMPORTANT: do NOT crash just because KD checkpoints contain extra modules.
    # If you still want a safety check, only crash if *most* keys are missing.
    if len(net_sd) < 100:
        raise RuntimeError(f"Checkpoint seems to contain no student weights (net.*): {ckpt_path}")

    return model



def _apply_ignore_mask(
    true: np.ndarray, pred: np.ndarray, ignore_index: int | None
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten arrays and drop ignored pixels (for confusion matrix consistency)."""
    t = true.reshape(-1)
    p = pred.reshape(-1)
    if ignore_index is None:
        return t, p
    keep = t != ignore_index
    return t[keep], p[keep]


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    data_root: Path,
    split: str,
    device: torch.device,
    ignore_index: int | None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[dict, np.ndarray]:
    model = build_model(num_classes=6).to(device)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    if split == "val":
        ds = BiodiversityValDataset(data_root=str(data_root))
    elif split == "test":
        ds = BiodiversityTestWithMasksDataset(data_root=str(data_root))
    else:
        raise ValueError(f"Unknown split: {split}")

    loader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = Evaluator(num_class=6, ignore_index=ignore_index)
    cm = np.zeros((6, 6), dtype=np.int64)

    softmax = nn.Softmax(dim=1)

    for batch in tqdm(loader, desc=f"Evaluating {ckpt_path.name}", leave=False):
        images = batch["img"].to(device)
        masks = batch["gt_semantic_seg"].cpu().numpy()  # (B,H,W)

        outputs = model(images)
        preds = softmax(outputs).argmax(dim=1).cpu().numpy()  # (B,H,W)

        for true, pred in zip(masks, preds):
            t_flat, p_flat = _apply_ignore_mask(true, pred, ignore_index)
            cm += confusion_matrix(t_flat, p_flat, labels=list(range(6)))
            evaluator.add_batch(t_flat, p_flat)

    iou_all = evaluator.Intersection_over_Union()
    f1_all = evaluator.F1()
    oa = float(evaluator.OA())

    # Macro metrics excluding Background (class 0)
    iou_no_bg = iou_all[1:]
    f1_no_bg = f1_all[1:]
    miou = float(np.nanmean(iou_no_bg))
    mf1 = float(np.nanmean(f1_no_bg))

    metrics = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "data_root": str(data_root),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "ignore_index": ignore_index,
        "OA": oa,
        "mIoU_excluding_bg": miou,
        "mF1_excluding_bg": mf1,
        "per_class_iou": {CLASS_NAMES_6[i]: float(iou_all[i]) for i in range(6)},
        "per_class_f1": {CLASS_NAMES_6[i]: float(f1_all[i]) for i in range(6)},
    }
    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_pct = cm / row_sums

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_pct, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(6), CLASS_NAMES_6, rotation=45, ha="right")
    plt.yticks(range(6), CLASS_NAMES_6)

    for i in range(6):
        for j in range(6):
            plt.text(j, i, f"{cm_pct[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_bars(values: list[float], labels: list[str], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(labels)), values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate .ckpt checkpoints and write metrics + plots")
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Which split to evaluate.")
    ap.add_argument(
        "--base-dir",
        type=str,
        default="model_weights",
        help="Directory to search for .ckpt files (recursively).",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/val",
        help="Split root (contains images/ and masks/). Use val or test root depending on --split.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="evaluation/evaluation_results",
        help="Where to write evaluation outputs.",
    )
    ap.add_argument("--pattern", type=str, default="*.ckpt", help="Checkpoint filename pattern.")
    ap.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument(
        "--ignore-index",
        type=int,
        default=0,
        help="Label value to ignore in evaluation (biodiversity: 0).",
    )
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing evaluation outputs without prompting.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpts = sorted(base_dir.rglob(args.pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {base_dir} with pattern {args.pattern}")

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Found {len(ckpts)} checkpoints under {base_dir}")
    logging.info(f"Split={args.split}  data_root={data_root}")
    logging.info(f"Using ignore_index={args.ignore_index}")

    for ckpt in ckpts:
        safe_name = ckpt.parent.name # Should only be keeping one checkpoint per ablation stage.
        run_dir = out_root / safe_name
        run_dir.mkdir(parents=True, exist_ok=True)

        existing_metrics = run_dir / "metrics.json"
        if existing_metrics.exists() and not args.force:
            logging.warning(
                f"Skipping {safe_name}: {existing_metrics} already exists. "
                "Pass --force to overwrite."
            )
            continue

        logging.info(f"Evaluating: {ckpt}")

        metrics, cm = evaluate_checkpoint(
            ckpt_path=ckpt,
            data_root=data_root,
            split=args.split,
            device=device,
            ignore_index=args.ignore_index,
            batch_size=1,
            num_workers=args.num_workers,
        )

        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        plot_confusion_matrix(cm, run_dir)
        np.save(run_dir / "confusion_matrix.npy", cm)
        np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

        iou_no_bg = [metrics["per_class_iou"][c] for c in CLASS_NAMES_5]
        f1_no_bg = [metrics["per_class_f1"][c] for c in CLASS_NAMES_5]

        plot_class_bars(
            iou_no_bg,
            CLASS_NAMES_5,
            title=f"Per-Class IoU (excl. Background)\n{ckpt.name}",
            ylabel="IoU",
            out_path=run_dir / "class_iou_scores.png",
        )
        plot_class_bars(
            f1_no_bg,
            CLASS_NAMES_5,
            title=f"Per-Class F1 (excl. Background)\n{ckpt.name}",
            ylabel="F1",
            out_path=run_dir / "class_f1_scores.png",
        )

        with open(run_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
            f.write("=== Evaluation Report ===\n\n")
            f.write(f"Checkpoint: {metrics['checkpoint']}\n")
            f.write(f"Split: {metrics['split']}\n")
            f.write(f"Data root: {metrics['data_root']}\n")
            f.write(f"Ignore index: {metrics['ignore_index']}\n\n")
            f.write(f"Overall Accuracy (OA): {metrics['OA']:.4f}\n")
            f.write(f"Mean IoU (excl. bg): {metrics['mIoU_excluding_bg']:.4f}\n")
            f.write(f"Mean F1 (excl. bg): {metrics['mF1_excluding_bg']:.4f}\n\n")
            f.write("Per-class IoU:\n")
            for c in CLASS_NAMES_6:
                f.write(f"  {c}: {metrics['per_class_iou'][c]:.4f}\n")

    logging.info(f"Done. Outputs written to {out_root}")


if __name__ == "__main__":
    main()
