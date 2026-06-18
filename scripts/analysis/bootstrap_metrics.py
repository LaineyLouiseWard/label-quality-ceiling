#!/usr/bin/env python3
"""
Tile-level bootstrap confidence intervals for all ablation stages (1, 2, 3b, 4, 5).

Runs inference on val and test splits, collects per-tile confusion matrices,
then bootstraps (tile-level resampling) to produce 95% CIs for mIoU, mF1, OA.

Usage:
    python scripts/analysis/bootstrap_metrics.py [--device cuda|cpu] [--n-boot 2000]

Outputs:
    analysis/bootstrap_results.md   — markdown summary
    analysis/per_tile_cms/          — cached per-tile confusion matrices (.npz)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── repo root ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.utils import find_repo_root

REPO = find_repo_root()

from evaluation.compute_metrics import (
    build_model,
    load_checkpoint_into_model,
    _apply_ignore_mask,
)
from geoseg.datasets.biodiversity_dataset import (
    BiodiversityValDataset,
    BiodiversityTestWithMasksDataset,
)

# ── constants ───────────────────────────────────────────────────────────────
NUM_CLASSES = 6
IGNORE_INDEX = 0
CLASS_NAMES_5 = ["Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]

STAGES = {
    "stage1_baseline": REPO / "model_weights" / "biodiversity" / "stage1_baseline" / "stage1_baseline.ckpt",
    "stage2b_oem_finetune": REPO / "model_weights" / "biodiversity" / "stage2b_oem_finetune" / "stage2b_oem_finetune.ckpt",
    "stage3_sampler": REPO / "model_weights" / "biodiversity" / "stage3_sampler" / "stage3_sampler.ckpt",
    "stage4_kd": REPO / "model_weights" / "biodiversity" / "stage4_kd" / "stage4_kd.ckpt",
}


# ── per-tile inference ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_per_tile_cms(
    ckpt_path: Path,
    split: str,
    device: torch.device,
) -> list[np.ndarray]:
    """Return a list of (6,6) int64 confusion matrices, one per tile."""
    model = build_model(num_classes=NUM_CLASSES).to(device)
    model = load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    data_root = str(REPO / "data" / "biodiversity_split" / split)
    if split == "val":
        ds = BiodiversityValDataset(data_root=data_root)
    else:
        ds = BiodiversityTestWithMasksDataset(data_root=data_root)

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    softmax = nn.Softmax(dim=1)
    tile_cms = []

    for batch in tqdm(loader, desc=f"{ckpt_path.stem}/{split}", leave=False):
        img = batch["img"].to(device)
        mask = batch["gt_semantic_seg"].cpu().numpy()[0]  # (H,W)
        pred = softmax(model(img)).argmax(dim=1).cpu().numpy()[0]  # (H,W)

        t, p = _apply_ignore_mask(mask, pred, IGNORE_INDEX)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for gt_val, pr_val in zip(t, p):
            cm[gt_val, pr_val] += 1
        tile_cms.append(cm)

    return tile_cms


# ── metrics from confusion matrix ──────────────────────────────────────────

def metrics_from_cm(cm: np.ndarray) -> dict:
    """Compute mIoU, mF1, OA from a (6,6) confusion matrix (bg-aware)."""
    eps = 1e-8
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    oa = float(tp.sum() / (cm.sum() + eps))
    miou = float(np.nanmean(iou[1:]))  # foreground only
    mf1 = float(np.nanmean(f1[1:]))
    per_class_iou = {CLASS_NAMES_5[i]: float(iou[i + 1]) for i in range(5)}

    return {"mIoU": miou, "mF1": mf1, "OA": oa, "per_class_iou": per_class_iou}


# ── bootstrap ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    tile_cms: list[np.ndarray],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Tile-level bootstrap: resample tiles, aggregate CMs, compute metrics."""
    rng = np.random.default_rng(seed)
    n = len(tile_cms)
    cms = np.stack(tile_cms)  # (N, 6, 6)

    boot_miou, boot_mf1, boot_oa = [], [], []
    boot_per_class = {c: [] for c in CLASS_NAMES_5}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        agg_cm = cms[idx].sum(axis=0)
        m = metrics_from_cm(agg_cm)
        boot_miou.append(m["mIoU"])
        boot_mf1.append(m["mF1"])
        boot_oa.append(m["OA"])
        for c in CLASS_NAMES_5:
            boot_per_class[c].append(m["per_class_iou"][c])

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    # Point estimate from full aggregate
    full_cm = cms.sum(axis=0)
    point = metrics_from_cm(full_cm)

    result = {
        "n_tiles": n,
        "n_boot": n_boot,
        "point": point,
        "ci_95": {
            "mIoU": [float(np.percentile(boot_miou, lo)), float(np.percentile(boot_miou, hi))],
            "mF1": [float(np.percentile(boot_mf1, lo)), float(np.percentile(boot_mf1, hi))],
            "OA": [float(np.percentile(boot_oa, lo)), float(np.percentile(boot_oa, hi))],
            "per_class_iou": {
                c: [float(np.percentile(boot_per_class[c], lo)),
                    float(np.percentile(boot_per_class[c], hi))]
                for c in CLASS_NAMES_5
            },
        },
    }
    return result


# ── caching ────────────────────────────────────────────────────────────────

def cache_path(stage: str, split: str) -> Path:
    return REPO / "analysis" / "per_tile_cms" / f"{stage}_{split}.npz"


def save_cms(tile_cms: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, *tile_cms)


def load_cms(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for all ablation stages (1, 2, 3b, 4, 5)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cache exists")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = {}

    for stage, ckpt in STAGES.items():
        for split in ("val", "test"):
            key = f"{stage}/{split}"
            cp = cache_path(stage, split)

            if cp.exists() and not args.force:
                print(f"[cache] Loading {cp}")
                tile_cms = load_cms(cp)
            else:
                print(f"[infer] {key} on {device}")
                tile_cms = collect_per_tile_cms(ckpt, split, device)
                save_cms(tile_cms, cp)
                print(f"[cache] Saved {len(tile_cms)} tile CMs → {cp}")

            result = bootstrap_ci(tile_cms, n_boot=args.n_boot)
            all_results[key] = result
            p = result["point"]
            ci = result["ci_95"]
            print(f"  {key}: mIoU={p['mIoU']:.1%} [{ci['mIoU'][0]:.1%}, {ci['mIoU'][1]:.1%}]")

    # ── write markdown report ──────────────────────────────────────────────
    out_dir = REPO / "analysis"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / "bootstrap_results.md"

    lines = [
        "# Bootstrap Confidence Intervals (Tile-Level Resampling)",
        "",
        f"Resamples: {args.n_boot} | Seed: 42 | Level: 95%",
        "",
    ]

    for key, r in all_results.items():
        p = r["point"]
        ci = r["ci_95"]
        lines.append(f"## {key}")
        lines.append(f"- Tiles: {r['n_tiles']}")
        lines.append(f"- **mIoU**: {p['mIoU']:.1%}  [{ci['mIoU'][0]:.1%}, {ci['mIoU'][1]:.1%}]")
        lines.append(f"- **mF1**:  {p['mF1']:.1%}  [{ci['mF1'][0]:.1%}, {ci['mF1'][1]:.1%}]")
        lines.append(f"- **OA**:   {p['OA']:.1%}  [{ci['OA'][0]:.1%}, {ci['OA'][1]:.1%}]")
        lines.append("")
        lines.append("| Class | IoU | 95% CI |")
        lines.append("|-------|-----|--------|")
        for c in CLASS_NAMES_5:
            iou_val = p["per_class_iou"][c]
            lo, hi = ci["per_class_iou"][c]
            lines.append(f"| {c} | {iou_val:.1%} | [{lo:.1%}, {hi:.1%}] |")
        lines.append("")

    # ── Stage 1→4 delta CI ──────────────────────────────────────────────────
    for split in ("val", "test"):
        s1 = all_results.get(f"stage1_baseline/{split}")
        s5 = all_results.get(f"stage4_kd/{split}")
        if s1 and s5:
            lines.append(f"## Stage 1→4 improvement ({split})")
            for metric in ("mIoU", "mF1", "OA"):
                delta = s5["point"][metric] - s1["point"][metric]
                # Delta of values rounded to 1 dp -- the convention used in the manuscript
                # tables, where per-class/mean deltas are differences of the displayed rounded
                # cells. This can differ from the raw delta by up to ~0.1 pp due to endpoint
                # rounding (e.g. test mIoU: raw +10.5 vs rounded-endpoint +10.6).
                delta_rounded = round(s5["point"][metric] * 100, 1) - round(s1["point"][metric] * 100, 1)
                # Width of individual CIs as proxy for uncertainty
                w1 = s1["ci_95"][metric][1] - s1["ci_95"][metric][0]
                w5 = s5["ci_95"][metric][1] - s5["ci_95"][metric][0]
                lines.append(f"- **Δ{metric}**: +{delta:.1%} raw (+{delta_rounded:.1f} pp from rounded endpoints, as reported in the manuscript)  (individual CI widths: ±{w1/2:.1%}, ±{w5/2:.1%})")
            lines.append("")

    md_path.write_text("\n".join(lines))
    print(f"\nResults written to {md_path}")

    # Also save raw JSON
    json_path = out_dir / "bootstrap_results.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"Raw JSON written to {json_path}")


if __name__ == "__main__":
    main()
