#!/usr/bin/env python3
"""
Analyze class distribution in training masks to identify which images to replicate.

Offline dataset analysis tool.
Produces a JSON list consumed by replicate_minority_samples.py.

Conventions:
- Class 0 (Background) is excluded from percentage calculations.
- Percentages are computed over non-background pixels only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# Cosmetic labels only
CLASSES: Dict[int, str] = {
    0: "Background",
    1: "Forest",
    2: "Grassland",
    3: "Cropland",
    4: "Settlement",
    5: "SemiNatural",
}

# Minority classes to target
TARGET_CLASSES = [4, 5]  # Settlement, SemiNatural
IGNORE_INDEX = 0


def analyze_mask(mask_path: Path) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Return class pixel percentages and counts for a single mask.
    Percentages are computed over non-background pixels only.
    """
    mask = np.array(Image.open(mask_path))

    valid = mask != IGNORE_INDEX
    valid_pixels = int(valid.sum())

    unique, counts = np.unique(mask[valid], return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}

    class_percentages: Dict[int, float] = {}
    for cls_id in range(6):
        if cls_id == IGNORE_INDEX or valid_pixels == 0:
            class_percentages[cls_id] = 0.0
        else:
            class_percentages[cls_id] = (class_counts.get(cls_id, 0) / valid_pixels) * 100.0

    return class_percentages, class_counts


def analyze_dataset(data_root: str, mask_dir_name: str = "masks") -> List[dict]:
    """Analyze all masks in a dataset root."""
    root = Path(data_root)
    mask_dir = root / mask_dir_name
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    results: List[dict] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        img_id = mask_path.stem
        percentages, counts = analyze_mask(mask_path)

        target_score = float(sum(percentages[c] for c in TARGET_CLASSES))

        results.append(
            {
                "img_id": img_id,
                "mask_path": str(mask_path),
                "class_percentages": percentages,
                "class_counts": counts,
                "target_score": target_score,
                "settlement_pct": float(percentages[4]),
                "seminatural_pct": float(percentages[5]),
            }
        )

    results.sort(key=lambda x: x["target_score"], reverse=True)
    return results


def print_top_images(results: List[dict], n: int = 20) -> None:
    """Print top N images with most target class presence."""
    n = min(n, len(results))
    print(f"\nTop {n} images with highest Settlement + SemiNatural presence:")
    print("=" * 100)
    print(f"{'Image ID':<30} {'Settlement %':>12} {'SemiNatural %':>15} {'Total %':>10}")
    print("=" * 100)

    for item in results[:n]:
        print(
            f"{item['img_id']:<30} {item['settlement_pct']:>11.2f}% "
            f"{item['seminatural_pct']:>14.2f}% {item['target_score']:>9.2f}%"
        )


def save_augmentation_list(
    results: List[dict],
    output_file: str,
    threshold_settlement: float,
    threshold_seminatural: float,
    overwrite: bool,
) -> dict:
    """Save JSON list of image IDs meeting target thresholds."""
    settlement_images = [
        item["img_id"] for item in results if item["settlement_pct"] >= threshold_settlement
    ]
    seminatural_images = [
        item["img_id"] for item in results if item["seminatural_pct"] >= threshold_seminatural
    ]

    augmentation_list = {
        "settlement_images": settlement_images,
        "seminatural_images": seminatural_images,
        "settlement_count": len(settlement_images),
        "seminatural_count": len(seminatural_images),
        "thresholds": {
            "settlement": threshold_settlement,
            "seminatural": threshold_seminatural,
        },
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Use --overwrite to regenerate."
        )

    out_path.write_text(json.dumps(augmentation_list, indent=2))

    print(f"\nSaved augmentation list to: {out_path.resolve()}")
    print(f"  - Settlement images (≥{threshold_settlement}%): {len(settlement_images)}")
    print(f"  - SemiNatural images (≥{threshold_seminatural}%): {len(seminatural_images)}")

    return augmentation_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze masks for target-class presence (offline)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/biodiversity_split/train",
        help="Dataset root containing images/ and masks/",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="masks",
        help="Mask subdirectory name",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Print top N images")
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/train_augmentation_list.json",
        help="Output JSON file",
    )
    parser.add_argument("--threshold-settlement", type=float, default=5.0)
    parser.add_argument("--threshold-seminatural", type=float, default=5.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON if it already exists.",
    )
    args = parser.parse_args()

    print(f"Analyzing dataset: {args.data_root}/{args.mask_dir}")
    results = analyze_dataset(args.data_root, mask_dir_name=args.mask_dir)
    print(f"Total images analyzed: {len(results)}")

    print_top_images(results, n=args.top_n)

    save_augmentation_list(
        results,
        args.out,
        threshold_settlement=args.threshold_settlement,
        threshold_seminatural=args.threshold_seminatural,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
