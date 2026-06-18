"""Shared helpers for reproducible robustness analysis scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# ── Repo root detection ─────────────────────────────────────────────────────

def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for p in [start, *start.parents]:
        if (p / "artifacts").is_dir() and (p / "geoseg").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")


REPO_ROOT = find_repo_root()

# ── Canonical stage definitions ─────────────────────────────────────────────

# Paper ablation stages mapped to val evaluation-result folder paths (4-stage, no-replication).
# Stage 2a (OEM pre-train on the combined set) is omitted: it trains on OEM, not biodiversity,
# so only the Biodiversity-finetuned endpoint (Stage 2b) is reported in the ablation table.
STAGES = [
    ("1", "stage1_baseline"),
    ("2", "stage2b_oem_finetune"),
    ("3", "stage3_sampler"),
    ("4", "stage4_kd"),
]

VAL_ROOT = REPO_ROOT / "evaluation" / "evaluation_results" / "val"

# ── Class indices (0-indexed, matching confusion matrix rows/cols) ──────────

CLASS_NAMES = ["Background", "Forest", "Grassland", "Cropland", "Settlement", "Seminatural"]
IDX_BACKGROUND  = 0
IDX_FOREST      = 1
IDX_GRASSLAND   = 2
IDX_CROPLAND    = 3
IDX_SETTLEMENT  = 4
IDX_SEMINATURAL = 5

MAJORITY_INDICES = [IDX_FOREST, IDX_GRASSLAND, IDX_CROPLAND]
MINORITY_INDICES = [IDX_SETTLEMENT, IDX_SEMINATURAL]

# ── Loaders ─────────────────────────────────────────────────────────────────

def load_confusion_matrix(stage_dir: str) -> list[list[int]]:
    """Load a 6x6 confusion matrix from CSV (raw pixel counts)."""
    path = VAL_ROOT / stage_dir / "confusion_matrix.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        return [[int(x) for x in row] for row in reader]


def load_metrics(json_path: Path) -> dict:
    """Load a metrics.json file."""
    with open(json_path) as f:
        return json.load(f)


def load_val_metrics(stage_dir: str) -> dict:
    """Load val metrics.json for a given stage directory."""
    return load_metrics(VAL_ROOT / stage_dir / "metrics.json")


def load_weights_tsv(path: Path | None = None) -> dict[str, float]:
    """Load sampler_weights.tsv → {img_id: weight}."""
    path = path or REPO_ROOT / "artifacts" / "sampler_weights.tsv"
    weights = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, w = line.split("\t")
            weights[img_id] = float(w)
    return weights


def load_augmentation_list(path: Path | None = None) -> dict:
    """Load train_augmentation_list.json."""
    path = path or REPO_ROOT / "artifacts" / "train_augmentation_list.json"
    with open(path) as f:
        return json.load(f)