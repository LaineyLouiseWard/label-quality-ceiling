#!/usr/bin/env python3
"""
Compare the no-replication test arm against the current (replication) pipeline, focused on
the minority classes. Reads the per-class IoU written by evaluation/compute_metrics.py.

Pairs compared (val split):
  stage4_norep   vs  stage4_sampling
  stage5_norep   vs  stage5_kd

Run AFTER both arms are trained + evaluated:
  PYTHONPATH=. python scripts/analysis/compare_norep.py
"""
from __future__ import annotations

import json
from pathlib import Path

MINORITY = ["Settlement", "Seminatural Grassland"]


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "evaluation").exists():
            return p
    return start.parents[2]


def load(root: Path, split: str, stage: str):
    for base in [
        root / "evaluation/evaluation_results/norep_test" / split / stage / "metrics.json",
        root / "evaluation/evaluation_results" / split / stage / "metrics.json",
    ]:
        if base.exists():
            return json.loads(base.read_text())
    return None


def row(label, m):
    if not m:
        return f"  {label:<16} (missing)"
    iou = m["per_class_iou"]
    mi = m.get("mIoU_excluding_bg")
    cells = "  ".join(f"{k.split()[0]:>11}={iou.get(k, float('nan')):.3f}" for k in MINORITY)
    return f"  {label:<16} mIoU={mi:.3f}   {cells}"


def main():
    root = find_repo_root(Path(__file__).resolve())
    print("No-replication test vs current pipeline (val split)")
    print("=" * 72)
    for norep, current in [("stage4_norep", "stage4_sampling"), ("stage5_norep", "stage5_kd")]:
        mn = load(root, "val", norep)
        mc = load(root, "val", current)
        print(f"\n[{norep}  vs  {current}]")
        print(row("CURRENT", mc))
        print(row("NO-REP", mn))
        if mn and mc:
            for cls in MINORITY:
                d = mn["per_class_iou"].get(cls, float("nan")) - mc["per_class_iou"].get(cls, float("nan"))
                arrow = "↓" if d < 0 else "↑"
                print(f"    Δ {cls:<22} {arrow} {d:+.3f}")
    print("\nDecision rule: if NO-REP minority IoU ≈ CURRENT (within seed noise), replication is")
    print("redundant -> drop it. If NO-REP is clearly lower, the exposure mattered (recover via")
    print("x2 minority weights -- see compute_replication_exposure.py s_match=2.0 -- not physical replication).")


if __name__ == "__main__":
    main()
