#!/usr/bin/env python3
"""
Per-arm preflight for the bake-off Sonic job's FIRST STEP — exit 1 on failure so no arm trains on
corrupt inputs. Runs the gates relevant to the course-corrected 3-arm run:
  - config builds (py2cfg) -> the arm's sampler aligns (missing==0) and copy-paste configures;
  - G5 anchor-join: all 5 foreground classes resolve on both the CSV and JSON sides (scoring);
  - G6 copy-paste pixel-lock asserts (only for the copy-paste arm).

Usage:  PYTHONPATH=. python scripts/bakeoff/preflight_run.py <arm_config_stem>
        e.g. ... preflight_run.py stage3_copypaste
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

arm = sys.argv[1] if len(sys.argv) > 1 else ""
cfg_path = REPO / "config" / "biodiversity" / f"{arm}.py"
if not cfg_path.exists():
    print(f"[preflight] no such arm config: {cfg_path}")
    sys.exit(1)

fail = []

# 1) config builds (sampler missing==0 raises inside py2cfg; copy-paste configures)
from geoseg.utils.cfg import py2cfg  # noqa: E402

cfg = py2cfg(str(cfg_path))
print(f"[preflight] {arm}: config OK, loss={type(cfg.loss.first.loss).__name__}, "
      f"sampler={Path(str(cfg.weights_path_tsv)).name}, pretrained={Path(cfg.pretrained_ckpt_path).name}")
assert Path(cfg.pretrained_ckpt_path).name == "stage2b_oem_finetune.ckpt", "wrong branch point"

# 2) G5 anchor-join (so no-TTA scoring vs A0/N0 never silently drops a class)
import csv  # noqa: E402
from evaluation.compute_metrics import CLASS_NAMES_6  # noqa: E402

alias = {"Semi-natural": "Seminatural Grassland", "Forest": "Forest land",
         "Settlement": "Settlement", "Grassland": "Grassland", "Cropland": "Cropland"}
csv_classes = set()
with open(REPO / "analysis" / "seed_aggregate" / "per_seed_metrics.csv") as f:
    for r in csv.DictReader(f):
        if r["class"]:
            csv_classes.add(r["class"])
json_classes = set(CLASS_NAMES_6)
for a, b in alias.items():
    if a not in csv_classes or b not in json_classes:
        fail.append(f"G5 join: {a}->{b}")
print(f"[preflight] G5 anchor-join: {'OK' if not fail else 'FAIL ' + str(fail)}")

# 3) G6 copy-paste integrity (only for the copy-paste arm)
if "copypaste" in arm:
    r = subprocess.run([sys.executable, "scripts/bakeoff/gate_G6_copypaste.py"],
                       cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
    if r.returncode != 0:
        fail.append("G6 copy-paste integrity")
    print(f"[preflight] G6 copy-paste: {'OK' if r.returncode == 0 else 'FAIL'}")

if fail:
    print(f"[preflight] {arm} BLOCK -> {fail}")
    sys.exit(1)
print(f"[preflight] {arm} ALL GATES PASS")
sys.exit(0)
