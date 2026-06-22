#!/usr/bin/env python3
"""
Stage-3 bake-off READOUT — triage table + ladder verdict, run once the no-TTA metrics are fetched
back from Sonic (sonic/4b_fetch_bakeoff.sh) into evaluation/evaluation_results/val_bakeoff/.

Read-out logic (course-correction §15):
  Sampler ships = clsbal if Semi-nat IoU TIES A0; else richonly if it ties; else A0.
  Copy-paste    = does Settlement IoU beat A0?  (the headline)
"Tie" is judged paired at seed 42 (the arms' seed) with the 5-seed-mean as the stability bar.

Run:  PYTHONPATH=. python scripts/bakeoff/readout.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAL = REPO / "evaluation" / "evaluation_results" / "val_bakeoff"
CSV = REPO / "analysis" / "seed_aggregate" / "per_seed_metrics.csv"

# G5 alias map: CSV name -> compute_metrics.py JSON name
ALIAS = {"Settlement": "Settlement", "Semi-natural": "Seminatural Grassland",
         "Forest": "Forest land", "Grassland": "Grassland", "Cropland": "Cropland"}
MAJ = ["Forest", "Grassland", "Cropland"]
ARMS = ["stage3_clsbal", "stage3_richonly", "stage3_copypaste"]

# seed-42 std (5-seed) for a rough tie band, from the anchors
TIE_BAND = None  # filled from A0's per-seed spread below


def anchor(rows, stage, cls):
    v = {int(r["seed"]): float(r["value_pct"]) for r in rows
         if r["split"] == "val" and r["stage_folder"] == stage and r["metric"] == "IoU" and r["class"] == cls}
    return v


def main():
    rows = list(csv.DictReader(open(CSV)))

    def anc(stage, cls):
        v = anchor(rows, stage, cls)
        return v.get(42, float("nan")), (st.mean(v.values()) if v else float("nan")), \
               (st.pstdev(v.values()) if len(v) > 1 else float("nan"))

    a0 = {c: anc("stage3_sampler", c) for c in ["Settlement", "Semi-natural", "Forest", "Grassland", "Cropland"]}
    n0 = {c: anc("stage3null_nosampler", c) for c in ["Settlement", "Semi-natural"]}
    a0_sn_s42, a0_sn_mean, a0_sn_std = a0["Semi-natural"]
    a0_set_s42, a0_set_mean, _ = a0["Settlement"]
    band = max(0.5, 2 * a0_sn_std)  # ~2σ tie band on Semi-nat (min 0.5pp)

    print("=" * 92)
    print("STAGE-3 BAKE-OFF READOUT (seed 42, no-TTA) vs A0/N0 anchors")
    print("=" * 92)
    print(f"Anchors (s42 / 5-mean):  A0 Settlement {a0_set_s42:.2f}/{a0_set_mean:.2f}  "
          f"Semi-nat {a0_sn_s42:.2f}/{a0_sn_mean:.2f}   N0 Semi-nat {n0['Semi-natural'][1]:.2f}   "
          f"(tie band ±{band:.2f})")
    print()
    hdr = f"{'arm':18s} {'Settle':>7} {'ΔA0':>7} | {'Semi-nat':>9} {'ΔA0':>7} | {'maj-IoU':>8} | {'mIoU':>7} | tta"
    print(hdr); print("-" * len(hdr))

    results = {}
    for arm in ARMS:
        mj = VAL / arm / "metrics.json"
        if not mj.exists():
            print(f"{arm:18s}   (no metrics.json yet — fetch from Sonic)")
            continue
        m = json.load(open(mj))
        iou = m["per_class_iou"]
        settle = iou[ALIAS["Settlement"]] * 100
        semin = iou[ALIAS["Semi-natural"]] * 100
        maj = st.mean(iou[ALIAS[c]] for c in MAJ) * 100
        miou = m["mIoU_excluding_bg"] * 100
        results[arm] = dict(settle=settle, semin=semin, maj=maj, miou=miou, tta=m["tta"])
        print(f"{arm:18s} {settle:7.2f} {settle - a0_set_s42:+7.2f} | {semin:9.2f} {semin - a0_sn_s42:+7.2f} | "
              f"{maj:8.2f} | {miou:7.2f} | {m['tta']}")

    print("\n" + "-" * 92)
    print("VERDICT")
    # sampler ladder
    ship = "A0 (frozen, fallback)"
    for arm, label in [("stage3_clsbal", "clsbal (freq-only, most defensible)"),
                       ("stage3_richonly", "richonly (A0 minus hardness)")]:
        if arm in results:
            d = results[arm]["semin"] - a0_sn_s42
            ties = d >= -band
            print(f"  sampler {arm:16s}: Semi-nat {results[arm]['semin']:.2f} ({d:+.2f} vs A0 s42) "
                  f"-> {'TIES A0' if ties else 'regresses'}")
            if ties:
                ship = label
                break
    print(f"  => SHIP SAMPLER: {ship}")
    # copy-paste
    if "stage3_copypaste" in results:
        d = results["stage3_copypaste"]["settle"] - a0_set_s42
        print(f"  copy-paste: Settlement {results['stage3_copypaste']['settle']:.2f} ({d:+.2f} vs A0 s42) "
              f"-> {'LIFTS Settlement' if d > 0 else 'no lift'} (headline)")
    print("\nNote: seed-42 single-seed — confirm the winning recipe (shipped sampler + copy-paste) at multi-seed.")


if __name__ == "__main__":
    main()
