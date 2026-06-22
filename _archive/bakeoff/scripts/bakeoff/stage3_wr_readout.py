#!/usr/bin/env python3
"""Harmonised Stage-3 (CosineAnnealingWarmRestarts) readout — does matching the sibling stages'
scheduler, at the SAME 45-epoch budget, recover the warm-restart gain the 90-epoch control showed?

Compares three per-seed sets (val, no-TTA):
  - snapshot : analysis/clsbal_results/seed*_stage3_clsbal.json      (OLD plain single-cycle cosine)
  - harmonised: analysis/clsbal_results/seed*_stage3_clsbal_wr.json  (NEW WarmRestarts, 45 ep)
  - control  : evaluation/evaluation_results/selfdistil/control/seed*/metrics.json  (90 ep, restart)

Decision aid (NOT a hard gate):
  - If harmonised ≈ control on mIoU + Semi-nat, the restart (not the extra epochs) was the driver →
    ship harmonised at the matched 45-ep budget (cleanest: every stage same scheduler, no extra compute).
  - If harmonised sits between snapshot and control, both matter → consider the longer (90-ep) budget for
    every stage, or report harmonised as the matched-budget number and note the headroom.
"""
from __future__ import annotations
import glob, json, re, statistics as st

SETTLE, SEMINAT = "Settlement", "Seminatural Grassland"


def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"seed[_-]?(\d+)", f)
        d = json.load(open(f))
        assert d.get("tta") is False, f"{f}: must be no-TTA"
        out[int(m.group(1))] = d
    return out


def trip(d):
    p = d["per_class_iou"]
    return d["mIoU_excluding_bg"] * 100, p[SETTLE] * 100, p[SEMINAT] * 100


def main():
    # glob 'seed*_stage3_clsbal.json' does NOT match 'seed*_stage3_clsbal_wr.json' (different suffix).
    snap = load("analysis/clsbal_results/seed*_stage3_clsbal.json")
    harm = load("analysis/clsbal_results/seed*_stage3_clsbal_wr.json")
    ctrl = load("evaluation/evaluation_results/selfdistil/control/seed*/metrics.json")

    if not harm:
        print("No harmonised (_wr) metrics found yet — run sonic/8_submit_stage3_wr.slurm + 8b_fetch.")
        return

    seeds = sorted(set(harm) & set(ctrl))
    print("=" * 92)
    print("HARMONISED STAGE-3 (WarmRestarts, 45 ep) vs snapshot (plain cosine, 45 ep) vs control (90 ep)")
    print("=" * 92)
    hdr = f"{'seed':>5} | {'snap':>6} {'harm':>6} {'ctrl':>6} mIoU | {'snap':>6} {'harm':>6} {'ctrl':>6} Semi"
    print(hdr); print("-" * len(hdr))
    rows = {"snap": [], "harm": [], "ctrl": []}
    srows = {"snap": [], "harm": [], "ctrl": []}
    for s in seeds:
        sm = trip(snap[s]) if s in snap else (float("nan"),) * 3
        hm, cm = trip(harm[s]), trip(ctrl[s])
        rows["snap"].append(sm[0]); rows["harm"].append(hm[0]); rows["ctrl"].append(cm[0])
        srows["snap"].append(sm[2]); srows["harm"].append(hm[2]); srows["ctrl"].append(cm[2])
        print(f"{s:>5} | {sm[0]:6.2f} {hm[0]:6.2f} {cm[0]:6.2f}      | {sm[2]:6.2f} {hm[2]:6.2f} {cm[2]:6.2f}")
    print("-" * len(hdr))
    def mean(x): return st.mean([v for v in x if v == v])
    print(f"{'mean':>5} | {mean(rows['snap']):6.2f} {mean(rows['harm']):6.2f} {mean(rows['ctrl']):6.2f}      | "
          f"{mean(srows['snap']):6.2f} {mean(srows['harm']):6.2f} {mean(srows['ctrl']):6.2f}")
    print()
    dm = mean(rows["harm"]) - mean(rows["snap"])
    ds = mean(srows["harm"]) - mean(srows["snap"])
    gap_m = mean(rows["ctrl"]) - mean(rows["snap"])
    gap_s = mean(srows["ctrl"]) - mean(srows["snap"])
    rec_m = dm / gap_m * 100 if gap_m else float("nan")
    rec_s = ds / gap_s * 100 if gap_s else float("nan")
    print(f"harmonised − snapshot:  mIoU {dm:+.2f}  Semi-nat {ds:+.2f}")
    print(f"recovers of the control's gain:  mIoU {rec_m:.0f}%  Semi-nat {rec_s:.0f}%")
    print("  >=~80% on both → restart was the driver; ship harmonised at matched 45 ep (no extra compute).")
    print("  much less → both matter; consider the 90-ep budget for every stage, or report harmonised + headroom.")


if __name__ == "__main__":
    main()
