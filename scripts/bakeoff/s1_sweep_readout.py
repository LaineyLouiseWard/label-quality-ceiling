#!/usr/bin/env python3
"""S1 readout — clsbal SAMPLER robustness sweep (docs/SWEEP_PLAN_2026-06-21.md).

Reads the per-cell val (no-TTA) metrics from sonic/9_submit_s1_sampler_sweep.slurm and reports whether
the MINORITY IoUs (Settlement, Semi-natural — the point of the sampler) are STABLE across q and
settlement_target. Reference = the shipped clsbal (q=1.0, settlement_target=1.27).

*** Robustness only. *** Shipped sampler stays q=1.0 / target=1.27; a cell counts as "different" only if
its gap to the reference exceeds the ~2σ seed band (default 0.34 pp). Single-seed max is noise; do not
swap the shipped sampler on this basis.

Run:  PYTHONPATH=. python scripts/bakeoff/s1_sweep_readout.py \
        --dir evaluation/evaluation_results/s1_sampler_sweep [--band 0.34]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SETTLE = "Settlement"
SEMINAT = "Seminatural Grassland"
REF = "_q10_st127"   # shipped clsbal


def load(d: Path):
    cells = {}
    for mj in sorted(d.glob("stage3_clsbal_q*_st*/metrics.json")):
        tag = re.search(r"(_q\d+_st\d+)", mj.parent.name)
        if not tag:
            continue
        j = json.load(open(mj))
        assert j.get("tta") is False, f"{mj}: must be no-TTA"
        cells[tag.group(1)] = j
    return cells


def label(tag):
    m = re.match(r"_q(\d+)_st(\d+)", tag)
    q = int(m.group(1)) / 10.0
    st = int(m.group(2)) / 100.0
    return f"q={q}, target={st}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("evaluation/evaluation_results/s1_sampler_sweep"))
    ap.add_argument("--band", type=float, default=0.34)
    args = ap.parse_args()

    cells = load(args.dir)
    if REF not in cells:
        print(f"Reference cell {REF} (shipped clsbal) not in {args.dir}.")
        print(f"  cells present: {sorted(cells) or 'none yet — fetch S1 results first'}")
        return

    def mi(j): return j["mIoU_excluding_bg"] * 100
    def se(j): return j["per_class_iou"][SETTLE] * 100
    def sm(j): return j["per_class_iou"][SEMINAT] * 100

    r = cells[REF]
    print("=" * 88)
    print(f"S1 SAMPLER ROBUSTNESS — {len(cells)} cells (val, no-TTA, single seed). Reference = {label(REF)}")
    print("=" * 88)
    hdr = f"{'cell':>16} {'Settle':>8} {'ΔSet':>7} {'Semi':>8} {'ΔSemi':>7} {'mIoU':>8} {'ΔmIoU':>7}"
    print(hdr); print("-" * len(hdr))
    outside = []
    for tag in sorted(cells):
        c = cells[tag]
        dse, dsm, dmi = se(c) - se(r), sm(c) - sm(r), mi(c) - mi(r)
        mark = "  <ref" if tag == REF else ""
        print(f"{label(tag):>16} {se(c):8.2f} {dse:+7.2f} {sm(c):8.2f} {dsm:+7.2f} {mi(c):8.2f} {dmi:+7.2f}{mark}")
        if tag != REF and (abs(dse) > args.band or abs(dsm) > args.band):
            outside.append((label(tag), dse, dsm))

    print("\n" + "-" * 88)
    print("VERDICT (robustness only — NOT sampler selection)")
    if not outside:
        print(f"  ROBUST: every cell's Settlement & Semi-nat IoU is within ±{args.band:.2f} pp of the shipped sampler.")
        print(f"  -> report 'minority IoU stable across q∈{{0.5,1.0}} and settlement_target∈{{1.0,1.27,1.5}}'.")
    else:
        print(f"  {len(outside)} cell(s) move a minority IoU beyond ±{args.band:.2f} pp:")
        for lab, dse, dsm in outside:
            print(f"    {lab}: ΔSettle {dse:+.2f}, ΔSemi {dsm:+.2f}")
        print("  -> report the dependence honestly; the shipped sampler stays q=1.0 / target=1.27.")


if __name__ == "__main__":
    main()
