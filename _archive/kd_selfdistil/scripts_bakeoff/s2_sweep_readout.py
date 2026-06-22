#!/usr/bin/env python3
"""S2 readout — self-distillation T x alpha ROBUSTNESS grid (docs/SWEEP_PLAN_2026-06-21.md).

Reads the 9 per-cell val (no-TTA) metrics.json produced by sonic/8_submit_selfdistil_sweep.slurm
and reports whether the recipe is STABLE across the grid (the only claim this experiment makes).

*** This does NOT pick a winning (T,alpha). *** Single-seed cell maxima are mostly noise (winner's
curse, Cawley & Talbot 2010). The shipped (T,alpha) stays the pre-registered default (2, 0.5);
KEEP/DROP is decided elsewhere (Arm A vs the no-KD control). A cell only counts as "different" from
the centre if its gap exceeds the ~2 sigma seed band (default 0.34 pp); otherwise: within noise = robust.

Run:  PYTHONPATH=. python scripts/bakeoff/s2_sweep_readout.py \
        --dir evaluation/evaluation_results/selfdistil/sweep [--band 0.34]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SETTLE = "Settlement"
SEMINAT = "Seminatural Grassland"
TS = [1, 2, 5]
ALPHAS = [0.3, 0.5, 0.7]
CENTER = (2, 0.5)


def load_cells(d: Path):
    cells = {}
    for mj in sorted(d.glob("stage4_selfdistil_T*_a*/metrics.json")):
        m = re.search(r"_T(\d+)_a(\d+)", mj.parent.name)
        if not m:
            continue
        t = int(m.group(1))
        a = int(m.group(2)) / 10.0
        j = json.load(open(mj))
        assert j.get("tta") is False, f"{mj}: must be no-TTA, got tta={j.get('tta')}"
        cells[(t, a)] = j
    return cells


def grid(cells, get, title, center_val):
    print(f"\n{title}  (rows = alpha, cols = T; Δ vs centre {CENTER} = {center_val:.2f})")
    hdr = "  alpha\\T " + "".join(f"{t:>10}" for t in TS)
    print(hdr)
    for a in ALPHAS:
        row = f"  {a:>6} "
        for t in TS:
            if (t, a) in cells:
                v = get(cells[(t, a)])
                d = v - center_val
                mark = "*" if (t, a) == CENTER else " "
                row += f" {v:6.2f}{mark}{d:+5.2f}"
            else:
                row += f" {'--':>11}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("evaluation/evaluation_results/selfdistil/sweep"))
    ap.add_argument("--band", type=float, default=0.34, help="~2 sigma seed band; |Δ|<band = within noise")
    args = ap.parse_args()

    cells = load_cells(args.dir)
    if CENTER not in cells:
        present = sorted(cells)
        print(f"Centre cell {CENTER} (the default = Arm A) not found in {args.dir}.")
        print(f"  cells present: {present if present else 'none yet — fetch the sweep results first'}")
        return

    def mi(j): return j["mIoU_excluding_bg"] * 100
    def se(j): return j["per_class_iou"][SETTLE] * 100
    def sm(j): return j["per_class_iou"][SEMINAT] * 100

    c_mi, c_se, c_sm = mi(cells[CENTER]), se(cells[CENTER]), sm(cells[CENTER])

    print("=" * 92)
    print(f"S2 SELF-DISTILLATION ROBUSTNESS GRID — {len(cells)}/9 cells (val, no-TTA, single seed)")
    print("=" * 92)
    grid(cells, mi, "mIoU (excl-bg)", c_mi)
    grid(cells, se, "Settlement IoU", c_se)
    grid(cells, sm, "Semi-natural IoU", c_sm)

    # robustness verdict on mIoU
    outside = [(t, a, mi(cells[(t, a)]) - c_mi) for (t, a) in cells
               if (t, a) != CENTER and abs(mi(cells[(t, a)]) - c_mi) > args.band]
    print("\n" + "-" * 92)
    print("VERDICT (robustness only — NOT a recipe selection)")
    if not outside:
        print(f"  ROBUST: every off-centre cell is within ±{args.band:.2f} pp (~2σ) of the default mIoU.")
        print(f"  -> report 'self-distillation is stable across T∈{{1,2,5}}, α∈{{0.3,0.7}}'.")
    else:
        print(f"  {len(outside)} cell(s) exceed the ±{args.band:.2f} pp band (possible sensitivity/interaction):")
        for t, a, d in sorted(outside, key=lambda x: -abs(x[2])):
            print(f"    T={t}, α={a}: {d:+.2f} pp vs default")
        print("  -> report the dependence honestly; do NOT swap the shipped (T=2,α=0.5) for a cell.")
    print("\n  Shipped (T,α) stays the pre-registered DEFAULT (2, 0.5). KEEP/DROP is Arm A vs no-KD control,")
    print("  not this grid. (docs/SWEEP_PLAN_2026-06-21.md)")


if __name__ == "__main__":
    main()
