#!/usr/bin/env python3
"""Stage-4 SELF-DISTILLATION readout — paired significance + recovery fraction + keep/drop verdict.

Implements the §11 decision rule (docs/MINORITY_STRATEGY_2026-06-19.md):
  KEEP Stage 4 iff the distilled student beats the best single seed by >=0.5 pp mIoU,
  PAIRED across seeds with 95% CI excluding 0, AND benchmarked vs the raw ensemble
  (recovery fraction = (distilled - best_single) / (ensemble - best_single)). Else ship a
  clean 3-stage pipeline — do NOT dress up a wash.

Reads the no-TTA val metrics.json produced for:
  - the N ensemble MEMBERS (= best-single candidates: the shipped Stage-3 recipe per seed)
  - the N DISTILLED students (stage4_selfdistil per seed)
  - the RAW ENSEMBLE (scripts/bakeoff/eval_ensemble.py) — the recovery denominator
Pairs distilled_k with member_k by the seed parsed from the path.

Run:
  PYTHONPATH=. python scripts/bakeoff/selfdistil_readout.py \
    --single  'evaluation/evaluation_results/selfdistil/single/seed*/metrics.json' \
    --distilled 'evaluation/evaluation_results/selfdistil/distilled/seed*/metrics.json' \
    --ensemble  evaluation/evaluation_results/selfdistil/ensemble/metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

SETTLE = "Settlement"
SEMINAT = "Seminatural Grassland"

# two-sided t critical values @95% by dof (fallback 1.96 for large/unknown dof)
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
         15: 2.131, 19: 2.093, 29: 2.045}


def tcrit(dof):
    if dof in TCRIT:
        return TCRIT[dof]
    return min((TCRIT[k] for k in TCRIT if k >= dof), default=1.96)


def seed_of(path):
    m = re.search(r"seed[_-]?(\d+)", str(path))
    return int(m.group(1)) if m else None


def load_glob(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        s = seed_of(f)
        m = json.load(open(f))
        assert m.get("tta") is False, f"{f}: metrics must be no-TTA (tta=false), got {m.get('tta')}"
        out[s if s is not None else f] = m
    return out


def miou(m):
    return m["mIoU_excluding_bg"] * 100


def cls(m, name):
    return m["per_class_iou"][name] * 100


def ci95(diffs):
    n = len(diffs)
    mean = st.mean(diffs)
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = st.stdev(diffs)
    half = tcrit(n - 1) * sd / (n ** 0.5)
    return mean, mean - half, mean + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="glob for member/best-single per-seed metrics.json")
    ap.add_argument("--distilled", required=True, help="glob for distilled per-seed metrics.json")
    ap.add_argument("--control", default=None,
                    help="glob for the STEP-MATCHED no-KD control per-seed metrics.json (the shipped "
                         "Stage-3 recipe + the SAME extra epochs, KD off). REQUIRED for a valid KEEP "
                         "verdict: distilled-vs-Stage-3 is epoch-contaminated (45 extra epochs alone "
                         "buy ~+1.2 mIoU), so the binding test is distilled-vs-control. See §9.1.")
    ap.add_argument("--ensemble", required=True, type=Path, help="raw-ensemble metrics.json")
    ap.add_argument("--seed-std", type=float, default=0.17, help="per-seed mIoU wobble (campaign=0.17)")
    args = ap.parse_args()

    single = load_glob(args.single)
    distilled = load_glob(args.distilled)
    control = load_glob(args.control) if args.control else {}
    if not single or not distilled:
        print("Missing metrics:")
        print(f"  single   ({args.single}): {len(single)} found")
        print(f"  distilled({args.distilled}): {len(distilled)} found")
        print("  -> run the N-seed members + distilled students first, then re-run.")
        return

    ens = json.load(open(args.ensemble)) if args.ensemble.exists() else None
    best_single = max(miou(m) for m in single.values())
    ens_miou = miou(ens) if ens else float("nan")
    gap = (ens_miou - best_single) if ens else float("nan")

    print("=" * 96)
    print("STAGE-4 SELF-DISTILLATION READOUT (val, no-TTA)")
    print("=" * 96)
    print(f"members(best-single) seeds: {sorted(k for k in single)} | best single mIoU = {best_single:.2f}")
    if ens:
        print(f"raw ensemble mIoU = {ens_miou:.2f}  (gap over best single = {gap:+.2f} pp; "
              f"Settlement {cls(ens,SETTLE)-max(cls(m,SETTLE) for m in single.values()):+.2f}, "
              f"Semi-nat {cls(ens,SEMINAT)-max(cls(m,SEMINAT) for m in single.values()):+.2f})")
    else:
        print("raw ensemble metrics.json MISSING — recovery fraction unavailable (run eval_ensemble.py).")
    print()

    # paired distilled_k - single_k
    common = sorted(set(single) & set(distilled), key=lambda x: (str(type(x)), x))
    if not common:
        print("No common seeds between single and distilled — cannot pair. Check path seed tags.")
        return
    hdr = f"{'seed':>6} {'single':>8} {'distil':>8} {'Δpair':>7} | {'set Δ':>7} | {'semi Δ':>7}"
    print(hdr); print("-" * len(hdr))
    d_mi, d_set, d_sem = [], [], []
    for k in common:
        s, d = single[k], distilled[k]
        dm = miou(d) - miou(s)
        ds = cls(d, SETTLE) - cls(s, SETTLE)
        de = cls(d, SEMINAT) - cls(s, SEMINAT)
        d_mi.append(dm); d_set.append(ds); d_sem.append(de)
        print(f"{str(k):>6} {miou(s):8.2f} {miou(d):8.2f} {dm:+7.2f} | {ds:+7.2f} | {de:+7.2f}")

    mean_d, lo, hi = ci95(d_mi)
    mean_distilled = st.mean(miou(distilled[k]) for k in common)
    vs_best = mean_distilled - best_single
    print("-" * len(hdr))
    print(f"paired Δ mIoU (distilled - same-seed single): {mean_d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (n={len(d_mi)})")
    print(f"mean distilled mIoU = {mean_distilled:.2f}  -> vs best single = {vs_best:+.2f} pp")
    print(f"per-class paired Δ: Settlement {st.mean(d_set):+.2f}, Semi-nat {st.mean(d_sem):+.2f}")
    if ens and gap > 0:
        rec = vs_best / gap
        print(f"RECOVERY FRACTION = (distilled - best_single)/(ensemble - best_single) = {rec*100:.0f}%  "
              f"[keep-bar ≈42% (+0.5pp); Hinton-strong ≈80%]")

    # ---- BINDING TEST: distilled vs the STEP-MATCHED no-KD control (isolates distillation) ----
    ctrl_common = sorted(set(control) & set(distilled), key=lambda x: (str(type(x)), x)) if control else []
    if ctrl_common:
        d_ctrl = [miou(distilled[k]) - miou(control[k]) for k in ctrl_common]
        c_set = [cls(distilled[k], SETTLE) - cls(control[k], SETTLE) for k in ctrl_common]
        c_sem = [cls(distilled[k], SEMINAT) - cls(control[k], SEMINAT) for k in ctrl_common]
        mean_c, clo, chi = ci95(d_ctrl)
        print(f"\nBINDING Δ mIoU (distilled − step-matched no-KD control): {mean_c:+.2f}  "
              f"95% CI [{clo:+.2f}, {chi:+.2f}]  (n={len(d_ctrl)})")
        print(f"  per-class binding Δ: Settlement {st.mean(c_set):+.2f}, Semi-nat {st.mean(c_sem):+.2f}")

    print("\n" + "-" * 96)
    print("VERDICT")
    if not ctrl_common:
        print("  *** NO STEP-MATCHED no-KD CONTROL SUPPLIED (--control) — cannot declare KEEP. ***")
        print("  The distilled student trains ~45 epochs beyond the Stage-3 members; epochs alone buy")
        print("  ~+1.2 mIoU (the weak OEM-KD'd clsbal already 'cleared' the vs-Stage-3 bar). So vs-Stage-3")
        print("  is epoch-contaminated and uninterpretable as a KD test (§9.1). Run stage4null_nokd on the")
        print("  shipped recipe (Stage-3 + same epochs, KD off) and pass it as --control.")
        print(f"  [context only] distilled vs Stage-3 best-single = {vs_best:+.2f}; "
              f"paired vs same-seed Stage-3 = {mean_d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        return
    c_excludes_0 = (clo > 0) or (chi < 0)
    keep = (mean_c >= 0.5) and c_excludes_0
    ctx = f"vs Stage-3 best-single {vs_best:+.2f}"
    if ens and gap > 0:
        ctx += f"; recovery vs ensemble {vs_best / gap * 100:.0f}%"
    print(f"  binding (distilled − no-KD control): {mean_c:+.2f}  95% CI [{clo:+.2f}, {chi:+.2f}]")
    print(f"    beats control by ≥0.5 pp  : {'YES' if mean_c >= 0.5 else 'no'}")
    print(f"    paired 95% CI excludes 0  : {'YES' if c_excludes_0 else 'no'}")
    print(f"  [context] {ctx}")
    print(f"  => {'KEEP Stage 4 (self-distillation beats train-longer)' if keep else 'DROP Stage 4 — ship a clean 3-stage pipeline (do not dress up a wash)'}")
    print("\nNote: control (c) — also run eval_ensemble.py over the DISTILLED checkpoints and over an")
    print("equal count of from-scratch seeds (cho2019): distilled students may correlate and fail to ensemble.")


if __name__ == "__main__":
    main()
