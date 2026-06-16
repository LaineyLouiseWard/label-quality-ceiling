#!/usr/bin/env python3
"""
Compute the minority-class SAMPLING EXPOSURE that Stage-4's replication+sampler produces,
and the exposure of a no-replication "sampler-as-is" arm, from artifacts already on disk
(no GPU, no model). Reproducible companion to the replication-redundancy test.

Why this exists
---------------
The Stage-4 WeightedRandomSampler uses `replacement=True` and the weights TSV is keyed by
BASE tile id (replicas share their base weight). Under replacement sampling, a minority tile
duplicated x(1+r) with weight w has the SAME draw-mass (1+r)*w as a single copy with weight
(1+r)*w. So static replication is a special case of weight-scaling. This script quantifies:
  - P_min^R : minority draw-probability of the CURRENT arm (train_rep, replicas present)
  - P_min^N : minority draw-probability of the NO-REP arm (train, sampler-as-is)
  - s_match : the minority weight multiplier that makes the no-rep arm reproduce P_min^R
              (equals 1+r when all minority tiles share one replica count -> proves equivalence)

Inputs (all already in the repo):
  artifacts/stage4_sampling_weights.tsv          (base_id -> weight)
  data/biodiversity_split/train_rep/images/      (*_repN.tif reveal the replicated/minority set)
  data/biodiversity_split/train_rep/replication_log.json

Output:
  artifacts/replication_exposure_report.json     (+ printed summary)

Run:
  PYTHONPATH=. python scripts/analysis/compute_replication_exposure.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "artifacts").exists() and (p / "data").exists():
            return p
    return start.parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="artifacts/stage4_sampling_weights.tsv")
    ap.add_argument("--train_rep", default="data/biodiversity_split/train_rep/images")
    ap.add_argument("--out", default="artifacts/replication_exposure_report.json")
    args = ap.parse_args()

    root = find_repo_root(Path(__file__).resolve())
    tsv = root / args.tsv
    rep_dir = root / args.train_rep

    # --- weights (base_id -> weight) ---
    id_to_w: dict[str, float] = {}
    for line in tsv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        i, w = line.split("\t")
        id_to_w[i] = float(w)

    # --- minority set M and per-tile replica counts, read straight from train_rep ---
    rep_re = re.compile(r"^(?P<base>.+)_rep(?P<n>\d+)\.(tif|tiff|png)$", re.IGNORECASE)
    replica_counts: Counter[str] = Counter()
    for f in rep_dir.iterdir():
        m = rep_re.match(f.name)
        if m:
            replica_counts[m.group("base")] += 1
    minority = sorted(replica_counts)  # base ids that were replicated == minority-rich set

    missing = [m for m in minority if m not in id_to_w]
    if missing:
        raise SystemExit(f"{len(missing)} minority ids absent from TSV (id mismatch): {missing[:5]}")

    all_w = sum(id_to_w.values())                                   # one copy of every base tile
    min_w = sum(id_to_w[m] for m in minority)                       # minority base mass (x1)
    # current arm: minority tile appears (1 + r_m) times, all at weight w_m
    rep_extra = sum(id_to_w[m] * replica_counts[m] for m in minority)  # the EXTRA replica mass
    denom_R = all_w + rep_extra
    num_R = sum(id_to_w[m] * (1 + replica_counts[m]) for m in minority)

    P_min_R = num_R / denom_R          # current (replication + sampler)
    P_min_N = min_w / all_w            # no-rep, sampler as-is

    # multiplier on minority weights that makes the no-rep arm match P_min_R exactly:
    #   s*min_w / (all_w - min_w + s*min_w) = P_min_R   ->  solve for s
    nonmin_w = all_w - min_w
    s_match = (P_min_R * nonmin_w) / (min_w * (1 - P_min_R))

    rc = sorted(set(replica_counts.values()))
    report = {
        "n_base_tiles": len(id_to_w),
        "n_minority_tiles": len(minority),
        "replica_counts_present": rc,
        "uniform_replica_count": rc[0] if len(rc) == 1 else None,
        "sum_weight_all": round(all_w, 6),
        "sum_weight_minority": round(min_w, 6),
        "P_min_current_replication_plus_sampler": round(P_min_R, 6),
        "P_min_norep_sampler_as_is": round(P_min_N, 6),
        "exposure_ratio_R_over_N": round(P_min_R / P_min_N, 6),
        "minority_weight_multiplier_to_match (s_match)": round(s_match, 6),
        "note": (
            "replacement=True => replication is weight-scaling. s_match≈(1+r) confirms a FULLY "
            "exposure-matched + step-matched no-rep arm is provably ≈ the current arm (a null). "
            "The informative test is the NO-REP, SAMPLER-AS-IS arm at P_min_N (this lower exposure) "
            "— judged on Settlement/Semi-natural IoU vs the current arm."
        ),
    }

    out = root / args.out
    out.write_text(json.dumps(report, indent=2))

    print("Replication exposure report")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
