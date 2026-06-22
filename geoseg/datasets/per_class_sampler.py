"""
Lever 2 (per-class sampler) — config-side combine of the per-class raw tile weights.

The pooled Stage-3 sampler routes ~69% of the oversampling budget to Seminatural and leaves
Settlement at ~1.27x (docs/STAGE4_KD_AND_SAMPLER_DIAGNOSIS.md §3). This splits the pooled
richness into two independent per-class raw weights w4 (Settlement) and w5 (Seminatural),
written by scripts/data_prep/build_sampler_weights.py --per_class, then combines them HERE so the
Settlement boost s4 is a tunable config knob without rebuilding the TSV.

Combine recipe (handoff §3 Lever 2, step 2 — combine THEN clip+mix ONCE; do NOT clip/mix each
term separately and sum, which floors every tile near 1.0 and swamps the per-class signal):
    w_raw  = s4 * w4 + s5 * w5                 (default s4 = s5 = 1)
    w_clip = clip(w_raw, p_lo, p_hi)           (5th-95th pct over all tiles)
    w_norm = w_clip / mean(w_clip)
    w_tile = (1 - alpha) + alpha * w_norm       (alpha = 0.5, uniform mix)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PER_CLASS_HEADER = ["img_id", "w4", "w5"]


def _norm_id(x: str) -> str:
    """Strip _repN suffix so replicas share the same base weight (matches the builder)."""
    if "_rep" in x:
        base, rep = x.rsplit("_rep", 1)
        if rep.isdigit():
            return base
    return x


def load_per_class_tile_weights(
    tsv_path,
    img_ids,
    *,
    s4: float = 1.0,
    s5: float = 1.0,
    alpha_mix: float = 0.5,
    clip_lo: float = 5.0,
    clip_hi: float = 95.0,
    eps: float = 1e-6,
):
    """Return (weights_aligned_to_img_ids, missing_count, id_to_wtile).

    weights_aligned_to_img_ids is the per-tile scalar for WeightedRandomSampler, in the order of
    img_ids. Tiles with no entry get weight 1.0 (and are counted in missing_count; the caller
    should refuse to train if missing > 0).
    """
    tsv_path = Path(tsv_path)
    keys: list[str] = []
    w4: list[float] = []
    w5: list[float] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == PER_CLASS_HEADER, (
            f"Per-class TSV header must be {PER_CLASS_HEADER}, got {header} in {tsv_path}"
        )
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            iid, a, b = line.split("\t")
            keys.append(iid)
            w4.append(float(a))
            w5.append(float(b))

    w4 = np.asarray(w4)
    w5 = np.asarray(w5)
    w_raw = s4 * w4 + s5 * w5
    lo = np.percentile(w_raw, clip_lo)
    hi = np.percentile(w_raw, clip_hi)
    w_clip = np.clip(w_raw, lo, hi)
    w_norm = w_clip / (w_clip.mean() + eps)
    w_tile = (1.0 - alpha_mix) + alpha_mix * w_norm

    id_to_w = {k: float(w) for k, w in zip(keys, w_tile)}

    weights: list[float] = []
    missing = 0
    for iid in img_ids:
        w = id_to_w.get(_norm_id(iid))
        if w is None:
            weights.append(1.0)
            missing += 1
        else:
            weights.append(w)
    return weights, missing, id_to_w
