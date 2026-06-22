#!/usr/bin/env python3
"""
Standard class-balanced minority oversampler (Kang et al. 2020, arXiv:1910.09217) — the
defensibility arm for Stage 3 (docs/BAKEOFF_COURSE_CORRECTION_2026-06-19.md; MINORITY_STRATEGY §15).

FREQUENCY-ONLY by design — drops A0's bespoke hardness term AND pooled pixel-richness. A tile's
weight is the inverse tile-PRESENCE frequency (q=1.0, class-balanced) of the rarest MINORITY class
it contains; tiles with no minority get the uniform baseline 1.0:

    f_c   = fraction of train tiles containing >=1 pixel of class c   (any-pixel presence)
    w_I   = max( 1, scale * max_{c in {4,5} present in I} (f_c)^(-q) )      (q=1.0)

`scale` is auto-calibrated so the realised SETTLEMENT oversampling (a3 mean-with/mean-without over
the documented augmentation list, the same way A0's 1.27x was measured) ≈ A0's 1.27x — i.e.
Settlement is held FLAT at the A0 level while Semi-natural gets the strongest oversampling that
allows (~2x). The 3x A0 level is NOT a target: it came from pixel-AREA concentration that this
frequency-only sampler deliberately drops and structurally cannot reach (see §15).

NEVER overwrite artifacts/sampler_weights.tsv (A0's frozen file). Writes a new TSV.
Run:  PYTHONPATH=. python scripts/data_prep/build_clsbal_sampler.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from utils import load_augmentation_list  # noqa: E402

MINORITY = (4, 5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/biodiversity_split/train")
    p.add_argument("--out", default="artifacts/sampler_weights_clsbal.tsv")
    p.add_argument("--q", type=float, default=1.0, help="inverse-frequency exponent (1.0=class-balanced)")
    p.add_argument("--settlement_target", type=float, default=1.27,
                   help="calibrate scale so realised Settlement uplift ≈ this (A0 level)")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = REPO / args.out
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists; pass --force")
    assert out_path.name != "sampler_weights.tsv", "refusing to overwrite A0's frozen TSV"

    mask_dir = REPO / args.data_root / "masks"
    ids = sorted(osp.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith(".png"))
    N = len(ids)

    # any-pixel tile presence of the two minority classes
    present = {c: set() for c in MINORITY}
    for i in ids:
        m = np.array(Image.open(mask_dir / (i + ".png")).convert("L"))
        for c in MINORITY:
            if (m == c).any():
                present[c].add(i)
    f = {c: len(present[c]) / N for c in MINORITY}
    inv = {c: f[c] ** (-args.q) for c in MINORITY}
    print(f"any-pixel presence: " + "  ".join(
        f"class{c} {len(present[c])} (f={f[c]:.3f}, f^-{args.q:g}={inv[c]:.3f})" for c in MINORITY))

    # augmentation-list presence = the set A0's uplift was measured over (a3); use it to calibrate.
    aug = load_augmentation_list()
    aug_present = {4: set(aug["settlement_images"]), 5: set(aug["seminatural_images"])}

    def weights(scale):
        w = {}
        for i in ids:
            boosts = [scale * inv[c] for c in MINORITY if i in present[c]]
            w[i] = max([1.0] + boosts)
        return w

    def uplift(w, pos):
        ww = [w[i] for i in ids if i in pos]
        wo = [w[i] for i in ids if i not in pos]
        return (sum(ww) / len(ww)) / (sum(wo) / len(wo))

    # binary-search scale so Settlement realised uplift (over auglist) ≈ target
    lo, hi = 0.1, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        u = uplift(weights(mid), aug_present[4])
        if u < args.settlement_target:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2
    w = weights(scale)
    u_set = uplift(w, aug_present[4])
    u_sn = uplift(w, aug_present[5])
    print(f"calibrated scale={scale:.4f}  ->  realised uplift (over auglist): "
          f"Settlement {u_set:.3f}x  Seminat {u_sn:.3f}x   (A0: 1.27x / 3.08x)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for i in ids:
            fh.write(f"{i}\t{w[i]:.8f}\n")
    print(f"[OK] Wrote {N} class-balanced weights -> {out_path}")


if __name__ == "__main__":
    main()
