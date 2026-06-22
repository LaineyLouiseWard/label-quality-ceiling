#!/usr/bin/env python3
"""
§9 Gate G1 (course-corrected) — does the candidate sampler realise the intended oversampling?
Shows realised per-class ratios for the standard class-balanced sampler vs A0, TWO ways:
  (1) a3 uplift = mean(w | class present) / mean(w | absent), over the augmentation list (the way
      A0's documented 1.27x / 3.08x was measured), and
  (2) realised SAMPLING exposure from an actual draw of the real WeightedRandomSampler
      (num_samples=2646): E[fraction of draws on a class-c tile] / base fraction.

Decision geometry (course-correction §15): Settlement is EXPECTED ~flat (~1.27x, A0 level) — it is
tile-ubiquitous and the sampler can't move it; the lift comes from copy-paste. Semi-natural should
be clearly oversampled (~2x) but need not match A0's 3.08x (that came from pixel-area concentration
we deliberately dropped). The SHIP decision is made later on val IoU, not on these ratios.

Run:  PYTHONPATH=. python scripts/bakeoff/gate_G1_sampler.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import WeightedRandomSampler

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from utils import load_augmentation_list  # noqa: E402

TRAIN_ROOT = REPO / "data" / "biodiversity_split" / "train"
CLSBAL_TSV = REPO / "artifacts" / "sampler_weights_clsbal.tsv"
A0_TSV = REPO / "artifacts" / "sampler_weights.tsv"
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def _norm_id(x):
    if "_rep" in x:
        b, r = x.rsplit("_rep", 1)
        if r.isdigit():
            return b
    return x


def load_single_col(tsv, ids):
    d = {}
    with open(tsv) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            i, w = ln.split("\t")
            d[i] = float(w)
    return [d.get(_norm_id(i), 1.0) for i in ids]


def main():
    img_ids = sorted(osp.splitext(f)[0] for f in os.listdir(TRAIN_ROOT / "images") if f.endswith(".tif"))
    N = len(img_ids)
    # any-pixel presence
    pres = {4: np.zeros(N, bool), 5: np.zeros(N, bool)}
    for k, i in enumerate(img_ids):
        m = np.array(Image.open(TRAIN_ROOT / "masks" / (i + ".png")).convert("L"))
        pres[4][k] = (m == 4).any()
        pres[5][k] = (m == 5).any()
    aug = load_augmentation_list()
    aug_pos = {4: set(aug["settlement_images"]), 5: set(aug["seminatural_images"])}
    aug_col = {c: np.array([img_ids[k] in aug_pos[c] for k in range(N)]) for c in (4, 5)}

    w_a0 = np.array(load_single_col(A0_TSV, img_ids))
    w_cb = np.array(load_single_col(CLSBAL_TSV, img_ids))

    def uplift(w, col):
        return (w[col].mean()) / (w[~col].mean())

    print("G1 — realised oversampling (method 1: a3 uplift over augmentation list)")
    print(f"    {'sampler':10s} {'Settlement':>11} {'Seminat':>9}")
    print(f"    {'A0 pooled':10s} {uplift(w_a0, aug_col[4]):11.3f} {uplift(w_a0, aug_col[5]):9.3f}")
    cb_set, cb_sn = uplift(w_cb, aug_col[4]), uplift(w_cb, aug_col[5])
    print(f"    {'clsbal':10s} {cb_set:11.3f} {cb_sn:9.3f}")

    # method 2: actual sampler draw (200k draws from the real WeightedRandomSampler)
    g = torch.Generator().manual_seed(42)
    draws = np.array(list(WeightedRandomSampler(
        weights=w_cb.tolist(), num_samples=200000, replacement=True, generator=g)))
    # exposure ratio = P(draw hits a class-c tile) / base fraction of class-c tiles
    print("\nG1 — realised oversampling (method 2: 200k draws from the actual WeightedRandomSampler)")
    for c in (4, 5):
        p_sample = pres[c][draws].mean()
        p_base = pres[c].mean()
        name = {4: "Settlement", 5: "Seminat"}[c]
        print(f"    {name:10s} exposure {p_sample/p_base:.3f}x   (base {p_base:.3f} -> sampled {p_sample:.3f})")

    # asserts
    check("Settlement held ~flat (a3 uplift <= 1.4, near A0's 1.27)", cb_set <= 1.4, f"{cb_set:.3f}x")
    check("Seminat clearly oversampled (a3 uplift >= 1.6)", cb_sn >= 1.6, f"{cb_sn:.3f}x")
    check("Seminat oversampled MORE than Settlement", cb_sn > cb_set, f"{cb_sn:.3f} > {cb_set:.3f}")
    check("A0 frozen TSV present (never overwritten)", A0_TSV.exists())

    print("\n" + "=" * 70)
    if FAIL:
        print(f"G1 RESULT: {len(FAIL)} FAILED -> {FAIL}")
        sys.exit(1)
    print("G1 RESULT: PASS — Settlement ~flat (lift rides on copy-paste), Seminat oversampled ~2x.")
    print("Ship decision is by Semi-nat val IoU vs A0, decided AFTER training.")
    sys.exit(0)


if __name__ == "__main__":
    main()
