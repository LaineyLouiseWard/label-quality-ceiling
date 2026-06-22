#!/usr/bin/env python3
"""
§9 Gate G6 — copy-paste keeps image and label PIXEL-LOCKED (highest-severity trap: the exact shape
of the prior class-mapping bug). Dumps >=20 composited (image, mask) pairs to PNG with the label
contour overlaid on the RGB, and asserts the label is never blended/interpolated.

Run from repo root:  PYTHONPATH=. python scripts/bakeoff/gate_G6_copypaste.py
Writes PNGs to evaluation/bakeoff_gates/G6/ for the human to eyeball before any GPU time.

NOTE on the handoff's literal assertion `np.array_equal(out_lbl==4, alpha)`: Settlement is present
in 64% of destination tiles, so out_lbl==4 = (dst's pre-existing Settlement) UNION (pasted region),
which cannot equal the donor blob alpha when dst already contains Settlement. We therefore assert the
EXACT superseding relation that DOES hold universally and captures the true intent
(image/label pixel-locked, label never blended):
    (out_lbl == 4)            == ((dst_lbl == 4) | paste)
    out_lbl[~paste]           == dst_lbl[~paste]
    out_img[paste]            == src_img[paste]
plus: unique(out_lbl) subset {0..5}; no donor label other than 4 introduced; label bit-identical
before/after the image-only photometric jitter.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import albumentations as albu

from geoseg.taxonomy import STUDENT_PALETTE, STUDENT_CLASSES
from geoseg.datasets.biodiversity_dataset import (
    SETTLEMENT_INDEX,
    _read_tif_as_rgb_uint8,
    _compose_settlement,
    _donor_ids,
    _donor_pool,
    _load_donor,
    train_aug_random,
    configure_settlement_copypaste,
)

OUT = REPO / "evaluation" / "bakeoff_gates" / "G6"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_ROOT = REPO / "data" / "biodiversity_split" / "train"
N_DUMP = 24
FAILURES: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _colour(mask):
    pal = np.array(STUDENT_PALETTE, dtype=np.uint8)
    return pal[np.clip(mask, 0, 5)]


def _contour_overlay(rgb, mask, target=SETTLEMENT_INDEX):
    """Red contour where mask==target sits on the RGB, so the eye can check the blob edge."""
    out = rgb.copy()
    m = (mask == target).astype(np.uint8)
    # 4-neighbour boundary
    edge = np.zeros_like(m)
    edge[1:, :] |= m[1:, :] ^ m[:-1, :]
    edge[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    out[edge.astype(bool)] = [255, 0, 0]
    return out


def main():
    print("=" * 78)
    print("§9 GATE G6 — copy-paste image/label pixel-lock")
    print("=" * 78)

    img_dir, mask_dir = TRAIN_ROOT / "images", TRAIN_ROOT / "masks"
    all_ids = sorted(osp.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".tif"))
    # Primary arm uses uniform-random Settlement donors (confidence-gating parked, §15.4). Integrity
    # is donor-agnostic; set targeted=True here + _donor_pool to also exercise the parked refinement.
    donor_ids = _donor_ids(str(TRAIN_ROOT), SETTLEMENT_INDEX)
    donor_p = None
    print(f"donor pool (Settlement-bearing tiles, uniform-random = primary arm): {len(donor_ids)}")
    rng = np.random.RandomState(42)

    n_with_dst_settlement = 0
    jitter = albu.RandomBrightnessContrast(0.25, 0.25, p=1.0)  # image-only, applied at p=1 to test

    for k in range(N_DUMP):
        # destination tile (any train tile)
        did = all_ids[rng.randint(len(all_ids))]
        dst_img = np.array(_read_tif_as_rgb_uint8(str(img_dir / (did + ".tif"))).resize((512, 512), Image.BICUBIC))
        dst_mask = np.array(Image.open(mask_dir / (did + ".png")).convert("L").resize((512, 512), Image.NEAREST))
        dst_mask[dst_mask == 255] = 0
        # donor — confidence-weighted (targeted) selection
        sid = donor_ids[int(rng.choice(len(donor_ids), p=donor_p))]
        src_img, src_mask = _load_donor(str(TRAIN_ROOT), sid, dst_mask.shape, flip=True)

        out_img, out_mask, paste, alpha = _compose_settlement(dst_img, dst_mask, src_img, src_mask)

        had_settle = bool((dst_mask == 4).any())
        n_with_dst_settlement += int(had_settle)

        # ---- asserts (per composite) ----
        u = set(np.unique(out_mask).tolist())
        check(f"[{k:02d}] unique(out_mask) subset {{0..5}}", u.issubset({0, 1, 2, 3, 4, 5}), str(sorted(u)))
        check(f"[{k:02d}] (out==4) == ((dst==4)|paste)",
              np.array_equal(out_mask == 4, (dst_mask == 4) | paste))
        check(f"[{k:02d}] out_mask[~paste] == dst_mask[~paste] (rest untouched)",
              np.array_equal(out_mask[~paste], dst_mask[~paste]))
        check(f"[{k:02d}] out_img[paste] == src_img[paste] (pixel-locked)",
              np.array_equal(out_img[paste], src_img[paste]))
        # donor introduced no label other than Settlement(4)
        introduced = out_mask[paste]
        check(f"[{k:02d}] donor introduced only label 4", bool((introduced == 4).all()))
        # label bit-identical before/after IMAGE-ONLY photometric jitter
        jittered = jitter(image=out_img, mask=out_mask)
        check(f"[{k:02d}] label bit-identical after photometric jitter",
              np.array_equal(jittered["mask"], out_mask))

        # ---- dump PNG: dst | composite+contour | mask(colour) ----
        panel = np.concatenate([
            _contour_overlay(dst_img, dst_mask),
            _contour_overlay(out_img, out_mask),
            _colour(out_mask),
        ], axis=1)
        Image.fromarray(panel).save(OUT / f"g6_{k:02d}_dst-{did}_donor-{sid}.png")

    print(f"\ndumped {N_DUMP} panels to {OUT}")
    print(f"  ({n_with_dst_settlement}/{N_DUMP} destinations already contained Settlement — "
          "exactly why the literal `out==4 == alpha` form is wrong and the relation form is used)")

    # Also exercise the FULL pipeline path (train_aug_random with copy-paste configured) once,
    # to confirm the flag wiring composes end-to-end and labels stay in-range.
    configure_settlement_copypaste(enabled=True, donor_root=str(TRAIN_ROOT), prob=1.0, n_donors=1)
    did = all_ids[0]
    pim = _read_tif_as_rgb_uint8(str(img_dir / (did + ".tif")))
    pmk = Image.open(mask_dir / (did + ".png")).convert("L")
    img_t, mask_t = train_aug_random(pim, pmk)
    configure_settlement_copypaste(enabled=False)
    check("train_aug_random (copy-paste ON) returns in-range labels",
          set(np.unique(mask_t).tolist()).issubset({0, 1, 2, 3, 4, 5}),
          f"labels={sorted(set(np.unique(mask_t).tolist()))}")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"G6 RESULT: {len(FAILURES)} CHECK(S) FAILED -> BLOCK")
        sys.exit(1)
    print(f"G6 RESULT: ALL {N_DUMP} COMPOSITES PIXEL-LOCKED — eyeball the PNGs in {OUT}")
    sys.exit(0)


if __name__ == "__main__":
    main()
