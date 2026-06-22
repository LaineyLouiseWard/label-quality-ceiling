#!/usr/bin/env python3
"""Visual check of the TARGETED Settlement copy-paste fix (paste_onto=(0,2,3)).

Unlike gate_G6 (which dumps random pairs and whose pixel-lock asserts pass trivially when the paste is
empty), this dumps the NEW placement rule and prints the realised paste fraction per panel, so the eye
can confirm Settlement is actually deposited at the Grassland/Cropland boundary (not a no-op).

Run:  /home/lainey/miniconda3/envs/ClassImbalance/bin/python scripts/bakeoff/verify_copypaste_visual.py
"""
from __future__ import annotations
import os, os.path as osp, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO); sys.path.insert(0, str(REPO))

from geoseg.taxonomy import STUDENT_PALETTE
from geoseg.datasets.biodiversity_dataset import (
    SETTLEMENT_INDEX, _read_tif_as_rgb_uint8, _compose_settlement, _donor_ids, _load_donor,
)

OUT = REPO / "evaluation" / "bakeoff_gates" / "G6_targeted"
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = REPO / "data" / "biodiversity_split" / "train"
PASTE_ONTO = (0, 2, 3)
N = 10


def colour(m):
    return np.array(STUDENT_PALETTE, dtype=np.uint8)[np.clip(m, 0, 5)]


def contour(rgb, mask, t=SETTLEMENT_INDEX):
    out = rgb.copy(); m = (mask == t).astype(np.uint8)
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] ^ m[:-1, :]; e[:, 1:] |= m[:, 1:] ^ m[:, :-1]
    out[e.astype(bool)] = [255, 0, 0]; return out


img_dir, mask_dir = TRAIN / "images", TRAIN / "masks"
ids = sorted(osp.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".tif"))
donors = _donor_ids(str(TRAIN), SETTLEMENT_INDEX)
rng = np.random.RandomState(7)
print(f"train={len(ids)} donors={len(donors)} paste_onto={PASTE_ONTO}")
for k in range(N):
    did = ids[rng.randint(len(ids))]
    di = np.array(_read_tif_as_rgb_uint8(str(img_dir / (did + ".tif"))).resize((512, 512), Image.BICUBIC))
    dm = np.array(Image.open(mask_dir / (did + ".png")).convert("L").resize((512, 512), Image.NEAREST))
    dm[dm == 255] = 0
    sid = donors[rng.randint(len(donors))]
    si, sm = _load_donor(str(TRAIN), sid, dm.shape, flip=True)
    oi, om, paste, alpha = _compose_settlement(di, dm, si, sm, SETTLEMENT_INDEX, PASTE_ONTO)
    frac = paste.mean() * 100
    panel = np.concatenate([contour(di, dm), contour(oi, om), colour(om)], axis=1)
    Image.fromarray(panel).save(OUT / f"t{k:02d}_paste{frac:04.1f}pc_dst-{did}_donor-{sid}.png")
    print(f"  t{k:02d} dst={did} donor={sid}  deposited={frac:.1f}% of tile")
print(f"dumped to {OUT}")
