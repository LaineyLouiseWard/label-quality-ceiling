#!/usr/bin/env python3
"""
§9 pre-flight verification gates for the Stage-3 minority bake-off (CPU set: G1, G2, G3, G5, G8).

Each gate catches a "trains-fine-but-corrupts" trap; any failure exits 1 and BLOCKS training.
Run from repo root:  PYTHONPATH=. python scripts/bakeoff/preflight_gates.py
The Sonic array job re-runs this as its first step. (G6 — copy-paste integrity — is a separate
script with PNG dumps for the human eyeball; G4/G7 are runtime checks done at score/train time.)
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from geoseg.taxonomy import STUDENT_CLASSES
from geoseg.datasets.per_class_sampler import load_per_class_tile_weights, _norm_id
from geoseg.losses import SoftCrossEntropyLoss, RecallCrossEntropyLoss

PER_CLASS_TSV = REPO / "artifacts" / "sampler_weights_perclass.tsv"
POOLED_TSV = REPO / "artifacts" / "sampler_weights.tsv"
TRAIN_ROOT = REPO / "data" / "biodiversity_split" / "train"
ANCHOR_CSV = REPO / "analysis" / "seed_aggregate" / "per_seed_metrics.csv"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _train_img_ids() -> list[str]:
    img_dir = TRAIN_ROOT / "images"
    return sorted(osp.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".tif"))


def _class_presence(img_ids):
    """Per-tile boolean presence of each class id (0..5), scanned from mask PNGs."""
    mask_dir = TRAIN_ROOT / "masks"
    pres = np.zeros((len(img_ids), 6), dtype=bool)
    for i, iid in enumerate(img_ids):
        m = np.array(Image.open(mask_dir / (iid + ".png")).convert("L"))
        u = np.unique(m)
        for c in u:
            if 0 <= c <= 5:
                pres[i, c] = True
    return pres


def _exposure_ratio(weights, present_col):
    """E[fraction of draws hitting a class-c tile] / base fraction of class-c tiles."""
    w = np.asarray(weights, dtype=np.float64)
    p_sample = (w * present_col).sum() / w.sum()
    p_base = present_col.mean()
    return p_sample / p_base


# ===========================================================================
def gate_G1_G2(img_ids, pres):
    print("\nG2 — per-class TSV parsed correctly")
    with open(PER_CLASS_TSV) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    check("header == ['img_id','w4','w5']", header == ["img_id", "w4", "w5"], str(header))
    w4 = np.array([float(r[1]) for r in rows])
    w5 = np.array([float(r[2]) for r in rows])
    check("sum(w4) > 0", w4.sum() > 0, f"sum(w4)={w4.sum():.4f}")
    check("sum(w5) > 0", w5.sum() > 0, f"sum(w5)={w5.sum():.4f}")
    print("    sample rows:")
    for r in rows[:3]:
        print("      ", r)
    check("A0 pooled TSV (artifacts/sampler_weights.tsv) still present (never overwritten)",
          POOLED_TSV.exists())
    import config.biodiversity.stage3_sampler  # noqa: F401  (its weights_path_tsv must be the pooled file)
    a0_tsv = config.biodiversity.stage3_sampler.weights_path_tsv
    check("A0 config points at the original pooled TSV",
          Path(a0_tsv).name == "sampler_weights.tsv", str(a0_tsv))

    print("\nG1 — sampler actually re-balances (realised oversampling ratio)")
    # A1 per-class weights (s4=s5=1) vs A0 pooled weights.
    w_a1, miss1, _ = load_per_class_tile_weights(PER_CLASS_TSV, img_ids, s4=1.0, s5=1.0)
    check("A1 per-class weights aligned (missing==0)", miss1 == 0, f"missing={miss1}")
    # A0 pooled
    id2w = {}
    with open(POOLED_TSV) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            iid, val = ln.split("\t")
            id2w[iid] = float(val)
    w_a0 = [id2w.get(_norm_id(i), 1.0) for i in img_ids]

    set_col, sn_col = pres[:, 4], pres[:, 5]
    r0_set, r0_sn = _exposure_ratio(w_a0, set_col), _exposure_ratio(w_a0, sn_col)
    r1_set, r1_sn = _exposure_ratio(w_a1, set_col), _exposure_ratio(w_a1, sn_col)
    print(f"    A0 pooled   : Settlement {r0_set:.3f}x  Seminat {r0_sn:.3f}x   "
          f"(docs report 1.27x / 3.08x)")
    print(f"    A1 per-class: Settlement {r1_set:.3f}x  Seminat {r1_sn:.3f}x")
    check("A1 Settlement ratio > 1.27 by a clear margin", r1_set > 1.27 + 0.10,
          f"{r1_set:.3f}x")
    check("A1 Settlement ratio rose vs A0", r1_set > r0_set, f"{r1_set:.3f} > {r0_set:.3f}")
    # zero-inflation backstop (handoff §3 step3): p95(combined w_raw) > p50
    w4r = np.array([float(r[1]) for r in rows]); w5r = np.array([float(r[2]) for r in rows])
    w_raw = 1.0 * w4r + 1.0 * w5r
    p50, p95 = np.percentile(w_raw, 50), np.percentile(w_raw, 95)
    check("p95(combined w_raw) > p50 (no zero-inflation collapse)", p95 > p50,
          f"p50={p50:.3e} p95={p95:.3e}")
    # Settlement-only vs no-minority tile w_tile comparison
    a1_map = {iid: w for iid, w in zip(img_ids, w_a1)}
    raw_map = {r[0]: (float(r[1]), float(r[2])) for r in rows}
    set_only = [i for i in range(len(img_ids)) if set_col[i] and not sn_col[i]]
    no_min = [i for i in range(len(img_ids)) if not set_col[i] and not sn_col[i]]
    if set_only and no_min:
        si, ni = set_only[0], no_min[0]
        sid, nid = img_ids[si], img_ids[ni]
        sw4, sw5 = raw_map.get(_norm_id(sid), (float("nan"),) * 2)
        nw4, nw5 = raw_map.get(_norm_id(nid), (float("nan"),) * 2)
        print(f"    Settlement-only tile {sid}: w4={sw4:.4e} w5={sw5:.4e} w_tile={a1_map[sid]:.4f}")
        print(f"    no-minority   tile {nid}: w4={nw4:.4e} w5={nw5:.4e} w_tile={a1_map[nid]:.4f}")
        check("Settlement-only tile w_tile clearly > no-minority tile",
              a1_map[sid] > a1_map[nid] + 0.05, f"{a1_map[sid]:.4f} vs {a1_map[nid]:.4f}")


# ===========================================================================
def gate_G3():
    print("\nG3 — recall-weight vector correctly indexed (length-6, weight[0]=0)")
    rl = RecallCrossEntropyLoss(num_classes=6, ignore_index=0, smooth_factor=0.05, momentum=0.0)
    rl.train()
    # Forced batch: Settlement(4) all wrong (recall 0); Forest(1) all right (recall 1).
    tgt = torch.full((1, 4, 4), 4)
    tgt[0, 0, :] = 1
    logits = torch.full((1, 6, 4, 4), -10.0)
    logits[0, 2, :, :] = 10.0   # predict Grassland everywhere -> Settlement wrong
    logits[0, 1, 0, :] = 10.0   # Forest row predicted Forest -> right
    rl.update_recall(logits, tgt)
    print("    c  class          weight")
    for c in range(6):
        print(f"    {c}  {STUDENT_CLASSES[c]:13s} {float(rl.weight[c]):.4f}")
    check("weight.shape[0] == 6", rl.weight.shape[0] == 6)
    check("weight[0] == 0 (Background ignored)", float(rl.weight[0]) == 0.0)
    check("lowest-recall class carries largest weight (argmax(weight[1:])+1 == 4 = Settlement)",
          int(rl.weight[1:].argmax()) + 1 == 4)

    # reduction='none' per-pixel ratio: a hand-set weight must scale per-pixel BEFORE reduction.
    w = torch.tensor([0.0, 1.0, 1.0, 1.0, 100.0, 1.0])
    ln = SoftCrossEntropyLoss(smooth_factor=0.0, ignore_index=0, reduction="none", weight=w)
    lu = SoftCrossEntropyLoss(smooth_factor=0.0, ignore_index=0, reduction="none", weight=None)
    torch.manual_seed(0)
    lg = torch.randn(2, 6, 8, 8)
    tg = torch.randint(0, 6, (2, 8, 8))
    tn, tu = ln(lg, tg), lu(lg, tg)
    ratios_ok = True
    for cls, wt in [(4, 100.0), (1, 1.0)]:
        idx = (tg == cls).nonzero()
        if len(idx):
            n, h, ww = idx[0]
            r = float(tn[n, 0, h, ww] / tu[n, 0, h, ww])
            ratios_ok = ratios_ok and abs(r - wt) < 1e-3
    check("reduction='none' per-pixel weighting is exact (Settlement 100x, Forest 1x)", ratios_ok)

    # weight=None vs weight=ones bitwise-equal (multiply is a no-op post-pad-fill).
    a = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0, weight=None)(lg, tg)
    b = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=0, weight=torch.ones(6))(lg, tg)
    check("weight=None and weight=ones give bitwise-equal loss", bool(a == b),
          f"{float(a):.6f} == {float(b):.6f}")

    # absent-class batch must NOT overwrite the buffer with recall=0.
    rl2 = RecallCrossEntropyLoss(num_classes=6, ignore_index=0, momentum=0.9)
    rl2.train()
    before = rl2.recall_ema.clone()
    tgt_nosettle = torch.full((1, 4, 4), 2)  # only Grassland present -> Settlement absent
    rl2.update_recall(torch.zeros(1, 6, 4, 4), tgt_nosettle)
    check("absent Settlement batch leaves recall_ema[4] untouched (no recall=0 bias)",
          float(rl2.recall_ema[4]) == float(before[4]),
          f"{float(rl2.recall_ema[4]):.3f}")


# ===========================================================================
def gate_G5():
    print("\nG5 — anchor join resolves all 5 foreground classes (CSV vs JSON names)")
    alias = {
        "Semi-natural": "Seminatural Grassland",
        "Forest": "Forest land",
        "Settlement": "Settlement",
        "Grassland": "Grassland",
        "Cropland": "Cropland",
    }
    # CSV-side names
    import csv
    csv_classes = set()
    with open(ANCHOR_CSV) as f:
        for r in csv.DictReader(f):
            if r["class"]:
                csv_classes.add(r["class"])
    from evaluation.compute_metrics import CLASS_NAMES_6
    json_classes = set(CLASS_NAMES_6)
    all_ok = True
    for csv_name, json_name in alias.items():
        ok = (csv_name in csv_classes) and (json_name in json_classes)
        all_ok = all_ok and ok
        print(f"    {csv_name:13s} (CSV {'ok' if csv_name in csv_classes else 'MISSING'})  ->  "
              f"{json_name:22s} (JSON {'ok' if json_name in json_classes else 'MISSING'})")
    check("all 5 foreground classes resolve on BOTH the CSV and JSON sides", all_ok)


# ===========================================================================
def gate_G8():
    print("\nG8 — sampler weights built from the right teacher")
    # The build-time assert (build_sampler_weights.load_student) already enforced
    # basename==stage2b_oem_finetune.ckpt and missing==0. Here we re-confirm the artifact
    # is present and well-formed (1846 base tiles, 3 columns).
    check("per-class TSV exists", PER_CLASS_TSV.exists(), str(PER_CLASS_TSV))
    with open(PER_CLASS_TSV) as f:
        n = sum(1 for _ in f) - 1  # minus header
    check("per-class TSV has 1846 tile rows", n == 1846, f"rows={n}")


def main():
    print("=" * 78)
    print("§9 PRE-FLIGHT GATES (CPU set: G1, G2, G3, G5, G8)")
    print("=" * 78)
    img_ids = _train_img_ids()
    pres = _class_presence(img_ids)
    gate_G1_G2(img_ids, pres)
    gate_G3()
    gate_G5()
    gate_G8()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} GATE(S) FAILED -> BLOCK: {FAILURES}")
        sys.exit(1)
    print("RESULT: ALL CPU GATES PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
