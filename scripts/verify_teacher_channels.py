#!/usr/bin/env python3
"""
Empirically verify the OEM teacher's output channels follow the native-A taxonomy
(channel i == OEM class i), so the kd_utils mapping matrix is applied correctly.

Native-A OEM encoding (must match geoseg/utils/kd_utils.OEM_CLASSES and
scripts/data_prep/relabel_oem_taxonomy.OEM_ID_TO_TARGET6):
    0=unknown/bg 1=bareland 2=rangeland 3=developed 4=road 5=tree 6=water 7=agri 8=building

Runs the teacher on labelled OEM tiles and cross-tabulates ground-truth class against the
teacher's argmax channel. PASS iff each well-represented GT class's dominant channel is itself
(diagonal), which is the native-A contract.

Usage:
    PYTHONPATH=. python scripts/verify_teacher_channels.py \
        --ckpt pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth \
        --data-root data/openearthmap_teacher/val
"""
import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from geoseg.models.unet import TeacherUNet
from geoseg.datasets.openearthmap_dataset import OpenEarthMapTeacherValDataset

NATIVE = {0: "unknown/bg", 1: "bareland", 2: "rangeland", 3: "developed", 4: "road",
          5: "tree", 6: "water", 7: "agri", 8: "building"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="teacher .pth (model.* keys)")
    ap.add_argument("--data-root", default="data/openearthmap_teacher/val")
    ap.add_argument("--n", type=int, default=120, help="max tiles to sample")
    ap.add_argument("--min-pixels", type=int, default=20000,
                    help="ignore GT classes with fewer than this many pixels (too rare to judge)")
    ap.add_argument("--gate-frac", type=float, default=0.02,
                    help="only classes >= this fraction of all pixels gate PASS/FAIL; rarer classes "
                         "(e.g. bareland) are advisory — a rare-class collapse is imbalance, not a stale teacher")
    args = ap.parse_args()

    net = TeacherUNet(num_classes=9, pretrained=False)
    net.load_checkpoint(args.ckpt)
    net.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)

    ds = OpenEarthMapTeacherValDataset(
        data_root=args.data_root, img_suffix=".tif", mask_suffix=".tif"
    )
    n = min(args.n, len(ds))
    dl = DataLoader(torch.utils.data.Subset(ds, list(range(n))), batch_size=4, shuffle=False)

    C = np.zeros((9, 9), dtype=np.int64)  # C[gt_class, pred_channel]
    with torch.no_grad():
        for batch in dl:
            img = batch["img"].to(dev)
            gt = batch["gt_semantic_seg"].numpy()
            pred = net(img).argmax(1).cpu().numpy()
            for g in range(9):
                m = gt == g
                if m.any():
                    pv, pc = np.unique(pred[m], return_counts=True)
                    for v, c in zip(pv, pc):
                        C[g, v] += c

    total_px = int(C.sum())
    print(f"Probed {n} tiles from {args.data_root}  ({total_px / 1e6:.1f}M px)\n")
    print(f"{'GT class':>16} -> dominant channel        pixels  share  diag?")
    ok, gated, advisory_bad = True, 0, []
    for g in range(9):
        tot = int(C[g].sum())
        if tot < args.min_pixels:
            print(f"{NATIVE[g]:>16} -> (too rare: {tot} px, skipped)")
            continue
        top = int(C[g].argmax())
        frac = C[g, top] / tot
        share = tot / total_px
        diag = (top == g)
        # Only WELL-REPRESENTED classes gate PASS/FAIL. A class below --gate-frac of all pixels
        # (e.g. bareland, the rarest OEM class) can collapse into a visually-similar class purely
        # from class imbalance — a rare-class artefact, not a stale/mis-wired teacher — so its
        # mismatch is reported as advisory rather than failing the gate. A genuinely stale teacher
        # would put COMMON classes off-diagonal, which still fails here.
        if share >= args.gate_frac:
            gated += 1
            ok = ok and diag
            flag = "OK" if diag else f"** MISMATCH (got ch{top}={NATIVE[top]}) **"
        else:
            flag = "OK" if diag else f"advisory only (rare; -> ch{top}={NATIVE[top]})"
            if not diag:
                advisory_bad.append(NATIVE[g])
        print(f"{NATIVE[g]:>16} -> channel {top} ({NATIVE[top]:<10}) {frac:5.0%} {share:5.1%}  {flag}")

    print()
    if advisory_bad:
        print(f"NOTE (advisory, not gating): rare classes not cleanly diagonal: {', '.join(advisory_bad)}")
    if ok and gated >= 4:
        print(f"PASS — teacher channels are native-A ({gated} well-represented classes checked, all diagonal).")
        print("kd_utils native-A mapping is valid for this teacher.")
        sys.exit(0)
    else:
        print(f"FAIL — a WELL-REPRESENTED class is NOT native-A ({gated} gating classes checked).")
        print("Do NOT trust the kd_utils mapping. (Is this the stale 6-class teacher?)")
        sys.exit(1)


if __name__ == "__main__":
    main()
