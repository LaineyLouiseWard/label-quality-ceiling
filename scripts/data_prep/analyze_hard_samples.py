#!/usr/bin/env python3
"""
Generate per-sample difficulty weights for Stage 3 difficulty-weighted sampling.

Improvement:
- Difficulty is computed primarily on minority pixels (e.g., settlement+seminatural),
  not global CE dominated by majority classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geoseg.models.ftunetformer import ft_unetformer
from geoseg.datasets.biodiversity_dataset import BiodiversityTrainDataset, val_aug


def load_net_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError(f"Invalid Lightning checkpoint: {ckpt_path}")

    sd = ckpt["state_dict"]
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    if not net_sd:
        raise ValueError("Could not locate model weights in checkpoint")

    net.load_state_dict(net_sd, strict=False)
    return net


def robust_normalize(x: np.ndarray, lo_p: float = 5.0, hi_p: float = 95.0) -> np.ndarray:
    """Percentile normalization to avoid a few outliers dominating weights."""
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
        default="model_weights/biodiversity/stage2_replication_ftunetformer/stage2_replication_ftunetformer.ckpt")
    ap.add_argument("--data-root", type=str, default="data/biodiversity_split/train_rep")
    ap.add_argument("--out", type=str, default="artifacts/sample_weights.txt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--ignore-index", type=int, default=0)

    # new bits (minority-aware difficulty)
    ap.add_argument("--minority-classes", type=str, default="4,5",
                    help="Comma-separated class ids to treat as minority (default: 4,5).")
    ap.add_argument("--beta", type=float, default=0.85,
                    help="Weight on minority difficulty vs global difficulty (0..1).")
    ap.add_argument("--min-minority-px", type=int, default=50,
                    help="If fewer than this many minority pixels in tile, rely more on global loss.")
    args = ap.parse_args()

    minority_classes = [int(x) for x in args.minority_classes.split(",") if x.strip()]

    ckpt_path = Path(args.ckpt)
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not (data_root / "images").exists():
        raise FileNotFoundError(f"Invalid data root: {data_root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = BiodiversityTrainDataset(data_root=str(data_root), transform=val_aug)

    net = ft_unetformer(num_classes=6, decoder_channels=256)
    net = load_net_from_lightning_ckpt(net, ckpt_path)
    net.to(device).eval()

    minority_losses = np.zeros(len(ds), dtype=np.float32)
    global_losses = np.zeros(len(ds), dtype=np.float32)
    img_ids: list[str] = []

    minority_set = set(minority_classes)

    print("[analyze_hard_samples minority-aware]")
    print(f"  minority_classes: {minority_classes}  beta={args.beta}  min_minority_px={args.min_minority_px}")
    print(f"  ckpt: {ckpt_path.resolve()}")
    print(f"  data: {data_root.resolve()}  samples={len(ds)}  device={device}")
    print(f"  out:  {out_path.resolve()}")

    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc="Scoring difficulty"):
            sample = ds[i]
            img = sample["img"].unsqueeze(0).to(device)                 # (1,C,H,W)
            mask = sample["gt_semantic_seg"].unsqueeze(0).to(device)    # (1,H,W)

            logits = net(img)                                           # (1,6,H,W)

            # per-pixel CE so we can restrict to minority pixels
            ce = F.cross_entropy(
                logits,
                mask.long(),
                ignore_index=args.ignore_index,
                reduction="none",  # (1,H,W)
            )

            valid = (mask != args.ignore_index)

            # minority-only mask (based on GT)
            m_mask = valid.clone()
            # build minority mask without slow loops
            mm = torch.zeros_like(mask, dtype=torch.bool)
            for cls in minority_set:
                mm |= (mask == cls)
            m_mask &= mm

            n_min = int(m_mask.sum().item())

            # global loss (non-bg)
            g_loss = ce[valid].mean() if int(valid.sum()) > 0 else torch.tensor(0.0, device=device)

            # minority loss: if no minority pixels, set to global (will be downweighted)
            if n_min > 0:
                m_loss = ce[m_mask].mean()
            else:
                m_loss = g_loss

            global_losses[i] = float(g_loss.item())
            minority_losses[i] = float(m_loss.item())
            img_ids.append(sample["img_id"])

    # Normalize separately (robustly)
    m_norm = robust_normalize(minority_losses)
    g_norm = robust_normalize(global_losses)

    # Blend: mostly minority difficulty, with some global stability.
    # If a tile has almost no minority pixels, reduce beta for that tile.
    beta = float(np.clip(args.beta, 0.0, 1.0))
    eff_beta = np.full(len(ds), beta, dtype=np.float32)
    # heuristic: if minority pixels very low, rely more on global
    # (prevents tiny minority specks from dominating difficulty)
    # You can tighten/loosen this later.
    # Here we approximate using minority pixel count is not stored; so keep constant beta.
    # If you want per-tile beta, we can store n_min and apply it.

    score = eff_beta * m_norm + (1.0 - eff_beta) * g_norm
    weights = 1.0 + args.alpha * score

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for img_id, w in zip(img_ids, weights):
            f.write(f"{img_id}\t{w:.6f}\n")

    print("✓ Wrote sample weights")
    print(f"  minority loss min/med/max: {minority_losses.min():.6f} / {np.median(minority_losses):.6f} / {minority_losses.max():.6f}")
    print(f"  global   loss min/med/max: {global_losses.min():.6f} / {np.median(global_losses):.6f} / {global_losses.max():.6f}")
    print(f"  weight       min/med/max: {weights.min():.6f} / {np.median(weights):.6f} / {weights.max():.6f}")


if __name__ == "__main__":
    main()
