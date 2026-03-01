"""
Export teacher model weights from a Lightning checkpoint to a plain PyTorch state_dict.

This is used to:
- Load teacher weights in KD training without Lightning
- Reuse teacher pretraining as a standard .pth file

Run once after teacher training.
"""

import argparse
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
    "--ckpt",
    type=str,
    default="model_weights/teacher/teacher.ckpt",
    help="Teacher Lightning checkpoint (.ckpt)",
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # Lightning checkpoint typically stores weights under "state_dict"
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # If trained using train_supervision.py, weights are often under "net.*"
    # Strip "net." prefix if present.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("net."):
            cleaned[k.replace("net.", "", 1)] = v
        else:
            cleaned[k] = v

    torch.save(cleaned, str(out_path))
    print(f"Saved teacher state_dict to: {out_path}")
    print(f"Num keys: {len(cleaned)}")

if __name__ == "__main__":
    main()
