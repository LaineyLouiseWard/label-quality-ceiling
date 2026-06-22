"""
Stage 3 — RICHNESS-ONLY sampler = A0 MINUS the hardness term (defensibility ladder rung (ii)). Pooled
arXiv:1910.09217). Clone of stage3_sampler.py (A0) with ONLY the sampler weights swapped: A0's
bespoke `hardness^0.5 x pooled-richness^1.0` TSV -> a citable frequency-only inverse-tile-frequency
TSV (artifacts/sampler_weights_richonly.tsv), calibrated so Settlement stays flat (~1.27x) and
Semi-natural gets ~2.1x. Everything else (loss, init, optimiser, epochs, num_samples) is IDENTICAL
to A0 so the only difference is the sampler formula.

Ship rule (course-correction §15): ship this if it TIES A0 on Semi-natural val IoU (>=~74-75, no
majority damage); else keep A0's frozen sampler. Judge by IoU, not the ratio.

Run:
  SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_richonly.py --force
"""

from __future__ import annotations
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    val_aug,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# ====================== Training hyperparams (IDENTICAL to A0) ======================
max_epoch = 45
ignore_index = 0
train_batch_size = 2
val_batch_size = 2
lr = 3e-4
weight_decay = 2.5e-4
backbone_lr = 3e-5
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

# ====================== Logging / checkpoints ======================
weights_name = "stage3_richonly"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

pretrained_ckpt_path = (
    "model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt"
)
resume_ckpt_path = None

# ====================== Model ======================
net = ft_unetformer(
    pretrained=False, weight_path=None, num_classes=num_classes, decoder_channels=256
)

# ====================== Loss (IDENTICAL to A0) ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# ====================== Datasets ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train", transform=None
)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val", transform=val_aug
)
test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

# ====================== Sampler weights (class-balanced TSV) ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "sampler_weights_richonly.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing class-balanced sampler weights: {weights_path_tsv}\n"
        "Build with: python scripts/data_prep/build_sampler_weights.py --beta_temper 0 --gamma_rich 1.0 --ckpt <stage2b> --out artifacts/sampler_weights_richonly.tsv"
    )

id_to_weight = {}
with open(weights_path_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_id, w = line.split("\t")
        id_to_weight[img_id] = float(w)


def _norm_id(x):
    if "_rep" in x:
        b, r = x.rsplit("_rep", 1)
        if r.isdigit():
            return b
    return x


weights = []
missing = 0
for img_id in train_dataset.img_ids:
    w = id_to_weight.get(_norm_id(img_id), None)
    if w is None:
        weights.append(1.0)
        missing += 1
    else:
        weights.append(w)

print(f"[Stage3-richonly] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage3-richonly] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(weights=weights, num_samples=2646, replacement=True)

# ====================== Loaders (IDENTICAL to A0) ======================
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=4, pin_memory=True,
    sampler=sampler, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
    pin_memory=True, drop_last=False,
)

# ====================== Optimiser / scheduler (IDENTICAL to A0) ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
