"""
Stage 4 NULL CONTROL: Stage 3 sampler continued WITHOUT knowledge distillation.

Purpose: isolate the KD mechanism from the +45-epoch fine-tune. This config is IDENTICAL to
config/biodiversity/stage4_kd.py in EVERY respect that matters — same `train` split, same
`train_aug_random`, same hard x minority WeightedRandomSampler (num_samples=2646), same
lr 3e-4 / backbone 3e-5, same PLAIN CosineAnnealingLR(T_max=45), same init from the Stage 3
sampler checkpoint, same 45 epochs — EXCEPT it applies the plain hard loss (CE+Dice) instead of
the KD loss, and uses NO teacher. The comparison

    Stage 4 KD (stage4_kd)  -  this no-KD control

therefore measures the effect of KD ALONE, with the +45 warm-start epochs and the sampler held
constant. This is the MANDATORY per-seed control in the 5-seed campaign (it separates KD's effect
from the extra training that Stage 4 also adds).

Run with (NO KD -> plain supervision trainer):
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4null_nokd.py
"""

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    train_aug_random,
    val_aug,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# ======================
# Training hyperparams (IDENTICAL to stage4_kd)
# ======================
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


# ======================
# Logging / checkpoints
# ======================
weights_name = "stage4null_nokd"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Same init as stage4_kd: the Stage 3 sampler checkpoint. The ONLY differences vs stage4_kd
# are: this config applies no KD (plain CE+Dice) and has no teacher.
pretrained_ckpt_path = (
    "model_weights/biodiversity/stage3_sampler/stage3_sampler.ckpt"
)
resume_ckpt_path = None


# ======================
# Model (IDENTICAL)
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)


# ======================
# Loss — the Stage 4 HARD loss only (KD term removed; no teacher)
# ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False


# ======================
# Datasets (IDENTICAL to stage4_kd: un-replicated train + train_aug_random)
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",
    transform=train_aug_random,
)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)
test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# ======================
# Sampling weights (IDENTICAL to stage4_kd: base-keyed TSV, num_samples=2646)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "sampler_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(f"Missing sampler weights: {weights_path_tsv}")

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

print(f"[Stage4-null-noKD] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage4-null-noKD] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=2646,   # step-matched, IDENTICAL to stage4_kd
    replacement=True,
)


# ======================
# Loaders (IDENTICAL to stage4_kd)
# ======================
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)


# ======================
# Optimiser / scheduler (IDENTICAL to stage4_kd: plain cosine, no warm restart)
# ======================
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0
)
