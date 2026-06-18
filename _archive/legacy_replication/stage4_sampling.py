"""
Stage 4: Hard × minority-rich sampling on top of Stage 3b OEM finetuning.

Fair ablation rule:
- Keep EVERYTHING from Stage 3b the same:
  (data_root, transforms, loss, optimiser, scheduler)
- ONLY change the sampling distribution (WeightedRandomSampler)
- Initialise from the Stage 3b finetune checkpoint

Target: Biodiversity only

Cumulative ON:
- replication (train_rep)
- OEM pretraining + finetuning (via Stage 3a -> Stage 3b)

NEW in Stage 4:
- hard × minority-rich WeightedRandomSampler
  (weights built offline from Stage 3b checkpoint)

Run with:
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage4_sampling.py
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


# ======================
# Training hyperparams (BASE OFF STAGE 1 FAIR ABLATION)
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
weights_name = "stage4_sampling"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# IMPORTANT: initialise from Stage 3b finetune
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage3b_finetune/"
    "stage3b_finetune.ckpt"
)
resume_ckpt_path = None


# ======================
# Model
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)


# ======================
# Loss (STANDARD SUPERVISED; must match Stage 3b)
# ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False


# ======================
# Datasets (NO minority cropping)
# ======================
# NOTE: BiodiversityTrainDataset transform=None will still use train_aug_random internally
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    transform=None,
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# ======================
# Stage 4 sampling weights (DIRECT ID MATCH)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing Stage 4 weights: {weights_path_tsv}\n"
        "Generate with: python scripts/data_prep/build_stage4_weights.py --ckpt <stage3b_ckpt>"
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
    # The TSV is keyed by BASE id (build_stage4_weights strips _repN so replicas share
    # their base weight). Replicated tiles in train_rep carry a _repN suffix, so we must
    # strip it before the lookup or all 800 replicas silently default to weight 1.0.
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

print(f"[Stage4] Loaded weights for {len(id_to_weight)} ids. Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage4] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight "
        "(ID alignment broken). Refusing to train with silently mis-weighted sampling."
    )

sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),  # fair: same draws per epoch
    replacement=True,
)


# ======================
# Loaders
# ======================
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)


# ======================
# Optimiser / scheduler (must match Stage 3b)
# ======================
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}

net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(
    net_params, lr=lr, weight_decay=weight_decay
)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
