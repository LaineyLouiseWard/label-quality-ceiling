"""
Stage 2a (OEM transfer, pre-train): supervised training on the combined dataset
(Biodiversity + OEM, harmonised to 6 classes via the grounded OEM->student mapping).

First half of the Stage 2 OEM-transfer step; Stage 2b (stage2b_oem_finetune.py) fine-tunes
this checkpoint on Biodiversity alone.

Fair ablation rule (for the paper):
- Use the SAME core training hyperparams as Stage 1 (lr/backbone_lr/weight_decay/etc.)
- OEM data is used ONLY in this pretraining stage via the combined 6-class dataset.
- Validation is Biodiversity-only to avoid OEM leakage into reported val curves.

Run with:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2a_oem_pretrain.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityValDataset,
)
from geoseg.datasets.biodiversity_oem_dataset import BiodiversityOEMTrainDataset
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -------------------
# Training hyperparams (MATCH STAGE 1)
# -------------------
max_epoch = 45
ignore_index = 0  # background ignored in loss/metrics

train_batch_size = 2
val_batch_size = 2

# IMPORTANT: Use Stage 1 values (your chosen "fair ablation" base)
lr = 3e-4
weight_decay = 2.5e-4
backbone_lr = 3e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -------------------
# Logging / checkpoints
# -------------------
weights_name = "stage2a_oem_pretrain"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

# Stage 3a starts from scratch
pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = "auto"


# -------------------
# Model / loss (standard supervised)
# -------------------
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False


# -------------------
# Datasets
# -------------------
# Combined pretraining dataset (Biodiversity + OEM already harmonised to 0..5)
train_dataset = BiodiversityOEMTrainDataset(
    data_root="data/biodiversity_oem_combined/train",
    transform=train_aug_random,
)

# Validation is Biodiversity-only (no OEM leakage)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)


# -------------------
# Loaders
# -------------------
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
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


# -------------------
# Optimizer / scheduler
# -------------------
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
