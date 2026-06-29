"""
Stage 1 (baseline): supervised training on biodiversity_split with random crops/augs.
Saves Lightning checkpoints under weights_path, monitored by val_mIoU.
"""

import os
from torch.utils.data import DataLoader
import torch

# Mosaic augmentation ratio — env-gated for the pre-launch screen (0.0 = off).
MOSAIC_RATIO = float(os.environ.get("MOSAIC_RATIO", "0.0"))
# Optional output-name suffix so screen variants don't collide (e.g. "_nomosaic").
RUN_TAG = os.environ.get("RUN_TAG", "")

from geoseg.losses import *
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    val_aug,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# -----------------------
# Training hyperparams
# -----------------------
max_epoch = 45

# Loss/metric ignore label (you confirmed: background=0 should be ignored)
ignore_index = 0

# --- Batch/LR variant (env-gated): BATCH_VARIANT=b2 (default) | b4 — MUST match across all 5 cells ---
#   b2 = batch 2, lr 3e-4 / backbone_lr 3e-5  (linear-scaling-correct for batch 2)
#   b4 = batch 4, lr 6e-4 / backbone_lr 6e-5  (deployed-lineage setting; LR scaled x2 with batch)
_BV = os.environ.get("BATCH_VARIANT", "b2")
assert _BV in ("b2", "b4"), f"BATCH_VARIANT must be b2 or b4, got {_BV!r}"
_LR_SCALE = 2.0 if _BV == "b4" else 1.0
_BATCH = 4 if _BV == "b4" else 2

train_batch_size = _BATCH
val_batch_size = _BATCH

lr = 3e-4 * _LR_SCALE
weight_decay = 2.5e-4
backbone_lr = 3e-5 * _LR_SCALE
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# -----------------------
# Logging / checkpoints
# -----------------------
weights_name = f"stage1_baseline{RUN_TAG}"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
gpus = "auto"
resume_ckpt_path = None


# -----------------------
# Model (ADE20K-pretrained Swin-B backbone via stseg_base.pth)
# -----------------------
net = ft_unetformer(
    pretrained=True,
    weight_path="pretrain_weights/stseg_base.pth",
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


# -----------------------
# Datasets
# -----------------------
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",
    transform=train_aug_random,
    mosaic_ratio=MOSAIC_RATIO,
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# -----------------------
# Loaders
# -----------------------
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


# -----------------------
# Optimizer / scheduler
# -----------------------
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
