"""
Stage 4 NULL CONTROL: Stage 3b continued WITHOUT the hard x minority sampler.

Purpose: isolate the *epochs* confound from the *sampler* mechanism. This config is identical
to stage4_sampling.py in every respect (init from the Stage 3b checkpoint, same data_root
train_rep, same loss / optimiser / scheduler, same 45 epochs) EXCEPT it uses a plain shuffled
DataLoader instead of the WeightedRandomSampler. The comparison

    Stage 4 (stage4_sampling)  -  Stage 4 null (this)

therefore measures the effect of the hard x minority-aware sampling alone, with the extra 45
warm-start epochs held constant across both. Not part of the default run: triggered via
RUN_NULL_CONTROLS=1 (stage N4 in RUNBOOK.sh). Evaluated automatically by C1 (rglob).

Run with:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4null_nosampler.py
"""

from __future__ import annotations

from torch.utils.data import DataLoader
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
# Training hyperparams (identical to Stage 4)
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
weights_name = "stage4null_nosampler"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Same init as Stage 4: the Stage 3b finetune checkpoint.
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
# Loss (identical to Stage 4)
# ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False


# ======================
# Datasets (identical to Stage 4)
# ======================
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
# Loaders  (THE ONLY DIFFERENCE vs Stage 4: plain shuffle, no WeightedRandomSampler)
# ======================
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
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
# Optimiser / scheduler (identical to Stage 4)
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
