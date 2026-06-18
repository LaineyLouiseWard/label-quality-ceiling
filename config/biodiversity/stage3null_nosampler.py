"""
Stage 3 NULL CONTROL: Stage 2b (OEM transfer) continued WITHOUT the hard x minority weighting.

Purpose: isolate the hard x minority SAMPLING WEIGHTS from the +45-epoch fine-tune AND from the
draw-count. This config is IDENTICAL to config/biodiversity/stage3_sampler.py — same `train`
split, same init from the Stage 2b checkpoint, same loss / lr / PLAIN CosineAnnealingLR(T_max=45),
same 45 epochs, **same 2646 draws/epoch** — EXCEPT the sampler draws UNIFORMLY (with replacement)
instead of by hard x minority weights. The comparison

    Stage 3 sampler (stage3_sampler)  -  this uniform-draw control

therefore measures the effect of the WEIGHTING alone, holding the epoch budget AND the per-epoch
gradient-step count (2646) constant. (A plain shuffle would draw only 1846/epoch and so would
confound "no weighting" with "~30% fewer steps" — hence the uniform RandomSampler at num_samples=2646.)

Run with:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3null_nosampler.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, RandomSampler

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
# Training hyperparams (IDENTICAL to stage3_sampler)
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
weights_name = "stage3null_nosampler"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Same init as stage3_sampler: the Stage 2b OEM-transfer checkpoint.
pretrained_ckpt_path = (
    "model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt"
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
# Loss (IDENTICAL to stage3_sampler)
# ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False


# ======================
# Datasets (IDENTICAL to stage3_sampler: un-replicated train, transform=None)
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",
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
# Loaders — THE ONLY DIFFERENCE vs stage3_sampler: UNIFORM draws (no hard x minority weights),
# but the SAME 2646 draws/epoch so step-count is held constant.
# ======================
sampler = RandomSampler(train_dataset, replacement=True, num_samples=2646)

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
# Optimiser / scheduler (IDENTICAL to stage3_sampler: plain cosine, no warm restart)
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
