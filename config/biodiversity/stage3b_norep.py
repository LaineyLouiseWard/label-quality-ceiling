"""
Stage 3b (NO-REPLICATION variant): OEM->Biodiversity finetune on the UN-replicated train split.

Why this exists
---------------
The canonical `stage3b_finetune.py` trains on `train_rep` (replicated). The cheap redundancy test
(`stage4_norep.py`) therefore branches from a checkpoint that ALREADY saw replication, so it only
answers "is replication redundant AT THE SAMPLER STAGE". For a paper that claims a *single* imbalance
mechanism (the hard x minority sampler) with NO static duplication anywhere, the transfer/finetune
stage must also be replication-free — otherwise the reported pipeline still contains replication
upstream. This config provides that fully-clean upstream.

Decoupling rationale (normal sampling here, rebalance later):
- Kang et al. 2020 (Decoupling) and Zhou et al. 2020 (BBN) show instance-balanced (normal) sampling
  learns the most generalisable representations; rebalancing is deferred to a later stage.
- Caveat (dense prediction): this is a classification finding. In segmentation the i.i.d. premise is
  weaker (Cui et al. 2022, Region Rebalance; Li et al. 2024, Frequency-based Matcher). We therefore
  treat normal-sampling transfer as the *intuition*, and let the sampler stage do the rebalancing.

This is IDENTICAL to config/biodiversity/stage3b_finetune.py EXCEPT:
  - data_root: train_rep  ->  train   (no replicas; minorities seen at native frequency)
  - weights_name: stage3b_finetune -> stage3b_norep   (isolated weights/logs/eval dir)
It still initialises from the SAME Stage 3a OEM pretrain checkpoint.

For the fully-clean pipeline, point Stage 4/5 at this checkpoint instead of stage3b_finetune
(set `pretrained_ckpt_path` in stage4_norep.py / stage5_norep.py to model_weights/biodiversity/
stage3b_norep/stage3b_norep.ckpt). The cheap sampler-stage test leaves them pointing at the
replicated stage3b_finetune.

Run:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3b_norep.py --force
"""

from __future__ import annotations

from torch.utils.data import DataLoader
import torch

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
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


# -------------------
# Training hyperparams (IDENTICAL to stage3b_finetune.py)
# -------------------
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


# -------------------
# Logging / checkpoints  (ISOLATED name)
# -------------------
weights_name = "stage3b_norep"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

# init from the SAME Stage 3a OEM pretrain as stage3b_finetune
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage3a_pretrain/"
    "stage3a_pretrain.ckpt"
)

resume_ckpt_path = None
gpus = "auto"


# -------------------
# Model / loss (IDENTICAL)
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
# Datasets  -- THE ONLY DATA CHANGE: un-replicated train
# -------------------
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",   # <-- was train_rep
    transform=train_aug_random,
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# -------------------
# Loaders (vanilla shuffle; sampling comes in Stage 4) -- IDENTICAL
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
# Optimizer / scheduler (IDENTICAL)
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
