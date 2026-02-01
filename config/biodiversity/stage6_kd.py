"""
Stage 6: Knowledge Distillation (KD) on top of Stage 4 sampling (cumulative).

Goal: add a *gentle* KD regulariser without overriding the strong Stage 4 solution.

Cumulative ON:
- replication (train_rep split)
- Stage 4 hard × minority-rich sampling (WeightedRandomSampler from stage4_sampling_weights.tsv)
- Stage 4 train-time augmentation policy (random crops, NOT minority cropping)

PLUS:
- knowledge distillation (teacher -> student)

IMPORTANT:
- ignore_index = 0 everywhere
- student initialises from Stage 4 checkpoint
- KD is kept weak via low kd_alpha

Run:
  PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage6_kd.py
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
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
from geoseg.models.unet import TeacherUNet
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix
from geoseg.utils.optim import Lookahead, process_model_params


# ======================
# Training hyperparams (match your "fair ablation" base)
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
# KD parameters (KEEP GENTLE)
# ======================
kd_enabled = True
kd_temperature = 2.0
kd_alpha = 0.10  # <-- LOW so KD doesn't override Stage 4
rangeland_split_alpha = 0.7

# produced by scripts.export_teacher_checkpoint from model_weights/teacher/teacher.ckpt
teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints
# ======================
weights_name = "stage6_kd"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Initialise student from Stage 4 (do NOT start from scratch)
pretrained_ckpt_path = (
    "model_weights/biodiversity/stage4_sampling/"
    "stage4_sampling.ckpt"
)
resume_ckpt_path = None


# ======================
# Models
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)


# ======================
# Loss
# ======================
hard_loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)


class KDLoss(nn.Module):
    def __init__(self, hard_loss: nn.Module, kd_helper: KDHelper, alpha: float):
        super().__init__()
        self.hard_loss = hard_loss
        self.kd_helper = kd_helper
        self.alpha = alpha

    def forward(self, student_logits, targets, teacher_logits):
        loss_hard = self.hard_loss(student_logits, targets)
        loss_kd = self.kd_helper.compute_kd_loss(student_logits, teacher_logits)
        return (1.0 - self.alpha) * loss_hard + self.alpha * loss_kd


loss = KDLoss(hard_loss, kd_helper, alpha=kd_alpha)
use_aux_loss = False


# ======================
# Datasets (match Stage 4 policy: random crops)
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
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
# Stage 4 sampling weights (align by img_id)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing Stage 4 weights: {weights_path_tsv}\n"
        "Generate them with: python scripts/build_stage4_weights.py --ckpt <stage3b_finetune_ckpt>"
    )

id_to_weight = {}
with open(weights_path_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_id, w = line.split("\t")
        id_to_weight[img_id] = float(w)

sample_weights = []
missing = 0
for img_id in train_dataset.img_ids:
    w = id_to_weight.get(img_id, None)
    if w is None:
        sample_weights.append(1.0)
        missing += 1
    else:
        sample_weights.append(w)

print(f"[Stage6 KD] Loaded weights for {len(id_to_weight)} ids. Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    print("[Stage6 KD][WARN] Some train ids missing weights; defaulted to 1.0. Check ID alignment.")

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # fair: same draws per epoch
    replacement=True,
)


# ======================
# Loaders
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
# Optimiser / scheduler (match your base)
# ======================
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
