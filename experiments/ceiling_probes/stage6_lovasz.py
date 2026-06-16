"""
STAGE 6 (EXPERIMENTAL) — Lovasz-Softmax fine-tune on top of Stage 5 KD.

Question this answers
---------------------
Lovasz-Softmax (Berman et al., CVPR 2018) directly optimises the IoU/Jaccard index — the metric
we actually report. Nothing else in the pipeline does (CE optimises likelihood, Dice optimises
soft-overlap, the sampler changes exposure, KD matches the teacher). It is also TEACHER-INDEPENDENT,
so the teacher's inability to distinguish Seminatural/Grassland (it is carved from OEM "Rangeland")
does not limit it.

This is a SHORT fine-tune INITIALISED FROM the Stage 5 KD checkpoint, IDENTICAL to the Stage 6
confusion control EXCEPT the hard loss replaces Dice with Lovasz:
    hard = CE(label_smooth 0.05) + Lovasz   (was CE + Dice)
CE and KD (alpha=0.10) are kept. So it compares DIRECTLY against the lambda=0 control
(stage6_confusion_l0 = CE + Dice + KD continued 12 epochs): same init, same schedule, only Dice->Lovasz.

Run:
  PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage6_lovasz.py --force \
    2>&1 | tee /tmp/stage6_lovasz.log
"""

from __future__ import annotations
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from segmentation_models_pytorch.losses import LovaszLoss
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
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix, REMAP_OUTPUT_CLASSES
from geoseg.utils.optim import Lookahead, process_model_params

assert REMAP_OUTPUT_CLASSES == CLASSES


# ======================
# Training hyperparams — SHORT fine-tune (matched to the Stage 6 control)
# ======================
max_epoch = 12
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 1e-4
weight_decay = 2.5e-4
backbone_lr = 1e-5
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# KD parameters (IDENTICAL to Stage 5 — KD kept on)
# ======================
kd_enabled = True
kd_temperature = 2.0
kd_alpha = 0.10
rangeland_split_alpha = 0.7
teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints
# ======================
weights_name = "stage6_lovasz"
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
    "model_weights/biodiversity/stage5_norep/stage5_norep.ckpt"
)
resume_ckpt_path = None


# ======================
# Models (IDENTICAL to Stage 5)
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

assert Path(teacher_checkpoint).exists(), f"Missing teacher checkpoint: {teacher_checkpoint}"
teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)


# ======================
# Loss — KD + (CE + Lovasz), Dice replaced by Lovasz
# ======================
class LovaszKDLoss(nn.Module):
    """KD loss (Stage 5) with the hard term's Dice replaced by Lovasz-Softmax.
    hard = CE(label_smooth) + Lovasz ;  loss = (1-alpha)*hard + alpha*kd.
    Lovasz is computed in fp32 (its sort/cumsum is precision-sensitive under bf16).
    """

    def __init__(self, kd_helper, alpha, ignore_index=0, smooth_factor=0.05):
        super().__init__()
        self.kd_helper = kd_helper
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.smooth_factor = smooth_factor
        self.lovasz = LovaszLoss(mode="multiclass", ignore_index=ignore_index, from_logits=True)

    def forward(self, student_logits, targets, teacher_logits):
        valid = (targets != self.ignore_index)

        ce = F.cross_entropy(
            student_logits, targets, ignore_index=self.ignore_index,
            label_smoothing=self.smooth_factor,
        )
        # Lovasz in fp32 (bf16 sort is unstable)
        with torch.autocast(device_type="cuda", enabled=False):
            lov = self.lovasz(student_logits.float(), targets)
        loss_hard = ce + lov

        kd_map = self.kd_helper.compute_kd_loss(
            student_logits, teacher_logits, reduction="none"
        ).sum(dim=1)
        loss_kd = (kd_map * valid).sum() / valid.sum().clamp(min=1)

        return (1.0 - self.alpha) * loss_hard + self.alpha * loss_kd


loss = LovaszKDLoss(kd_helper, alpha=kd_alpha, ignore_index=ignore_index)
use_aux_loss = False

print(f"[Stage6-lovasz] hard = CE + Lovasz (Dice replaced) | KD alpha={kd_alpha} | init=stage5_norep")


# ======================
# Datasets — IDENTICAL to Stage 5
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
# Sampling weights (REUSE base-keyed TSV — sampler kept)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])
weights_path_tsv = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(f"Missing Stage 4 weights: {weights_path_tsv}")

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


sample_weights = []
missing = 0
for img_id in train_dataset.img_ids:
    w = id_to_weight.get(_norm_id(img_id), None)
    if w is None:
        sample_weights.append(1.0)
        missing += 1
    else:
        sample_weights.append(w)

print(f"[Stage6-lovasz] Loaded weights for {len(id_to_weight)} ids. Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(f"[Stage6-lovasz] {missing} train ids have no sampling weight.")

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=2646, replacement=True)


# ======================
# Loaders (IDENTICAL to Stage 5)
# ======================
train_loader = DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, num_workers=4,
    pin_memory=True, sampler=sampler, drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=val_batch_size, num_workers=4,
    shuffle=False, pin_memory=True, drop_last=False,
)


# ======================
# Optimiser / scheduler — short plain cosine (matched to the control)
# ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
