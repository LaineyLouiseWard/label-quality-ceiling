"""
KD MAPPING-TEST ARM **B** (full data-driven harmonisation).

Identical to config/biodiversity/stage5_norep.py EXCEPT the KD mapping matrix: EVERY OEM row is
the row-normalised teacher->GT soft confusion (a soft label-transition matrix, cf. Patrini 2017).
Same Stage 4 init, same teacher, same hyperparameters -> isolates the mapping effect vs the
existing stage5_norep control (0.831). See docs/KD_MAPPING_GROUNDING.md.
(rangeland_split_alpha is unused here — B overrides the whole matrix, including the Rangeland row.)

Run (after the confusion matrix exists at artifacts/teacher_oem_gt_confusion.npz):
  PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage5_mapB.py
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
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix, REMAP_OUTPUT_CLASSES, build_mapping_from_confusion
from geoseg.utils.optim import Lookahead, process_model_params

assert REMAP_OUTPUT_CLASSES == CLASSES, (
    f"KD channel mismatch — teacher remap order {REMAP_OUTPUT_CLASSES} "
    f"!= student CLASSES {CLASSES}"
)


# ======================
# Training hyperparams (IDENTICAL to stage5_norep.py)
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
# KD parameters (IDENTICAL)
# ======================
kd_enabled = True
kd_temperature = 2.0
kd_alpha = 0.10
rangeland_split_alpha = 0.73  # unused under B (whole matrix overridden)

teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints  (ISOLATED name)
# ======================
weights_name = "stage5_mapB"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Same Stage 4 init as the control (isolates the mapping change)
pretrained_ckpt_path = (
    "model_weights/biodiversity/stage4_norep/"
    "stage4_norep.ckpt"
)
resume_ckpt_path = None


# ======================
# Models (IDENTICAL)
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)

assert Path(teacher_checkpoint).exists(), (
    f"Missing teacher checkpoint: {teacher_checkpoint}"
)
teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

# --- THE ONLY METHODOLOGICAL CHANGE: full data-driven mapping (Option B) ---
mapping_matrix = build_mapping_from_confusion("B")
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)


# ======================
# Loss (IDENTICAL)
# ======================
hard_loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)


class KDLoss(nn.Module):
    def __init__(self, hard_loss: nn.Module, kd_helper: KDHelper, alpha: float,
                 ignore_index: int = 0):
        super().__init__()
        self.hard_loss = hard_loss
        self.kd_helper = kd_helper
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, student_logits, targets, teacher_logits):
        loss_hard = self.hard_loss(student_logits, targets)
        kd_map = self.kd_helper.compute_kd_loss(
            student_logits, teacher_logits, reduction="none"
        ).sum(dim=1)
        valid = (targets != self.ignore_index)
        loss_kd = (kd_map * valid).sum() / valid.sum().clamp(min=1)
        return (1.0 - self.alpha) * loss_hard + self.alpha * loss_kd


loss = KDLoss(hard_loss, kd_helper, alpha=kd_alpha, ignore_index=ignore_index)
use_aux_loss = False


# ======================
# Datasets (IDENTICAL to stage5_norep)
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
# Sampling weights (REUSE the existing base-keyed TSV)
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

print(f"[Stage5-mapB KD] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage5-mapB KD] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=2646,
    replacement=True,
)


# ======================
# Loaders (IDENTICAL)
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
# Optimiser / scheduler (IDENTICAL)
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
