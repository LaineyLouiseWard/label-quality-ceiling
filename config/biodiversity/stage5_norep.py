"""
NO-REPLICATION TEST ARM of Stage 5 (KD), continuing the no-replication branch.

Identical to config/biodiversity/stage5_kd.py EXCEPT:
  - data_root: train_rep -> train  (no replicas; sampler runs as-is, P_min=0.686)
  - weights_name: stage5_kd -> stage5_norep  (isolated weights/logs/eval dir)
  - init from stage4_norep instead of stage4_sampling
The SAME base-keyed artifacts/stage4_sampling_weights.tsv and the SAME teacher are reused, so
the ONLY difference vs stage5_kd is the absence of replication. This gives the apples-to-apples
endpoint to compare against stage5_kd (the current pipeline's final model).

Run (after stage4_norep finishes):
  PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage5_norep.py --force
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
from geoseg.utils.kd_utils import KDHelper, create_mapping_matrix, REMAP_OUTPUT_CLASSES
from geoseg.utils.optim import Lookahead, process_model_params

assert REMAP_OUTPUT_CLASSES == CLASSES, (
    f"KD channel mismatch — teacher remap order {REMAP_OUTPUT_CLASSES} "
    f"!= student CLASSES {CLASSES}"
)


# ======================
# Training hyperparams (IDENTICAL to stage5_kd.py)
# ======================
max_epoch = 45
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 3e-4  # proven LR (matches Stage 4); 1e-4 was empirically too slow. Must match Stage 4. See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14
weight_decay = 2.5e-4
backbone_lr = 3e-5  # 10x ratio kept
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# KD parameters (IDENTICAL)
# ======================
kd_enabled = True
kd_temperature = 2.0
kd_alpha = 0.10  # scale-corrected back to the working value: our per-pixel-summed KL puts ~0.1 at the seg-KD norm (Niemann λ=0.1); 0.5/0.7 over-weighted the teacher ~6x. See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §12-13
rangeland_split_alpha = 0.7

teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints  (ISOLATED name)
# ======================
weights_name = "stage5_norep"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Initialise student from the NO-REP Stage 4
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
    f"Missing teacher checkpoint: {teacher_checkpoint}\n"
    "Export it with: python -m scripts.data_prep.export_teacher_checkpoint "
    "--ckpt model_weights/teacher/teacher.ckpt "
    f"--out {teacher_checkpoint}"
)
teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

mapping_matrix = create_mapping_matrix(alpha=rangeland_split_alpha)
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
# Datasets  -- THE ONLY DATA CHANGE: un-replicated train
# ======================
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


# ======================
# Sampling weights (REUSE the existing base-keyed TSV)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing Stage 4 weights: {weights_path_tsv}\n"
        "Generate them with: python scripts/data_prep/build_stage4_weights.py --ckpt <stage3b_finetune_ckpt>"
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

print(f"[Stage5-norep KD] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage5-norep KD] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight "
        "(ID alignment broken). Refusing to train with silently mis-weighted sampling."
    )

sampler = WeightedRandomSampler(
    weights=sample_weights,
    # STEP-MATCHED to the replicated arm (1846 base + 800 replicas = 2646 draws/epoch) so the no-rep
    # comparison isolates the exposure drop rather than ~43% fewer gradient steps/epoch.
    # See docs/CROSSCHECK_REVIEW_2026-06-14.md (nuance N5).
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

# Plain cosine annealing (must match Stage 4). Switched from warm restarts for manuscript
# defensibility; the earlier plain-cosine "underperformance" (0.750) was a compile-from-scratch
# artefact, not a real schedule effect. See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14.
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0
)
