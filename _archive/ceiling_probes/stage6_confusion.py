"""
STAGE 6 (EXPERIMENTAL) — confusion-aware fine-tune on top of Stage 5 KD.

GROUNDED RE-TEST (2026-06-17): now uses the grounded KD mapping (build_mapping_from_confusion("B"))
and initialises from the grounded `stage5_norep` checkpoint. The earlier null result was measured on the
PRE-grounding pipeline (where the teacher's "Grassland" signal was corrupted by the Agriculture->Cropland
hand-map), so it was confounded — this re-tests whether the mapping fix unlocks it. Run ONLY after the
grounded KD-B (`stage5_norep`) finishes. The leakage figures quoted below are from the OLD confusion matrix;
re-derive them from the grounded Figure 09 before citing. See docs/KD_MAPPING_GROUNDING.md, docs/TODO.md (gate 3).

Question this answers
---------------------
The confusion matrix shows BOTH minority classes leak into the dominant majority, Grassland:
Seminatural->Grassland 10.2%, Settlement->Grassland 7.1% (Settlement also ->Forest 6.7%).
This stage adds a DIRECTIONAL cost-matrix penalty (cost-sensitive learning: Khan et al. 2018,
IEEE TNNLS, arXiv:1508.03422; Elkan 2001) that up-weights the CE on minority-class pixels in
proportion to the probability mass the student puts on Grassland. It targets the SHARED confuser
of both rare classes with a single term.

It is a SHORT fine-tune INITIALISED FROM the Stage 5 KD checkpoint — none of baseline / transfer /
sampler / KD is retrained. The sampler and KD teacher are KEPT active (this is KD + confusion).

Controlled sweep (lambda from env CONF_LAMBDA):
  - CONF_LAMBDA=0  -> CONTROL: identical fine-tune, NO confusion weighting. Any gain at lambda>0
                     over this control is attributable to the term, not to 12 extra epochs.
  - CONF_LAMBDA=2,4 -> the real penalties.
weights_name encodes lambda so the runs do not collide.

Run (sweep):
  for L in 0 2 4; do
    CONF_LAMBDA=$L PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage6_confusion.py --force \
      2>&1 | tee /tmp/stage6_confusion_l${L}.log
  done
"""

from __future__ import annotations
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import DiceLoss
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
from geoseg.utils.kd_utils import KDHelper, build_mapping_from_confusion, REMAP_OUTPUT_CLASSES
from geoseg.utils.optim import Lookahead, process_model_params

assert REMAP_OUTPUT_CLASSES == CLASSES, (
    f"KD channel mismatch — teacher remap order {REMAP_OUTPUT_CLASSES} "
    f"!= student CLASSES {CLASSES}"
)

# Class indices (Background=0, Forest=1, Grassland=2, Cropland=3, Settlement=4, Seminatural=5)
GRASS_IDX = 2
MINORITY_IDX = (4, 5)  # Settlement, Seminatural — both leak into Grassland


# ======================
# Sweep parameter
# ======================
CONF_LAMBDA = float(os.environ.get("CONF_LAMBDA", "2.0"))
_lam_tag = f"{CONF_LAMBDA:g}".replace(".", "p").replace("-", "m")


# ======================
# Training hyperparams — SHORT fine-tune from a converged checkpoint
# ======================
max_epoch = 12            # short refine, not a full 45-epoch run
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 1e-4                 # reduced LR: refine a converged model, do not disrupt it (Cui et al. 2020 regime)
weight_decay = 2.5e-4
backbone_lr = 1e-5        # 10x ratio kept
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# KD parameters (IDENTICAL to Stage 5 — KD is kept on)
# ======================
kd_enabled = True
kd_temperature = 2.0
kd_alpha = 0.10
teacher_checkpoint = "pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"


# ======================
# Logging / checkpoints (lambda-keyed so the sweep does not collide)
# ======================
weights_name = f"stage6_confusion_l{_lam_tag}"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Initialise from the finished Stage 5 KD checkpoint — nothing earlier is retrained.
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

assert Path(teacher_checkpoint).exists(), (
    f"Missing teacher checkpoint: {teacher_checkpoint}"
)
teacher = TeacherUNet(num_classes=9, pretrained=False)
teacher.load_checkpoint(teacher_checkpoint)
teacher.freeze()

mapping_matrix = build_mapping_from_confusion("B")  # grounded KD mapping (matches stage5_norep)
kd_helper = KDHelper(mapping_matrix=mapping_matrix, temperature=kd_temperature)


# ======================
# Loss — Stage 5 KD loss + directional minority->Grassland cost-matrix penalty
# ======================
class ConfusionKDLoss(nn.Module):
    """KD loss (Stage 5) with a cost-sensitive re-weighting of the hard CE:
    for pixels whose TRUE label is a minority class, multiply the CE by
        (1 + lambda * P(Grassland))
    so probability mass placed on the dominant confuser (Grassland) is penalised more.
    lambda=0 reproduces the Stage 5 KD loss exactly (the sweep control).
    Cost-sensitive framing: Khan et al. 2018 (IEEE TNNLS); Elkan 2001.
    """

    def __init__(self, kd_helper, dice, alpha, lam, ignore_index=0,
                 confuser=GRASS_IDX, minorities=MINORITY_IDX, smooth_factor=0.05):
        super().__init__()
        self.kd_helper = kd_helper
        self.dice = dice
        self.alpha = alpha
        self.lam = lam
        self.ignore_index = ignore_index
        self.confuser = confuser
        self.minorities = tuple(minorities)
        self.smooth_factor = smooth_factor

    def forward(self, student_logits, targets, teacher_logits):
        valid = (targets != self.ignore_index)

        # Per-pixel label-smoothed CE (ignored pixels contribute 0).
        ce_map = F.cross_entropy(
            student_logits, targets, reduction="none",
            ignore_index=self.ignore_index, label_smoothing=self.smooth_factor,
        )
        # Cost weight: only on minority-true pixels, scaled by predicted Grassland mass.
        p_conf = student_logits.softmax(dim=1)[:, self.confuser]
        is_min = torch.zeros_like(targets, dtype=torch.bool)
        for c in self.minorities:
            is_min |= (targets == c)
        weight = 1.0 + self.lam * p_conf * is_min.float()
        weighted_ce = (weight * ce_map * valid).sum() / valid.sum().clamp(min=1)

        loss_hard = weighted_ce + self.dice(student_logits, targets)

        # KD term (identical to Stage 5: per-pixel KL summed over classes, averaged over valid).
        kd_map = self.kd_helper.compute_kd_loss(
            student_logits, teacher_logits, reduction="none"
        ).sum(dim=1)
        loss_kd = (kd_map * valid).sum() / valid.sum().clamp(min=1)

        return (1.0 - self.alpha) * loss_hard + self.alpha * loss_kd


dice_term = DiceLoss(smooth=0.05, ignore_index=ignore_index)
loss = ConfusionKDLoss(kd_helper, dice_term, alpha=kd_alpha, lam=CONF_LAMBDA,
                       ignore_index=ignore_index)
use_aux_loss = False

print(f"[Stage6-confusion] lambda={CONF_LAMBDA} (tag l{_lam_tag}) | "
      f"penalise minority{MINORITY_IDX}->Grassland({GRASS_IDX}) | init=stage5_norep | "
      f"{'CONTROL (no confusion term)' if CONF_LAMBDA == 0 else 'confusion term ON'}")


# ======================
# Datasets — IDENTICAL to Stage 5 (un-replicated train, sampler kept)
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
# Sampling weights (REUSE the base-keyed TSV — sampler kept active)
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

print(f"[Stage6-confusion] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage6-confusion] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=2646,   # step-matched, same as Stage 4/5
    replacement=True,
)


# ======================
# Loaders (IDENTICAL to Stage 5)
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
# Optimiser / scheduler — short plain cosine over the 12-epoch refine
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
