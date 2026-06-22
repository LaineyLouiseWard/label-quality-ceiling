"""Stage 4 NULL CONTROL for SELF-DISTILLATION — CLSBAL lineage (no A0).

This is the matched no-KD control for `config/biodiversity/stage4_selfdistil.py`. It is a
byte-for-byte copy of the self-distil arm EXCEPT the loss is the plain hard JointLoss (CE+Dice)
and there is NO teacher — so `distilled − this control` toggles ONLY the distillation, with the
Stage-3 init, the sampler, the 45 warm-start epochs, the augmentation, the LR/schedule and the
step budget all held identical.

Why this file exists (do NOT reuse stage4null_nokd.py): stage4null_nokd.py is the control for the
A0/OEM-teacher `stage4_kd` arm — it inits from the A0 Stage-3 student (`stage3_sampler`) and uses
the A0 sampler TSV (`sampler_weights.tsv`). Pairing THAT against a clsbal-init, clsbal-sampler
distilled student would confound distillation with a whole change of Stage-3 recipe (different
oversampler + different warm-start). This control is CLSBAL on both, so the only moving part vs the
distilled arm is the loss/teacher. A0 appears nowhere.

Run (NO KD -> plain supervision trainer), per seed in each clsbal worktree:
  SEED=42 \
  STUDENT_INIT_CKPT=model_weights/biodiversity/stage3_clsbal/stage3_clsbal.ckpt \
  SAMPLER_TSV=artifacts/sampler_weights_clsbal.tsv \
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4null_nokd_clsbal.py --force
(Both env vars already DEFAULT to clsbal below; supply them only to be explicit.)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
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
from geoseg.utils.optim import Lookahead, process_model_params


# ====================== Training hyperparams (IDENTICAL to stage4_selfdistil) ======================
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

# ====================== Logging / checkpoints (ISOLATED name) ======================
weights_name = "stage4null_nokd_clsbal"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# CLSBAL init (same default + env override as stage4_selfdistil.py — A0 is dead).
pretrained_ckpt_path = os.environ.get(
    "STUDENT_INIT_CKPT",
    "model_weights/biodiversity/stage3_clsbal/stage3_clsbal.ckpt",
)
resume_ckpt_path = None

repo_root = next((p for p in Path(__file__).resolve().parents if (p / "artifacts").exists()),
                 Path(__file__).resolve().parents[2])

# ====================== Model (IDENTICAL) ======================
net = ft_unetformer(pretrained=False, weight_path=None, num_classes=num_classes,
                    decoder_channels=256)

# ====================== Loss — HARD ONLY (no KD, no teacher) ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0,
)
use_aux_loss = False

# ====================== Datasets (IDENTICAL) ======================
train_dataset = BiodiversityTrainDataset(data_root="data/biodiversity_split/train",
                                         transform=train_aug_random)
val_dataset = BiodiversityValDataset(data_root="data/biodiversity_split/val", transform=val_aug)
test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

# ====================== Sampler — CLSBAL (same default + env override as stage4_selfdistil) ======================
weights_path_tsv = Path(os.environ.get("SAMPLER_TSV", str(repo_root / "artifacts" / "sampler_weights_clsbal.tsv")))
if not weights_path_tsv.exists():
    raise FileNotFoundError(f"Missing clsbal sampler weights: {weights_path_tsv}")

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


weights = []
missing = 0
for img_id in train_dataset.img_ids:
    w = id_to_weight.get(_norm_id(img_id), None)
    weights.append(w if w is not None else 1.0)
    missing += int(w is None)

print(f"[Stage4null-clsbal] sampler {weights_path_tsv.name}: {len(id_to_weight)} ids, "
      f"missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(f"[Stage4null-clsbal] {missing} train ids have no sampling weight.")

sampler = WeightedRandomSampler(weights=weights, num_samples=2646, replacement=True)

# ====================== Loaders (IDENTICAL) ======================
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4,
                          pin_memory=True, sampler=sampler, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
                        pin_memory=True, drop_last=False)

# ====================== Optimiser / scheduler (IDENTICAL) ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
