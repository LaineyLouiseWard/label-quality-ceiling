"""
Stage 3 — FULL clsbal recipe: clsbal sampler + RECALL LOSS + TARGETED copy-paste (docs/MINORITY_STRATEGY §16).

The "best shot" arm: standard class-balanced sampler (defensible) + recall-weighted CE (Tian 2022,
Semi-nat lever) + targeted Settlement copy-paste (§16.6, Settlement lever). Tests whether BOTH minorities
lift together beyond seed noise on a simple/defensible base. Attribution comes from the isolated arms
(stage3_clsbal_recall = recall only; stage3_copypaste = paste only); this shows the combined ceiling.
Compare to plain clsbal (seed42: Settlement 73.85, Semi-nat 73.16) and to A0 (75.15 Semi-nat).

Run:
  SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_clsbal_recall_cp.py --force
"""

from __future__ import annotations
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss, RecallCrossEntropyLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    val_aug,
    configure_settlement_copypaste,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# ====================== Training hyperparams (IDENTICAL to A0) ======================
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

# ====================== Logging / checkpoints ======================
weights_name = "stage3_clsbal_recall_cp"
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
    "model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt"
)
resume_ckpt_path = None

# ====================== Model ======================
net = ft_unetformer(
    pretrained=False, weight_path=None, num_classes=num_classes, decoder_channels=256
)

# ====================== Loss — recall-weighted CE + Dice (ONLY change vs clsbal) ======================
# RecallCrossEntropyLoss subclasses SoftCrossEntropyLoss and self-updates its (1-recall) per-class
# weight from training batches (val never perturbs it — train_supervision sets self.loss=config.loss,
# so .eval() propagates). smooth_factor matched to clsbal; momentum=0.9 EMA (recall_ce default, stable
# for the ~3.3% Settlement pixels).
loss = JointLoss(
    RecallCrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index,
                           smooth_factor=0.05, momentum=0.9),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# ====================== Lever — TARGETED Settlement copy-paste (same as stage3_copypaste §16.6) ======================
# Stacks the Settlement copy-paste on top of clsbal+recall. paste_onto=(0,2,3) = open land (fixes the
# Background-only no-op, §16.3); targeted=True = confidence-weighted donors. This is the full proposed
# clsbal recipe: standard sampler + recall loss (Semi-nat lever) + copy-paste (Settlement lever).
configure_settlement_copypaste(
    enabled=True,
    donor_root="data/biodiversity_split/train",
    prob=0.5,
    n_donors=1,
    targeted=True,
    paste_onto=(0, 2, 3),
)

# ====================== Datasets ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train", transform=None
)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val", transform=val_aug
)
test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

# ====================== Sampler weights (class-balanced TSV) ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "sampler_weights_clsbal.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing class-balanced sampler weights: {weights_path_tsv}\n"
        "Build with: python scripts/data_prep/build_clsbal_sampler.py"
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


weights = []
missing = 0
for img_id in train_dataset.img_ids:
    w = id_to_weight.get(_norm_id(img_id), None)
    if w is None:
        weights.append(1.0)
        missing += 1
    else:
        weights.append(w)

print(f"[Stage3-clsbal] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage3-clsbal] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(weights=weights, num_samples=2646, replacement=True)

# ====================== Loaders (IDENTICAL to A0) ======================
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=4, pin_memory=True,
    sampler=sampler, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
    pin_memory=True, drop_last=False,
)

# ====================== Optimiser / scheduler (IDENTICAL to A0) ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
