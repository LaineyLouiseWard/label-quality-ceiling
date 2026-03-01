"""
Stage 5: Late-phase mixed minority cropping on top of Stage 4 sampling.

Fair ablation rule:
- Start from Stage 4 checkpoint
- Keep Stage 4 sampler ON (same weights file, same num_samples)
- Only change train-time transform schedule:
    first (1 - late_frac): random crops
    last (late_frac): mixed random/minority crops

Run:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage5_cropping.py
"""

from __future__ import annotations
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    train_aug_random,
    train_aug_minority,
    val_aug,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
)


from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params


# ======================
# Hyperparams (match Stage 4)
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
# Stage 5 schedule knobs
# ======================
late_frac = 0.30          # last 30% epochs
p_minor_late = 0.30       # in late phase: 30% minority crops, 70% random


def make_mixed_aug(rng):
    def mixed_aug(img, mask):
        if rng.random() < p_minor_late:
            return train_aug_minority(img, mask)
        return train_aug_random(img, mask)
    return mixed_aug



class LatePhaseTransformSwitch(pl.Callback):
    def __init__(self, train_ds, switch_epoch: int, base_seed: int = 42):
        super().__init__()
        self.train_ds = train_ds
        self.switch_epoch = switch_epoch
        self.base_seed = base_seed
        self.did_switch = False

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch >= self.switch_epoch:
            # Deterministic per epoch
            rng = random.Random(self.base_seed + epoch)

            if not self.did_switch:
                print(f"[Stage5] Switching train transform to mixed_aug at epoch {epoch}.")
                self.did_switch = True

            self.train_ds.transform = make_mixed_aug(rng)



# ======================
# Logging / checkpoints
# ======================
weights_name = "stage5_cropping"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Start FROM Stage 4
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage4_sampling/"
    "stage4_sampling.ckpt"
)
resume_ckpt_path = None


# ======================
# Model / loss
# ======================
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


# ======================
# Datasets
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train_rep",
    transform=train_aug_random,  # start with random; we switch later
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# ======================
# Keep Stage 4 sampler ON (same weights file)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing Stage 4 weights: {weights_path_tsv}\n"
        "Generate with: python scripts/data_prep/build_stage4_weights.py --ckpt <stage3b_ckpt>"
    )

id_to_weight = {}
with open(weights_path_tsv, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_id, w = line.split("\t")
        id_to_weight[img_id] = float(w)

weights = []
for img_id in train_dataset.img_ids:
    weights.append(id_to_weight.get(img_id, 1.0))

sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ======================
# Loaders
# ======================
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    sampler=sampler,
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
# Optimizer / scheduler
# ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


# ======================
# Callback injected into Trainer via config
# ======================
switch_epoch = int(max_epoch * (1.0 - late_frac))
callbacks = [LatePhaseTransformSwitch(train_dataset, switch_epoch=switch_epoch)]
