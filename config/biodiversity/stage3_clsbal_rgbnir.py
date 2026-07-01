"""
RGB+NIR 4-channel variant of the class-balanced (clsbal) cell — EXPERIMENT (branch experiment/rgb-nir).

Purpose: test whether the semi-natural <-> improved-grassland confusion is partly an UNUSED-INPUT
problem. On Ireland-only data NDVI (d~0.5-0.75) and within-scene elevation (d~1.56) DO separate the
two classes, but the shipped model is RGB-only because the ADE20K Swin-B backbone is 3-channel; NIR
(band 4) sits in the tiles, unused. See docs/RGB_NIR_EXPERIMENT_PLAN.md and
docs/REPRODUCIBILITY_INVESTIGATION_2026-07-01.md.

Design note: OEM transfer (stage2a) trains on OpenEarthMap, which is RGB-only, so a 4-channel model
CANNOT use the OEM-transfer lever. This variant therefore inits directly from the ADE20K stem with the
Red->NIR inflated first conv (handled inside ft_unetformer), i.e. it is the from-stem + clsbal cell.
The matched RGB control is the RGB from-stem + clsbal cell (stage_sampler_only). Everything else
(loss, sampler + TSV, num_samples, optimiser, CosineAnnealingWarmRestarts, 45 epochs, batch/LR variant)
is IDENTICAL to stage3_clsbal.py; the ONLY differences are in_chans=4 and the from-stem init.

NIR channel: first-conv NIR filter seeded by COPYING the pretrained RED filter (timm in_chans default;
Pan et al. 2019, CoinNet). NIR normalised with its own train-split stat (mean 0.5855, std 0.2732,
post per-tile 2-98 stretch). RandomBrightnessContrast is left applying to all 4 channels (documented
simplification, kept for comparability with the RGB run).

Run:
  SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_clsbal_rgbnir.py --force
"""

from __future__ import annotations
import os
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

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


# ====================== Training hyperparams (IDENTICAL to stage3_clsbal) ======================
max_epoch = 45
ignore_index = 0
_BV = os.environ.get("BATCH_VARIANT", "b2")
assert _BV in ("b2", "b4"), f"BATCH_VARIANT must be b2 or b4, got {_BV!r}"
_LR_SCALE = 2.0 if _BV == "b4" else 1.0
_BATCH = 4 if _BV == "b4" else 2
train_batch_size = _BATCH
val_batch_size = _BATCH
lr = 3e-4 * _LR_SCALE
weight_decay = 2.5e-4
backbone_lr = 3e-5 * _LR_SCALE
backbone_weight_decay = 2.5e-4
num_classes = 6
classes = CLASSES

# ====================== Logging / checkpoints ======================
weights_name = "stage3_clsbal_rgbnir" + os.environ.get("SWEEP_TAG", "")
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# OEM (stage2a) is RGB-only, so there is NO 4-channel transfer checkpoint. Init directly from the
# ADE20K stem with the Red->NIR inflated first conv (done in ft_unetformer). pretrained_ckpt_path MUST
# be None so train_supervision does NOT try to load the 3-ch stage2a ckpt onto the 4-ch stem (that would
# silently random-init the stem — the frac<0.9 guard would not fire for a 2/439-param mismatch).
pretrained_ckpt_path = None
resume_ckpt_path = None

# ====================== Model (4-channel RGB+NIR) ======================
net = ft_unetformer(
    pretrained=True,
    weight_path="pretrain_weights/stseg_base.pth",
    num_classes=num_classes,
    decoder_channels=256,
    in_chans=4,
)

# ====================== Loss (IDENTICAL) ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# ====================== Datasets (IDENTICAL; loader now yields 4-band RGB+NIR) ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train", transform=None
)
val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val", transform=val_aug
)
test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

# ====================== Sampler weights (class-balanced TSV, IDENTICAL) ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = Path(os.environ.get("SAMPLER_TSV", str(repo_root / "artifacts" / "sampler_weights_clsbal.tsv")))
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

print(f"[Stage3-clsbal-rgbnir] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage3-clsbal-rgbnir] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight."
    )

sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)

# ====================== Loaders (IDENTICAL) ======================
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=4, pin_memory=True,
    sampler=sampler, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
    pin_memory=True, drop_last=False,
)

# ====================== Optimiser / scheduler (IDENTICAL) ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2
)
