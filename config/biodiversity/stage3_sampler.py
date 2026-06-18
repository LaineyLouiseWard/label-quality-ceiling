"""
Stage 3 (hard x minority sampling): the single imbalance-mitigation mechanism.

A WeightedRandomSampler (replacement=True) over the Biodiversity `train` split, with per-tile
weights from artifacts/sampler_weights.tsv (w proportional to hardness x minority-richness). This
is the only rebalancing in the pipeline — there is no static minority duplication. Minority
sampling exposure under these weights is P_min=0.686 on the un-replicated train set.

Initialises from the Stage 2b OEM-transfer checkpoint (stage2b_oem_finetune). Continues to the
same 45-epoch budget and matches the per-epoch gradient-step count via num_samples=2646 (see the
sampler block). The weights TSV is base-keyed, so the sampler builder runs it directly on `train`
with no replicas.

Run:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_sampler.py --force
"""

from __future__ import annotations
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


# ======================
# Training hyperparams
# ======================
max_epoch = 45
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 3e-4  # proven LR; 1e-4 was empirically too slow for the sampler's distribution shift. See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14
weight_decay = 2.5e-4
backbone_lr = 3e-5  # 10x ratio kept
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# Logging / checkpoints  (ISOLATED name)
# ======================
weights_name = "stage3_sampler"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# init from the Stage 2b OEM-transfer checkpoint
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage2b_oem_finetune/"
    "stage2b_oem_finetune.ckpt"
)
resume_ckpt_path = None


# ======================
# Model
# ======================
net = ft_unetformer(
    pretrained=False,
    weight_path=None,
    num_classes=num_classes,
    decoder_channels=256,
)


# ======================
# Loss (IDENTICAL)
# ======================
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0,
    1.0,
)

use_aux_loss = False


# ======================
# Datasets  -- Biodiversity train at native frequency (no replication; sampler does the rebalancing)
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",
    transform=None,
)

val_dataset = BiodiversityValDataset(
    data_root="data/biodiversity_split/val",
    transform=val_aug,
)

test_dataset = BiodiversityTestDataset(
    data_root="data/biodiversity_split/test",
)


# ======================
# Sampler weights (base-keyed TSV)
# ======================
here = Path(__file__).resolve()
repo_root = next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])

weights_path_tsv = repo_root / "artifacts" / "sampler_weights.tsv"
if not weights_path_tsv.exists():
    raise FileNotFoundError(
        f"Missing sampler weights: {weights_path_tsv}\n"
        "Generate with: python scripts/data_prep/build_sampler_weights.py --ckpt <stage2b_ckpt>"
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

print(f"[Stage3-sampler] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage3-sampler] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight "
        "(ID alignment broken). Refusing to train with silently mis-weighted sampling."
    )

sampler = WeightedRandomSampler(
    weights=weights,
    # Fixed per-epoch step budget (2646 draws) shared by Stage 3, Stage 4 (KD) and the null controls
    # so they are mutually step-matched: comparisons isolate the named mechanism (sampler / KD) from
    # the per-epoch gradient-step count. (2646 = the 1846 train tiles + 800 effective minority draws.)
    # See docs/CROSSCHECK_REVIEW_2026-06-14.md (nuance N5).
    num_samples=2646,
    replacement=True,
)


# ======================
# Loaders (IDENTICAL)
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
# Optimiser / scheduler (IDENTICAL)
# ======================
layerwise_params = {
    "backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}

net_params = process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(
    net_params, lr=lr, weight_decay=weight_decay
)
optimizer = Lookahead(base_optimizer)

# Plain cosine annealing over the full schedule. Chosen over CosineAnnealingWarmRestarts for
# MANUSCRIPT DEFENSIBILITY: warm restarts (SGDR) buy anytime-performance / snapshot-ensembling —
# neither of which this single-model fine-tune uses — and would need defending to a reviewer,
# whereas monotone cosine is the standard, no-questions default for fine-tuning from a strong init.
# The earlier "0.750 (cosine) vs 0.814 (warm restart)" gap was an ARTEFACT, not a schedule effect:
# both 0.750 runs trained from random init (torch.compile renamed keys before the Stage 2b load).
# See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14.
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0
)
