"""
NO-REPLICATION TEST ARM of Stage 4 (hard x minority sampling).

Question this answers
---------------------
Is static replication redundant given the WeightedRandomSampler? Because the sampler uses
replacement=True and the weights TSV is keyed by BASE tile id, replication is mathematically
a special case of weight-scaling (see scripts/analysis/compute_replication_exposure.py:
s_match = 2.0 == 1+r). So a FULLY exposure-matched no-rep arm would be a null by construction.
The INFORMATIVE test is this one: drop replication and run the sampler AS-IS on the
un-replicated train set. Minority sampling exposure then falls from P_min=0.814 (current)
to 0.686. If Settlement/Semi-natural IoU survive that drop, replication is redundant in
practice and the pipeline can be simplified. If they fall, the exposure mattered (and can be
recovered by x2 minority weights — no physical replication needed).

This is IDENTICAL to config/biodiversity/stage4_sampling.py EXCEPT:
  - data_root: train_rep  ->  train   (no replicas)
  - weights_name: stage4_sampling -> stage4_norep   (isolated weights/logs/eval dir)
The SAME artifacts/stage4_sampling_weights.tsv is reused (it is base-keyed, so it applies to
the un-replicated train set directly — no GPU recompute). Initialised from the CLEAN Stage 3b
(stage3b_norep), so the whole pipeline is replication-free end-to-end. (Faster preview: point
pretrained_ckpt_path back at stage3b_finetune — which still saw replication — to test the sampler
stage alone without re-training Stage 3b.)

Run:
  PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4_norep.py --force
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
# Training hyperparams (IDENTICAL to stage4_sampling.py)
# ======================
max_epoch = 45
ignore_index = 0

train_batch_size = 2
val_batch_size = 2

lr = 3e-4  # proven LR (old run -> 0.814); 1e-4 was empirically too slow for the sampler's distribution shift. See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14
weight_decay = 2.5e-4
backbone_lr = 3e-5  # 10x ratio kept
backbone_weight_decay = 2.5e-4

num_classes = 6
classes = CLASSES


# ======================
# Logging / checkpoints  (ISOLATED name)
# ======================
weights_name = "stage4_norep"
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# init from the CLEAN (un-replicated) Stage 3b so the pipeline is replication-free END-TO-END.
# (Faster preview alternative: set this back to stage3b_finetune/stage3b_finetune.ckpt to test the
#  sampler stage alone on top of the old replication-trained foundation — see docstring.)
pretrained_ckpt_path = (
    "model_weights/biodiversity/"
    "stage3b_norep/"
    "stage3b_norep.ckpt"
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
# Datasets  -- THE ONLY DATA CHANGE: un-replicated train
# ======================
train_dataset = BiodiversityTrainDataset(
    data_root="data/biodiversity_split/train",   # <-- was train_rep
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
# Stage 4 sampling weights (REUSE the existing base-keyed TSV)
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

print(f"[Stage4-norep] Loaded weights for {len(id_to_weight)} ids. "
      f"Missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(
        f"[Stage4-norep] {missing}/{len(train_dataset.img_ids)} train ids have no sampling weight "
        "(ID alignment broken). Refusing to train with silently mis-weighted sampling."
    )

sampler = WeightedRandomSampler(
    weights=weights,
    # STEP-MATCHED to the replicated arm: 1846 base + 800 minority replicas = 2646 draws/epoch.
    # If left at len(weights)=1846 the no-rep arm runs ~43% fewer gradient steps/epoch, which would
    # confound "replication removed" with "less training". Matching the draw count isolates the ONLY
    # thing this test is for: the minority-exposure drop 0.814 -> 0.686.
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
# both 0.750 runs trained from random init (torch.compile renamed keys before the Stage 3b load).
# See docs/MANUSCRIPT_IMPLICATIONS_NOREP.md §14.
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=0
)
