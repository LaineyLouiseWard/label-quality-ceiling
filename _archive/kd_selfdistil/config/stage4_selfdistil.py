"""Stage 4 — SELF/ENSEMBLE distillation (replaces the OEM-teacher KD).

Protocol: docs/MINORITY_STRATEGY_2026-06-19.md §11 (BRIEF B). The OEM EfficientNet-B4 teacher
is dropped — it is taxonomy-blind to our minorities and weaker than the student (~38% vs ~80%
mIoU), so "you cannot distil up" (docs/STAGE4_KD_AND_SAMPLER_DIAGNOSIS.md §2). Here the teacher
is the IN-DOMAIN uniform softmax-mean ENSEMBLE of the shipped Stage-3 recipe's N seed students,
which DOES beat the best single seed (+1.19 pp measured, job 428205) — a teacher strong enough to
distil from. Everything else (data regime, sampler, augmentation, optimiser, schedule, 45 epochs,
step budget 2646) is IDENTICAL to stage4_kd so "distilled vs best-single" is step-matched and the
only moving part is the teacher.

SHIPPED RECIPE = CLSBAL (resolved 2026-06-20). A0 is dead — the defaults below are CLSBAL, not A0,
so a forgotten override can never silently train A0. Ensemble teacher = the 5 clsbal Stage-3 students.

Run (one seed; repeat across the campaign's seeds for the paired test):
  SEED=42 \
  ENSEMBLE_MANIFEST=$BASE/ensemble_members_clsbal.txt \
  KD_ALPHA=0.5 KD_T=2.0 \
  PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage4_selfdistil.py --force
(STUDENT_INIT_CKPT and SAMPLER_TSV already default to clsbal; only ENSEMBLE_MANIFEST must be supplied.)

Keep-it bar (scripts/bakeoff/selfdistil_readout.py): distilled beats the STEP-MATCHED no-KD control
(clsbal Stage-3 + the SAME 45 epochs, KD off) by >=0.5 pp, paired, 95% CI excluding 0 — NOT vs the
Stage-3 snapshot (the distilled student trains 45 extra epochs that alone buy ~+1.2 mIoU; §9.1). The
readout refuses to declare KEEP without --control.

------------------------------------------------------------------------------------------------
Env overrides (all default to CLSBAL; change only if the shipped recipe ever changes):
  * ENSEMBLE_MANIFEST  paths to the 5 clsbal Stage-3 .ckpt across the seed worktrees (NOT committed —
                       env-specific absolute paths). The one var you MUST supply.
  * STUDENT_INIT_CKPT  per-worktree clsbal Stage-3 ckpt (default below; resolves locally per worktree).
  * SAMPLER_TSV        clsbal sampler weights (default artifacts/sampler_weights_clsbal.tsv).
  * COPYPASTE=1        only if a copy-paste arm ever ships (mirrors config/biodiversity/stage3_copypaste.py).
------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss
from geoseg.losses.selfdistill import SelfDistillLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    val_aug,
    train_aug_random,
    configure_settlement_copypaste,
)
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.models.ensemble_teacher import EnsembleTeacher
from geoseg.utils.optim import Lookahead, process_model_params


# ====================== Training hyperparams (IDENTICAL to stage4_kd) ======================
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

# ====================== Self-distillation knobs (protocol §11.3) ======================
# alpha can be HIGH here (teacher is STRONGER than the student) — the opposite of OEM-KD's 0.10.
kd_enabled = True
kd_alpha = float(os.environ.get("KD_ALPHA", "0.5"))   # §11.3 start point; sweepable
kd_temperature = float(os.environ.get("KD_T", "2.0"))  # T-search {1,2,5,10}

# ====================== Logging / checkpoints (ISOLATED name) ======================
# SWEEP_TAG lets the S2 robustness grid run many (T,alpha) cells at the same seed in the same
# worktree without colliding (e.g. SWEEP_TAG=_T5_a07). Empty for Arm A (the default cell).
weights_name = "stage4_selfdistil" + os.environ.get("SWEEP_TAG", "")
weights_path = f"model_weights/biodiversity/{weights_name}"
test_weights_name = weights_name
log_name = f"biodiversity/{weights_name}"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
gpus = "auto"

# Init the distilled student from the shipped Stage-3 checkpoint (per-worktree path, like stage4_kd).
pretrained_ckpt_path = os.environ.get(
    "STUDENT_INIT_CKPT",
    "model_weights/biodiversity/stage3_clsbal/stage3_clsbal.ckpt",  # clsbal ships; A0 is dead
)
resume_ckpt_path = None

repo_root = next((p for p in Path(__file__).resolve().parents if (p / "artifacts").exists()),
                 Path(__file__).resolve().parents[2])

# ====================== Student ======================
net = ft_unetformer(pretrained=False, weight_path=None, num_classes=num_classes,
                    decoder_channels=256)

# ====================== Teacher = in-domain ENSEMBLE (uniform softmax mean) ======================
def _build_member():
    return ft_unetformer(pretrained=False, weight_path=None, num_classes=num_classes,
                         decoder_channels=256)


def _read_manifest(path: Path):
    paths = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (repo_root / p)
        if not p.exists():
            raise FileNotFoundError(f"Ensemble member checkpoint missing: {p} (from {path})")
        paths.append(p)
    return paths


_manifest = Path(os.environ.get("ENSEMBLE_MANIFEST", str(repo_root / "artifacts" / "ensemble_members.txt")))
if not _manifest.exists():
    raise FileNotFoundError(
        f"Ensemble manifest not found: {_manifest}\n"
        "Self-distillation needs the N-seed ensemble of the SHIPPED Stage-3 recipe. After the\n"
        "bake-off names the recipe and its N-seed campaign finishes, write one checkpoint path per\n"
        "line into this file (the per-seed Stage-3 .ckpt across the seed worktrees), then re-run.\n"
        "Override the location with ENSEMBLE_MANIFEST=<path>."
    )
_member_ckpts = _read_manifest(_manifest)
print(f"[Stage4-selfdistil] ensemble teacher = {len(_member_ckpts)} members @ T={kd_temperature} "
      f"(alpha={kd_alpha})")
teacher = EnsembleTeacher.from_checkpoints(_member_ckpts, _build_member, temperature=kd_temperature)

# ====================== Loss ======================
hard_loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index),
    1.0, 1.0,
)
# ignore_index=0 → KD masked to foreground, matching stage4_kd's KDLoss convention.
loss = SelfDistillLoss(hard_loss, alpha=kd_alpha, temperature=kd_temperature,
                       ignore_index=ignore_index)
use_aux_loss = False

# ====================== Optional copy-paste (match the shipped recipe's data regime) ======================
if os.environ.get("COPYPASTE", "0") == "1":
    configure_settlement_copypaste(
        enabled=True,
        donor_root=os.environ.get("COPYPASTE_DONOR_ROOT", "data/biodiversity_split/train"),
        prob=float(os.environ.get("COPYPASTE_PROB", "0.5")),
        n_donors=int(os.environ.get("COPYPASTE_NDONORS", "1")),
        targeted=False,
    )
    print("[Stage4-selfdistil] Settlement copy-paste ENABLED (matching shipped copy-paste recipe).")

# ====================== Datasets (IDENTICAL to stage4_kd) ======================
train_dataset = BiodiversityTrainDataset(data_root="data/biodiversity_split/train",
                                         transform=train_aug_random)
val_dataset = BiodiversityValDataset(data_root="data/biodiversity_split/val", transform=val_aug)
test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

# ====================== Sampler (SHIPPED recipe = clsbal; A0 is dead) ======================
weights_path_tsv = Path(os.environ.get("SAMPLER_TSV", str(repo_root / "artifacts" / "sampler_weights_clsbal.tsv")))
if not weights_path_tsv.exists():
    raise FileNotFoundError(f"Missing sampler weights: {weights_path_tsv}")

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
    sample_weights.append(w if w is not None else 1.0)
    missing += int(w is None)

print(f"[Stage4-selfdistil] sampler {weights_path_tsv.name}: {len(id_to_weight)} ids, "
      f"missing={missing}/{len(train_dataset.img_ids)}")
if missing > 0:
    raise RuntimeError(f"[Stage4-selfdistil] {missing} train ids have no sampling weight.")

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=2646, replacement=True)

# ====================== Loaders (IDENTICAL to stage4_kd) ======================
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4,
                          pin_memory=True, sampler=sampler, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=4, shuffle=False,
                        pin_memory=True, drop_last=False)

# ====================== Optimiser / scheduler (IDENTICAL to stage4_kd) ======================
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)
