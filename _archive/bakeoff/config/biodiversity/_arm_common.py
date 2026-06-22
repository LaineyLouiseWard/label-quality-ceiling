"""
Shared builder for the Stage-3 minority bake-off arms (handoff
docs/BAKEOFF_EXPERIMENT_HANDOFF_2026-06-19.md).

Every arm is a SIBLING branch from the frozen Stage 2b checkpoint (not warm-started from another
arm). They differ ONLY by three composable levers, each behind a flag, so the marginal deltas are
clean:

  Lever 2 (per-class sampler) — ON for every arm (A1..A4). Combines the per-class raw weights
      w4/w5 from artifacts/sampler_weights_perclass.tsv ONCE (s4*w4 + s5*w5, clip 5-95, uniform-mix
      alpha=0.5). s4=s5=1 in triage (Step 1); raise s4 only in Step 2.
  Lever 3 (recall-weighted SoftCE) — recall=True for A2/A4. RecallCrossEntropyLoss replaces the
      SoftCE term in the JointLoss; Dice is unchanged so A2-A1 isolates the CE weighting. momentum
      0.9 (EMA, default) or 0.0 (Tian-exact control, A2 only).
  Lever 1 (Settlement copy-paste) — copy_paste=True for A3/A4. Hard-mask composite onto Background.

Fixed across all arms (fair ablation, same budget): lr 3e-4 / backbone 3e-5 / wd 2.5e-4,
max_epoch 45, ignore_index=0, num_samples=2646, monitor val_mIoU. Branch point:
model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from geoseg.losses import JointLoss, SoftCrossEntropyLoss, DiceLoss, RecallCrossEntropyLoss
from geoseg.datasets.biodiversity_dataset import (
    CLASSES,
    BiodiversityTrainDataset,
    BiodiversityValDataset,
    BiodiversityTestDataset,
    val_aug,
    configure_settlement_copypaste,
)
from geoseg.datasets.per_class_sampler import load_per_class_tile_weights
from geoseg.models.ftunetformer import ft_unetformer
from geoseg.utils.optim import Lookahead, process_model_params

# ---- fixed hyperparameters (shared, do NOT tune in Step 1) ------------------
MAX_EPOCH = 45
IGNORE_INDEX = 0
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
LR = 3e-4
WEIGHT_DECAY = 2.5e-4
BACKBONE_LR = 3e-5
BACKBONE_WEIGHT_DECAY = 2.5e-4
NUM_CLASSES = 6
NUM_SAMPLES = 1846  # 2026-06-22: was 2646 (a STALE former-replicated-train-set size giving ~43% extra
# steps/epoch, confounding the sampler effect). 1846 = current train-set size, step-matched to the
# non-sampler stages (matches shipped stage3_clsbal/stage_sampler_only which use len(train_dataset)).
# NOTE: only the RETIRED stage3_armA* bake-off configs use this; the 10-seed final run does NOT.
PRETRAINED_CKPT_PATH = (
    "model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt"
)
PER_CLASS_TSV = "artifacts/sampler_weights_perclass.tsv"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next((p for p in here.parents if (p / "artifacts").exists()), here.parents[2])


def build_arm(
    weights_name: str,
    *,
    recall: bool = False,
    recall_momentum: float = 0.9,
    copy_paste: bool = False,
    s4: float = 1.0,
    s5: float = 1.0,
    alpha_mix: float = 0.5,
    clip_lo: float = 5.0,
    clip_hi: float = 95.0,
    copy_paste_prob: float = 0.5,
    copy_paste_n_donors: int = 1,
) -> dict:
    classes = CLASSES

    # ---- model (fresh init; the Stage 2b weights are loaded by train_supervision) ----
    net = ft_unetformer(
        pretrained=False, weight_path=None, num_classes=NUM_CLASSES, decoder_channels=256
    )

    # ---- loss (Lever 3) ----
    if recall:
        ce = RecallCrossEntropyLoss(
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            smooth_factor=0.05,
            momentum=recall_momentum,
        )
    else:
        ce = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=IGNORE_INDEX)
    loss = JointLoss(ce, DiceLoss(smooth=0.05, ignore_index=IGNORE_INDEX), 1.0, 1.0)

    # ---- copy-paste (Lever 1) — module-level state, set per run ----
    configure_settlement_copypaste(
        enabled=copy_paste,
        donor_root="data/biodiversity_split/train",
        prob=copy_paste_prob,
        n_donors=copy_paste_n_donors,
    )

    # ---- datasets ----
    train_dataset = BiodiversityTrainDataset(
        data_root="data/biodiversity_split/train", transform=None
    )
    val_dataset = BiodiversityValDataset(
        data_root="data/biodiversity_split/val", transform=val_aug
    )
    test_dataset = BiodiversityTestDataset(data_root="data/biodiversity_split/test")

    # ---- per-class sampler (Lever 2) ----
    tsv = _repo_root() / PER_CLASS_TSV
    if not tsv.exists():
        raise FileNotFoundError(
            f"Missing per-class sampler TSV: {tsv}\n"
            "Build it once with: python scripts/data_prep/build_sampler_weights.py "
            "--per_class --ckpt <stage2b_ckpt> --out artifacts/sampler_weights_perclass.tsv"
        )
    weights, missing, _ = load_per_class_tile_weights(
        tsv, train_dataset.img_ids,
        s4=s4, s5=s5, alpha_mix=alpha_mix, clip_lo=clip_lo, clip_hi=clip_hi,
    )
    if missing > 0:
        raise RuntimeError(
            f"[{weights_name}] {missing}/{len(train_dataset.img_ids)} train ids have no per-class "
            "sampling weight (ID alignment broken). Refusing to train with mis-weighted sampling."
        )
    print(f"[{weights_name}] per-class sampler loaded ({len(weights)} tiles, missing={missing}). "
          f"recall={recall} momentum={recall_momentum if recall else '-'} copy_paste={copy_paste} "
          f"s4={s4} s5={s5}")
    sampler = WeightedRandomSampler(weights=weights, num_samples=NUM_SAMPLES, replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, pin_memory=True,
        sampler=sampler, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, num_workers=4, shuffle=False,
        pin_memory=True, drop_last=False,
    )

    # ---- optimiser / scheduler (identical to A0) ----
    layerwise_params = {"backbone.*": dict(lr=BACKBONE_LR, weight_decay=BACKBONE_WEIGHT_DECAY)}
    net_params = process_model_params(net, layerwise_params=layerwise_params)
    base_optimizer = torch.optim.AdamW(net_params, lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = Lookahead(base_optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCH, eta_min=0
    )

    weights_path = f"model_weights/biodiversity/{weights_name}"
    return dict(
        max_epoch=MAX_EPOCH,
        ignore_index=IGNORE_INDEX,
        train_batch_size=TRAIN_BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        backbone_lr=BACKBONE_LR,
        backbone_weight_decay=BACKBONE_WEIGHT_DECAY,
        num_classes=NUM_CLASSES,
        classes=classes,
        weights_name=weights_name,
        weights_path=weights_path,
        test_weights_name=weights_name,
        log_name=f"biodiversity/{weights_name}",
        monitor="val_mIoU",
        monitor_mode="max",
        save_top_k=1,
        save_last=False,
        check_val_every_n_epoch=1,
        gpus="auto",
        pretrained_ckpt_path=PRETRAINED_CKPT_PATH,
        resume_ckpt_path=None,
        net=net,
        loss=loss,
        use_aux_loss=False,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        base_optimizer=base_optimizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
