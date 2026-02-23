#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --------------------------------------------------------------------
# Updated training pipeline (new stage numbering)
#
# Assumptions:
# - run from repo root
# - conda env activated
# - data folders already exist
# - artifacts/ is writable
#
# Stages (as per your new setup):
#   Stage 1  : stage1_baseline.py
#   Stage 2  : stage2_replication.py
#   Stage 3a : stage3a_pretrain.py          (OEM+Biodiversity combined pretrain)
#   Stage 3b : stage3b_finetune.py          (Biodiversity train_rep finetune, init from 3a)
#   Build 4  : scripts/build_stage4_weights.py  (mining weights from stage3b ckpt)
#   Stage 4  : stage4_sampling.py           (Biodiversity train_rep, init from 3b, sampler only change)
#
# Optional later:
#   Stage 5  : stage5_kd.py + train_kd      (KD consolidation)
# --------------------------------------------------------------------

# Helpful: clean stale pyc so greps/tests behave
#echo "Cleaning __pycache__ / .pyc ..."
#find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
#find . -type f -name "*.pyc" -delete || true

# -----------------------
# Stage 1 — Baseline
# -----------------------
echo "Running Stage 1: Baseline (Biodiversity only)"
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage1_baseline.py

# -----------------------
# Stage 2 — Replication
# -----------------------
echo "Running Stage 2: Replication (train_rep)"
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2_replication.py

# -----------------------
# Stage 3a — OEM pretrain
# -----------------------
echo "Running Stage 3a: OEM Pretraining (combined OEM+Biodiversity)"
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3a_pretrain.py

# -----------------------
# Stage 3b — Finetune (init from 3a)
# -----------------------
echo "Running Stage 3b: Finetune on Biodiversity (train_rep) init from Stage 3a"
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3b_finetune.py

# -----------------------
# Build Stage 4 sampling weights (mined from Stage 3b ckpt)
# -----------------------
STAGE3B_CKPT="model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt"
OUT_WEIGHTS="artifacts/stage4_sampling_weights.tsv"

echo "Building Stage 4 sampling weights from: ${STAGE3B_CKPT}"
PYTHONPATH=. python scripts/build_stage4_weights.py \
  --ckpt "${STAGE3B_CKPT}" \
  --out "${OUT_WEIGHTS}" \
  --data_root "data/biodiversity_split/train_rep" \
  --batch_size 2 \
  --num_workers 4

# -----------------------
# Stage 4 — Sampling (init from 3b, sampler only change)
# -----------------------
echo "Running Stage 4: Hard×Minority sampling on top of Stage 3b"
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4_sampling.py

# ====================================================================
# OPTIONAL: Stage 5 — KD consolidation (only if/when you want KD)
# ====================================================================
# Notes:
# - train_kd will call config.loss(student_logits, mask, teacher_logits)
# - teacher weights must exist at:
#     pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
#
# If you need to (re)train/export teacher, uncomment the block below.
# ====================================================================

# echo "Running OPTIONAL Step: Train OEM Teacher"
# PYTHONPATH=. python -m train.train_teacher -c config/teacher/unet_oem.py
#
# echo "Running OPTIONAL Step: Export teacher checkpoint"
# PYTHONPATH=. python -m scripts.export_teacher_checkpoint \
#   --ckpt model_weights/teacher/teacher.ckpt \
#   --out pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
#
# echo "Running OPTIONAL Stage 5: KD consolidation"
# PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage5_kd.py

# -----------------------
# Step — Train OEM teacher (OEM-only, native taxonomy)
# -----------------------
echo "Running Step: Train OEM Teacher (OEM-only, native taxonomy)"
PYTHONPATH=. python -m train.train_teacher -c config/teacher/unet_oem.py

# -----------------------
# Step — Export teacher checkpoint to plain .pth for KD
# -----------------------
echo "Running Step: Export Teacher Checkpoint -> pretrain_weights/*.pth"
PYTHONPATH=. python -m scripts.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

# -----------------------
# Stage 6 — KD (student starts from Stage 4)
# -----------------------
echo "Running Stage 6: KD on top of Stage 4 sampling"
PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage6_kd.py


echo "DONE: pipeline finished."
