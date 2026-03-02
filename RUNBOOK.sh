#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ====================================================================
# Full reproducibility pipeline (Ward et al., Remote Sensing 2026)
#
# Runs every step from RUNBOOK.md end-to-end with overwrite flags,
# producing deterministic in-place outputs.  Raw data is never modified.
#
# Assumptions:
#   - run from repo root
#   - conda env ClassImbalance activated
#   - data/biodiversity_raw/ exists
#   - data/openearthmap_raw/ exists (with OpenEarthMap/OpenEarthMap_wo_xBD/ inside)
#
# Overwrite flags:
#   --overwrite  (data-prep scripts)
#   --force      (training, evaluation, and export scripts)
# ====================================================================

# ---- Canonical paths ----
BIO_RAW=data/biodiversity_raw
OEM_RAW=data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD

echo "================================================================"
echo " FULL PIPELINE -- all derived outputs will be overwritten"
echo "================================================================"

# ======================== PREFLIGHT CHECKS ===========================

echo "[preflight] Checking required inputs..."

fail=0

if [ ! -d "$BIO_RAW/images" ] || [ ! -d "$BIO_RAW/masks" ]; then
  echo "ERROR: Biodiversity raw data not found at $BIO_RAW/{images,masks}/"
  fail=1
fi

if [ ! -d "$OEM_RAW" ]; then
  echo "ERROR: OEM raw data not found at $OEM_RAW"
  echo "  Expected: data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD/<region>/{images,labels}/"
  fail=1
else
  # Check at least one region with images/ and labels/
  region_count=$(find "$OEM_RAW" -mindepth 2 -maxdepth 2 -type d -name labels 2>/dev/null | wc -l)
  if [ "$region_count" -eq 0 ]; then
    echo "ERROR: No OEM regions with labels/ found under $OEM_RAW"
    echo "  Expected: $OEM_RAW/<region>/{images,labels}/"
    fail=1
  else
    echo "  OEM regions with labels/: $region_count"
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "ABORT: preflight checks failed. Fix the paths above and re-run."
  exit 1
fi

echo "[preflight] OK"
echo ""

# ======================== A. DATA PREPARATION ========================

echo "[A1] Splitting Biodiversity into train / val / test"
PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
  --in-root  "$BIO_RAW" \
  --out-root data/biodiversity_split \
  --seed 42 --mode copy --overwrite

echo "[A2] Identifying minority-rich tiles"
PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
  --data-root data/biodiversity_split/train \
  --out       artifacts/train_augmentation_list.json \
  --overwrite

echo "[A3] Replicating minority samples"
PYTHONPATH=. python scripts/data_prep/replicate_minority_samples.py \
  --data-root          data/biodiversity_split/train \
  --augmentation-list  artifacts/train_augmentation_list.json \
  --out-root           data/biodiversity_split/train_rep \
  --overwrite

echo "[A4] Filtering OEM (pre-mapping, rural tiles only)"
PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
  --raw-root "$OEM_RAW" \
  --out-root data/openearthmap_filtered \
  --overwrite

echo "[A5] Relabelling OEM to 6-class taxonomy"
PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
  --in-root  data/openearthmap_filtered \
  --out-root data/openearthmap_relabelled \
  --overwrite

echo "[A6] Filtering OEM (post-mapping, settlement-dominant)"
PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
  --in-root  data/openearthmap_relabelled \
  --out-root data/openearthmap_relabelled_filtered \
  --overwrite

echo "[A7] Creating combined Biodiversity + OEM dataset"
PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
  --bio-root data/biodiversity_split \
  --oem-root data/openearthmap_relabelled_filtered \
  --out-root data/biodiversity_oem_combined \
  --overwrite

echo "[A8] Preparing OEM teacher training split"
PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
  --raw-root data/openearthmap_relabelled_filtered \
  --out-root data/openearthmap_teacher \
  --seed 42 \
  --overwrite

# ======================== B. TRAINING ================================

echo "[B1] Stage 1: Baseline"
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage1_baseline.py --force

echo "[B2] Stage 2: Minority replication"
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage2_replication.py --force

echo "[B3] Stage 3a: OEM pre-training"
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage3a_pretrain.py --force

echo "[B4] Stage 3b: Fine-tune (init from 3a)"
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage3b_finetune.py --force

echo "[B5] Building Stage 4 sampling weights"
PYTHONPATH=. python scripts/data_prep/build_stage4_weights.py \
  --ckpt      model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt \
  --out       artifacts/stage4_sampling_weights.tsv \
  --data_root data/biodiversity_split/train_rep \
  --batch_size 2 --num_workers 4 \
  --force

echo "[B6] Stage 4: Hard x minority sampling"
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage4_sampling.py --force

echo "[B7] Training OEM teacher"
PYTHONPATH=. python -m train.train_teacher \
  -c config/teacher/unet_oem.py --force

echo "[B8] Exporting teacher checkpoint"
PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth \
  --force

echo "[B9] Stage 5: Knowledge distillation"
PYTHONPATH=. python -m train.train_kd \
  -c config/biodiversity/stage5_kd.py --force

# ======================== C. EVALUATION ==============================

echo "[C1] Evaluating validation set (all stage checkpoints)"
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split val \
  --base-dir model_weights/biodiversity \
  --data-root data/biodiversity_split/val \
  --out-dir evaluation/evaluation_results/val \
  --force

echo "[C2] Evaluating held-out test set (final model only)"
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split test \
  --base-dir model_weights/biodiversity/stage5_kd \
  --data-root data/biodiversity_split/test \
  --out-dir evaluation/evaluation_results/test \
  --force

echo "[C3] Aggregating validation summary"
PYTHONPATH=. python evaluation/aggregate_metrics.py \
  --eval-root evaluation/evaluation_results/val \
  --out-file  evaluation/evaluation_results/val/metrics_summary.txt

echo "[C4] Exporting test-set LaTeX table"
python evaluation/export_final_test_table.py

# ======================== D. ANALYSES ================================

echo "[D] Running supplementary analyses (A1-A6)"
PYTHONPATH=. python scripts/analysis/a1_minority_recall.py
PYTHONPATH=. python scripts/analysis/a2_symmetric_confusion.py
PYTHONPATH=. python scripts/analysis/a3_stage4_weight_uplift.py
PYTHONPATH=. python scripts/analysis/a4_val_test_gap.py
PYTHONPATH=. python scripts/analysis/a5_majority_stability.py
PYTHONPATH=. python scripts/analysis/a6_weight_gini.py

# ======================== E. FIGURES =================================

echo "[E] Generating all paper figures"
python scripts/figures/build_all_figures.py --device cuda

echo "================================================================"
echo " DONE -- full pipeline finished"
echo "================================================================"
