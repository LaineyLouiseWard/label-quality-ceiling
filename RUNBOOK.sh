#!/bin/bash
set -euo pipefail
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Reduce CUDA fragmentation OOM on the 8 GB laptop GPU (pure stability; no result change).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fail fast (with the fix) if the active 'python' lacks PyTorch — i.e. the conda env is not active.
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: active 'python' has no PyTorch — the ClassImbalance conda env is not activated."
  echo "  Fix:  conda activate ClassImbalance   (then re-run bash RUNBOOK.sh)"
  echo "  Or:   PATH=\"\$HOME/miniconda3/envs/ClassImbalance/bin:\$PATH\" bash RUNBOOK.sh"
  exit 1
fi

# ====================================================================
# Full reproducibility pipeline
#
# Usage:
#   bash RUNBOOK.sh              # run everything from A1
#   bash RUNBOOK.sh --from B7    # resume from stage B7 onward
#   RUN_NULL_CONTROLS=1 bash RUNBOOK.sh   # ALSO run the optional N4/N5 attribution null controls
#
# Valid stages: A0 (taxonomy check), A1-A8, B1-B9, N4-N5 (optional null controls), C1-C4, D, E
#
# Overwrite flags:
#   --overwrite  (data-prep scripts)
#   --force      (training, evaluation, and export scripts)
# ====================================================================

# ---- Canonical paths ----
BIO_RAW=data/biodiversity_raw
OEM_RAW=data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD

# ---- Parse --from argument ----
FROM_STAGE="A0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM_STAGE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Ordered list of all stages
STAGES=(A0 A1 A2 A3 A4 A5 A6 A7 A8 B1 B2 B3 B4 B5 B6 B7 B8 B9 N4 N5 C1 C2 C3 C4 D E)

# Validate --from value
valid=false
for s in "${STAGES[@]}"; do
  if [[ "$s" == "$FROM_STAGE" ]]; then valid=true; break; fi
done
if ! $valid; then
  echo "ERROR: Invalid stage '$FROM_STAGE'"
  echo "  Valid stages: ${STAGES[*]}"
  exit 1
fi

# Find index of FROM_STAGE
from_idx=0
for i in "${!STAGES[@]}"; do
  if [[ "${STAGES[$i]}" == "$FROM_STAGE" ]]; then from_idx=$i; break; fi
done

# Helper: should we run this stage?
run_stage() {
  local stage="$1"
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$stage" ]]; then
      [[ $i -ge $from_idx ]] && return 0 || return 1
    fi
  done
  return 1
}

# Helper: check a directory has files (follows symlinks)
require_nonempty() {
  local dir="$1" stage="$2"
  if [ ! -d "$dir" ] || [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "ERROR: Required input '$dir' is missing or empty."
    echo "  Run stage $stage first (bash RUNBOOK.sh --from $stage)."
    exit 1
  fi
}

# Helper: check a file exists
require_file() {
  local f="$1" stage="$2"
  if [ ! -f "$f" ]; then
    echo "ERROR: Required file '$f' not found."
    echo "  Run stage $stage first (bash RUNBOOK.sh --from $stage)."
    exit 1
  fi
}

echo "================================================================"
echo " PIPELINE -- running from stage $FROM_STAGE onward"
echo "================================================================"
echo ""

# ======================== A0. PRE-FLIGHT CHECK =======================

if run_stage A0; then
  echo "[A0] Verifying taxonomy consistency (class orders / OEM->student KD mapping)"
  # Aborts the whole run (set -e) if any class order/index has drifted from geoseg/taxonomy.py.
  PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
fi

# ======================== A. DATA PREPARATION ========================

if run_stage A1; then
  echo "[A1] Splitting Biodiversity into train / val / test"
  PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
    --in-root  "$BIO_RAW" \
    --out-root data/biodiversity_split \
    --seed 42 --mode copy --overwrite
fi

if run_stage A2; then
  require_nonempty data/biodiversity_split/train/masks A1
  echo "[A2] Identifying minority-rich tiles"
  PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
    --data-root data/biodiversity_split/train \
    --out       artifacts/train_augmentation_list.json \
    --overwrite
fi

if run_stage A3; then
  require_nonempty data/biodiversity_split/train/images A1
  require_file artifacts/train_augmentation_list.json A2
  echo "[A3] Replicating minority samples"
  PYTHONPATH=. python scripts/data_prep/replicate_minority_samples.py \
    --data-root          data/biodiversity_split/train \
    --augmentation-list  artifacts/train_augmentation_list.json \
    --out-root           data/biodiversity_split/train_rep \
    --overwrite
fi

if run_stage A4; then
  echo "[A4] Filtering OEM (pre-mapping, rural tiles only)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_filtered \
    --overwrite
fi

if run_stage A5; then
  require_nonempty data/openearthmap_filtered/masks A4
  echo "[A5] Relabelling OEM to 6-class taxonomy"
  PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
    --in-root  data/openearthmap_filtered \
    --out-root data/openearthmap_relabelled \
    --overwrite
fi

if run_stage A6; then
  require_nonempty data/openearthmap_relabelled/masks A5
  echo "[A6] Filtering OEM (post-mapping, settlement-dominant)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
    --in-root  data/openearthmap_relabelled \
    --out-root data/openearthmap_relabelled_filtered \
    --overwrite
fi

if run_stage A7; then
  require_nonempty data/biodiversity_split/train/images A1
  require_nonempty data/openearthmap_relabelled_filtered/masks A6
  echo "[A7] Creating combined Biodiversity + OEM dataset"
  PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
    --bio-root data/biodiversity_split \
    --oem-root data/openearthmap_relabelled_filtered \
    --out-root data/biodiversity_oem_combined \
    --overwrite
fi

if run_stage A8; then
  require_nonempty "$OEM_RAW" A4
  echo "[A8] Preparing OEM teacher training split (FULL OEM, native 9-class taxonomy)"
  # Teacher trains on the FULL OEM (~3,500 tiles), NOT the rural-filtered subset:
  # the rural filter strips settlement-rich tiles, which would weaken the very
  # minority-class signal KD injects. Native labels 0..8 are preserved.
  PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_teacher \
    --official-split \
    --overwrite
fi

# ======================== B. TRAINING ================================

if run_stage B1; then
  require_nonempty data/biodiversity_split/train/images A1
  echo "[B1] Stage 1: Baseline"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage1_baseline.py --force
fi

if run_stage B2; then
  require_nonempty data/biodiversity_split/train_rep/images A3
  echo "[B2] Stage 2: Minority replication"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage2_replication.py --force
fi

if run_stage B3; then
  require_nonempty data/biodiversity_oem_combined/train/images A7
  echo "[B3] Stage 3a: OEM pre-training"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage3a_pretrain.py --force
fi

if run_stage B4; then
  require_file model_weights/biodiversity/stage3a_pretrain/stage3a_pretrain.ckpt B3
  echo "[B4] Stage 3b: Fine-tune (init from 3a)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage3b_finetune.py --force
fi

if run_stage B5; then
  require_file model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt B4
  require_nonempty data/biodiversity_split/train_rep/images A3
  echo "[B5] Building Stage 4 sampling weights"
  PYTHONPATH=. python scripts/data_prep/build_stage4_weights.py \
    --ckpt      model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt \
    --out       artifacts/stage4_sampling_weights.tsv \
    --data_root data/biodiversity_split/train_rep \
    --batch_size 2 --num_workers 4 \
    --force
fi

if run_stage B6; then
  require_file model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt B4
  require_file artifacts/stage4_sampling_weights.tsv B5
  echo "[B6] Stage 4: Hard x minority sampling"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage4_sampling.py --force
fi

if run_stage B7; then
  require_nonempty data/openearthmap_teacher/train/images A8
  echo "[B7] Training OEM teacher"
  PYTHONPATH=. python -m train.train_teacher \
    -c config/teacher/unet_oem.py --force
fi

if run_stage B8; then
  require_file model_weights/teacher/teacher.ckpt B7
  echo "[B8] Exporting teacher checkpoint"
  PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
    --ckpt model_weights/teacher/teacher.ckpt \
    --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth \
    --force
  echo "[B8] Verifying exported teacher outputs native-A channels (gate before KD)"
  # Aborts the run (set -e) if the teacher is not native-A — e.g. a stale 6-class checkpoint.
  PYTHONPATH=. python scripts/verify_teacher_channels.py \
    --ckpt pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth \
    --data-root data/openearthmap_teacher/val
fi

if run_stage B9; then
  require_file model_weights/biodiversity/stage4_sampling/stage4_sampling.ckpt B6
  require_file pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth B8
  echo "[B9] Re-verifying teacher is native-A before KD (guards --from B9 resumes / stale .pth)"
  # Hard gate: aborts (set -e) rather than silently distilling from a stale/6-class teacher.
  PYTHONPATH=. python scripts/verify_teacher_channels.py \
    --ckpt pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth \
    --data-root data/openearthmap_teacher/val
  echo "[B9] Stage 5: Knowledge distillation"
  PYTHONPATH=. python -m train.train_kd \
    -c config/biodiversity/stage5_kd.py --force
fi

# ============= N. NULL CONTROLS (optional; export RUN_NULL_CONTROLS=1) ===============
# Off by default. Two attribution controls that isolate each warm-start stage's NAMED mechanism
# from the extra-epochs effect (each differs from its parent stage in EXACTLY one component):
#   N4 = Stage 3b continued WITHOUT the sampler  ->  Stage4 - N4 = the sampler's effect
#   N5 = Stage 4 continued WITHOUT KD            ->  Stage5 - N5 = KD's effect
# They init from the same checkpoints as Stage 4 / Stage 5, run the same 45 epochs, and are
# evaluated automatically by C1 (compute_metrics rglobs all checkpoints under model_weights).

if run_stage N4 && [ "${RUN_NULL_CONTROLS:-0}" = "1" ]; then
  require_file model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt B4
  echo "[N4] Null control: Stage 3b continued WITHOUT the hard x minority sampler"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage4null_nosampler.py --force
fi

if run_stage N5 && [ "${RUN_NULL_CONTROLS:-0}" = "1" ]; then
  require_file model_weights/biodiversity/stage4_sampling/stage4_sampling.ckpt B6
  echo "[N5] Null control: Stage 4 continued WITHOUT knowledge distillation"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage5null_nokd.py --force
fi

# ======================== C. EVALUATION ==============================

if run_stage C1; then
  require_nonempty model_weights/biodiversity B1
  require_nonempty data/biodiversity_split/val/images A1
  echo "[C1] Evaluating validation set (all stage checkpoints)"
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --split val \
    --base-dir model_weights/biodiversity \
    --data-root data/biodiversity_split/val \
    --out-dir evaluation/evaluation_results/val \
    --force
fi

if run_stage C2; then
  require_file model_weights/biodiversity/stage1_baseline/stage1_baseline.ckpt B1
  require_file model_weights/biodiversity/stage5_kd/stage5_kd.ckpt B9
  require_nonempty data/biodiversity_split/test/images A1
  echo "[C2] Evaluating held-out test set (Stage 1 baseline + Stage 5 final; intermediate stages not on test)"
  # Baseline AND final on the test split — C4 (export_final_test_table) needs both metrics.json files.
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --split test \
    --base-dir model_weights/biodiversity/stage1_baseline \
    --data-root data/biodiversity_split/test \
    --out-dir evaluation/evaluation_results/test \
    --force
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --split test \
    --base-dir model_weights/biodiversity/stage5_kd \
    --data-root data/biodiversity_split/test \
    --out-dir evaluation/evaluation_results/test \
    --force
fi

if run_stage C3; then
  require_nonempty evaluation/evaluation_results/val C1
  echo "[C3] Aggregating validation summary"
  PYTHONPATH=. python evaluation/aggregate_metrics.py \
    --eval-root evaluation/evaluation_results/val \
    --out-file  evaluation/evaluation_results/val/metrics_summary.txt
fi

if run_stage C4; then
  require_nonempty evaluation/evaluation_results/test C2
  echo "[C4] Exporting test-set LaTeX table"
  python evaluation/export_final_test_table.py
fi

# ======================== D. ANALYSES ================================

if run_stage D; then
  require_nonempty evaluation/evaluation_results/val C1
  require_nonempty model_weights/biodiversity B9   # fail fast before running a1-a6
  echo "[D] Running supplementary analyses (A1-A6)"
  PYTHONPATH=. python scripts/analysis/a1_minority_recall.py
  PYTHONPATH=. python scripts/analysis/a2_symmetric_confusion.py
  PYTHONPATH=. python scripts/analysis/a3_stage4_weight_uplift.py
  PYTHONPATH=. python scripts/analysis/a4_val_test_gap.py
  PYTHONPATH=. python scripts/analysis/a5_majority_stability.py
  PYTHONPATH=. python scripts/analysis/a6_weight_gini.py
  echo "[D] Bootstrap confidence intervals (per-tile resampling; prerequisite for Figure 10)"
  # --force re-runs inference instead of reusing stale analysis/per_tile_cms/*.npz from a prior run.
  PYTHONPATH=. python scripts/analysis/bootstrap_metrics.py --device cuda --force
fi

# ======================== E. FIGURES =================================

if run_stage E; then
  echo "[E] Generating all paper figures"
  python scripts/figures/build_all_figures.py --device cuda
fi

echo "================================================================"
echo " DONE -- pipeline finished"
echo "================================================================"
