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
# Full reproducibility pipeline — 4-stage, no-replication ablation
#
#   Stage 1  baseline
#   Stage 2  OEM transfer        (2a pre-train on Bio+OEM -> 2b finetune on Bio)
#   Stage 3  hard x minority sampler
#   Stage 4  knowledge distillation (KD-B, grounded teacher->target mapping)
#
# The teacher is built UPSTREAM of the student lineage (teacher -> confusion ->
# grounded OEM relabel -> student), because the OEM->student mappings are derived
# from the teacher's measured confusion (see docs/KD_MAPPING_GROUNDING.md).
#
# Usage:
#   bash RUNBOOK.sh                      # run everything from A0
#   bash RUNBOOK.sh --from B1            # resume from Stage 1 training onward
#   RUN_NULL_CONTROLS=1 bash RUNBOOK.sh  # ALSO run the N3/N4 attribution null controls
#   SEED=1 bash RUNBOOK.sh --from B1     # student lineage at seed 1 (teacher stays fixed at 42)
#
# Valid stages: A0 (taxonomy check), A1-A10 (data prep + teacher build),
#               B1-B6 (student training), N3-N4 (optional null controls),
#               C1-C4 (evaluation), D (analyses), E (figures)
#
# Overwrite flags:  --overwrite (data-prep)   --force (training/eval/export)
# ====================================================================

# ---- Canonical paths ----
BIO_RAW=data/biodiversity_raw
OEM_RAW=data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD
TEACHER_PTH=pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth

# ---- Parse --from argument ----
FROM_STAGE="A0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM_STAGE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Ordered list of all stages
STAGES=(A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 B1 B2 B3 B4 B5 B6 N3 N4 C1 C2 C3 C4 D E)

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
echo " PIPELINE -- running from stage $FROM_STAGE onward  (SEED=${SEED:-42})"
echo "================================================================"
echo ""

# ======================== A0. PRE-FLIGHT CHECK =======================

if run_stage A0; then
  echo "[A0] Verifying taxonomy consistency (class orders / grounded OEM->student mappings)"
  # Aborts the whole run (set -e) if any class order/index has drifted from geoseg/taxonomy.py,
  # or if the grounded pre-train map != argmax(teacher confusion) when the confusion artifact exists.
  PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
fi

# ======================== A. DATA PREP + TEACHER BUILD ===============
# Order reflects the dependency teacher -> confusion -> grounded mappings -> OEM relabel -> student.

if run_stage A1; then
  echo "[A1] Splitting Biodiversity into train / val / test"
  PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
    --in-root  "$BIO_RAW" \
    --out-root data/biodiversity_split \
    --seed 42 --mode copy --overwrite
fi

if run_stage A2; then
  require_nonempty data/biodiversity_split/train/masks A1
  echo "[A2] Identifying minority-rich tiles (for the D-stage sampler-uplift analysis)"
  PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
    --data-root data/biodiversity_split/train \
    --out       artifacts/train_augmentation_list.json \
    --overwrite
fi

if run_stage A3; then
  echo "[A3] Filtering OEM (pre-mapping, rural tiles only)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_filtered \
    --overwrite
fi

if run_stage A4; then
  require_nonempty "$OEM_RAW" A3
  echo "[A4] Preparing OEM teacher training split (FULL OEM, native 9-class taxonomy)"
  # Teacher trains on the FULL OEM (~3,500 tiles), NOT the rural-filtered subset:
  # the rural filter strips settlement-rich tiles, which would weaken the very
  # minority-class signal KD injects. Native labels 0..8 are preserved.
  PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
    --raw-root "$OEM_RAW" \
    --out-root data/openearthmap_teacher \
    --official-split \
    --overwrite
fi

if run_stage A5; then
  require_nonempty data/openearthmap_teacher/train/images A4
  echo "[A5] Training OEM teacher (seed fixed at 42 — build-once, seed-invariant artifact)"
  # The teacher is held FIXED across the seed campaign (like the data), so it is NOT reseeded.
  PYTHONPATH=. python -m train.train_teacher \
    -c config/teacher/unet_oem.py --force
fi

if run_stage A6; then
  require_file model_weights/teacher/teacher.ckpt A5
  echo "[A6] Exporting teacher checkpoint + verifying native-A output channels"
  PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
    --ckpt model_weights/teacher/teacher.ckpt \
    --out  "$TEACHER_PTH" \
    --force
  # Aborts the run (set -e) if the teacher is not native-A — e.g. a stale 6-class checkpoint.
  PYTHONPATH=. python scripts/verify_teacher_channels.py \
    --ckpt "$TEACHER_PTH" \
    --data-root data/openearthmap_teacher/val
fi

if run_stage A7; then
  require_file "$TEACHER_PTH" A6
  require_nonempty data/biodiversity_split/train/masks A1
  echo "[A7] Measuring teacher->GT confusion on the training set (grounds the OEM->student mappings)"
  # Writes artifacts/teacher_oem_gt_confusion.npz (committed; the grounded pre-train map in
  # taxonomy.py is its argmax and the campaign KD map is its row-normalised soft form -- A0 asserts this).
  PYTHONPATH=. python scripts/analysis/teacher_oem_to_gt_confusion.py
fi

if run_stage A8; then
  require_nonempty data/openearthmap_filtered/masks A3
  echo "[A8] Relabelling OEM to the 6-class taxonomy (grounded argmax mapping)"
  PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
    --in-root  data/openearthmap_filtered \
    --out-root data/openearthmap_relabelled \
    --overwrite
fi

if run_stage A9; then
  require_nonempty data/openearthmap_relabelled/masks A8
  echo "[A9] Filtering OEM (post-mapping, settlement-dominant removal)"
  PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
    --in-root  data/openearthmap_relabelled \
    --out-root data/openearthmap_relabelled_filtered \
    --overwrite
fi

if run_stage A10; then
  require_nonempty data/biodiversity_split/train/images A1
  require_nonempty data/openearthmap_relabelled_filtered/masks A9
  echo "[A10] Creating combined Biodiversity + OEM dataset (Stage 2a pre-training pool)"
  PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
    --bio-root data/biodiversity_split \
    --oem-root data/openearthmap_relabelled_filtered \
    --out-root data/biodiversity_oem_combined \
    --overwrite
fi

# ======================== B. STUDENT LINEAGE =========================
# Seed-varying. Honours $SEED (default 42) in train_supervision / train_kd.

if run_stage B1; then
  require_nonempty data/biodiversity_split/train/images A1
  echo "[B1] Stage 1: Baseline"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage1_baseline.py --force
fi

if run_stage B2; then
  require_nonempty data/biodiversity_oem_combined/train/images A10
  echo "[B2] Stage 2a: OEM pre-training (combined Bio + OEM)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage2a_oem_pretrain.py --force
fi

if run_stage B3; then
  require_file model_weights/biodiversity/stage2a_oem_pretrain/stage2a_oem_pretrain.ckpt B2
  echo "[B3] Stage 2b: OEM-transfer finetune on Biodiversity (init from 2a)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage2b_oem_finetune.py --force
fi

if run_stage B4; then
  require_file model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt B3
  require_nonempty data/biodiversity_split/train/images A1
  echo "[B4] Building hard x minority sampler weights (from the Stage 2b checkpoint)"
  PYTHONPATH=. python scripts/data_prep/build_sampler_weights.py \
    --ckpt      model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt \
    --out       artifacts/sampler_weights.tsv \
    --data_root data/biodiversity_split/train \
    --batch_size 2 --num_workers 4 \
    --force
fi

if run_stage B5; then
  require_file model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt B3
  require_file artifacts/sampler_weights.tsv B4
  echo "[B5] Stage 3: Hard x minority sampling"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage3_sampler.py --force
fi

if run_stage B6; then
  require_file model_weights/biodiversity/stage3_sampler/stage3_sampler.ckpt B5
  require_file "$TEACHER_PTH" A6
  echo "[B6] Re-verifying teacher is native-A before KD (guards --from B6 resumes / stale .pth)"
  # Hard gate: aborts (set -e) rather than silently distilling from a stale/6-class teacher.
  PYTHONPATH=. python scripts/verify_teacher_channels.py \
    --ckpt "$TEACHER_PTH" \
    --data-root data/openearthmap_teacher/val
  echo "[B6] Stage 4: Knowledge distillation (KD-B, grounded mapping)"
  PYTHONPATH=. python -m train.train_kd \
    -c config/biodiversity/stage4_kd.py --force
fi

# ============= N. NULL CONTROLS (optional; export RUN_NULL_CONTROLS=1) ===============
# Off by default. Two attribution controls, each differing from its parent stage in EXACTLY one
# component, isolating the named mechanism from the +45-epoch / draw-count effect:
#   N3 = Stage 2b continued WITH uniform draws (no weighting)  ->  Stage3 - N3 = the sampler's effect
#   N4 = Stage 3 continued WITHOUT KD                          ->  Stage4 - N4 = KD's effect
# (N4 is the MANDATORY per-seed control in the 5-seed campaign.) Both run the same 45 epochs and
# 2646 draws/epoch as their parent, and are evaluated automatically by C1.

if run_stage N3 && [ "${RUN_NULL_CONTROLS:-0}" = "1" ]; then
  require_file model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt B3
  echo "[N3] Null control: Stage 2b continued WITHOUT the hard x minority weighting (uniform draws)"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage3null_nosampler.py --force
fi

if run_stage N4 && [ "${RUN_NULL_CONTROLS:-0}" = "1" ]; then
  require_file model_weights/biodiversity/stage3_sampler/stage3_sampler.ckpt B5
  echo "[N4] Null control: Stage 3 continued WITHOUT knowledge distillation"
  PYTHONPATH=. python -m train.train_supervision \
    -c config/biodiversity/stage4null_nokd.py --force
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
  require_file model_weights/biodiversity/stage4_kd/stage4_kd.ckpt B6
  require_nonempty data/biodiversity_split/test/images A1
  echo "[C2] Evaluating held-out test set (Stage 1 baseline + Stage 4 final; intermediate stages not on test)"
  # Baseline AND final on the test split — C4 (export_final_test_table) needs both metrics.json files.
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --split test \
    --base-dir model_weights/biodiversity/stage1_baseline \
    --data-root data/biodiversity_split/test \
    --out-dir evaluation/evaluation_results/test \
    --force
  PYTHONPATH=. python evaluation/compute_metrics.py \
    --split test \
    --base-dir model_weights/biodiversity/stage4_kd \
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
  require_nonempty model_weights/biodiversity B6   # fail fast before running a1-a6
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
