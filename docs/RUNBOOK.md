# Reproducibility Runbook

From-scratch instructions for reproducing all results in
*Ward et al., Remote Sensing 2026*.

All commands assume you are in the repository root with the
`ClassImbalance` conda environment activated.

```bash
conda env create -f environment.yaml
conda activate ClassImbalance
```

---

## Prerequisites

| Asset | Expected location | Source |
|-------|-------------------|--------|
| Biodiversity imagery + masks | `data/biodiversity_raw/` | ODOS Technologies (licensed) |
| OpenEarthMap tiles | `data/openearthmap_raw/` | [open-earth-map.org](https://open-earth-map.org) |

The Biodiversity dataset is not publicly redistributable.
Users with licensed access should place raw files before proceeding.

---

## A. Data preparation

Steps A1--A7 transform raw datasets into the splits, replicated sets,
and combined datasets consumed by training.

### A1. Split Biodiversity into train / val / test

```bash
PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
  --in-root  data/biodiversity_raw \
  --out-root data/biodiversity_split \
  --seed 42 --mode copy
```

**Input:** `data/biodiversity_raw/{images,masks}/`
**Output:** `data/biodiversity_split/{train,val,test}/{images,masks}/`

### A2. Identify minority-rich tiles

```bash
PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
  --data-root data/biodiversity_split/train \
  --out       artifacts/train_augmentation_list.json
```

**Input:** `data/biodiversity_split/train/masks/`
**Output:** `artifacts/train_augmentation_list.json`

### A3. Replicate minority samples (create train_rep)

```bash
PYTHONPATH=. python scripts/data_prep/replicate_minority_samples.py \
  --data-root          data/biodiversity_split/train \
  --augmentation-list  artifacts/train_augmentation_list.json \
  --out-root           data/biodiversity_split/train_rep
```

**Input:** `data/biodiversity_split/train/`, `artifacts/train_augmentation_list.json`
**Output:** `data/biodiversity_split/train_rep/{images,masks}/`

### A4. Filter OEM (pre-mapping, rural tiles only)

```bash
PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
  --raw-root data/openearthmap_raw \
  --out-root data/openearthmap_filtered
```

**Input:** `data/openearthmap_raw/<region>/{images,labels}/`
**Output:** `data/openearthmap_filtered/{images,masks}/`

### A5. Relabel OEM to 6-class taxonomy

```bash
PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
  --in-root  data/openearthmap_filtered \
  --out-root data/openearthmap_relabelled
```

**Input:** `data/openearthmap_filtered/{images,masks}/`
**Output:** `data/openearthmap_relabelled/{images,masks}/` (PNG, 6-class IDs)

### A6. Filter OEM (post-mapping, remove settlement-dominant)

```bash
PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
  --in-root  data/openearthmap_relabelled \
  --out-root data/openearthmap_relabelled_filtered
```

**Input:** `data/openearthmap_relabelled/`
**Output:** `data/openearthmap_relabelled_filtered/{images,masks}/`

### A7. Create combined Biodiversity + OEM dataset (for Stage 3a)

```bash
PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
  --bio-root data/biodiversity_split \
  --oem-root data/openearthmap_relabelled_filtered \
  --out-root data/biodiversity_oem_combined
```

**Input:** `data/biodiversity_split/`, `data/openearthmap_relabelled_filtered/`
**Output:** `data/biodiversity_oem_combined/{train,val,test}/{images,masks}/`

### A8. Prepare OEM teacher training split

```bash
PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
  --raw-root data/openearthmap_relabelled_filtered \
  --out-root data/openearthmap_teacher \
  --seed 42
```

**Input:** `data/openearthmap_relabelled_filtered/`
**Output:** `data/openearthmap_teacher/{train,val}/{images,masks}/`

---

## B. Training (all stages)

Stages run sequentially; each depends on the checkpoint from the
previous stage. The full pipeline can also be run via:

```bash
bash train_pipeline.sh
```

### B1. Stage 1 -- Baseline

```bash
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage1_baseline.py
```

**Data:** `data/biodiversity_split/train/`
**Output:** `model_weights/biodiversity/stage1_baseline/stage1_baseline.ckpt`

### B2. Stage 2 -- Minority replication

```bash
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage2_replication.py
```

**Data:** `data/biodiversity_split/train_rep/`
**Output:** `model_weights/biodiversity/stage2_replication/stage2_replication.ckpt`

### B3. Stage 3a -- OEM pre-training

```bash
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage3a_pretrain.py
```

**Data:** `data/biodiversity_oem_combined/train/`
**Output:** `model_weights/biodiversity/stage3a_pretrain/stage3a_pretrain.ckpt`

### B4. Stage 3b -- Fine-tune (init from 3a)

```bash
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage3b_finetune.py
```

**Data:** `data/biodiversity_split/train_rep/`
**Requires:** Stage 3a checkpoint
**Output:** `model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt`

### B5. Build Stage 4 sampling weights

This is an offline step between training stages, not a training run.

```bash
PYTHONPATH=. python scripts/data_prep/build_stage4_weights.py \
  --ckpt      model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt \
  --out       artifacts/stage4_sampling_weights.tsv \
  --data_root data/biodiversity_split/train_rep \
  --batch_size 2 --num_workers 4
```

**Requires:** Stage 3b checkpoint
**Output:** `artifacts/stage4_sampling_weights.tsv` (1,846 tiles)

### B6. Stage 4 -- Hard x minority sampling

```bash
PYTHONPATH=. python -m train.train_supervision \
  -c config/biodiversity/stage4_sampling.py
```

**Data:** `data/biodiversity_split/train_rep/`
**Requires:** Stage 3b checkpoint, `artifacts/stage4_sampling_weights.tsv`
**Output:** `model_weights/biodiversity/stage4_sampling/stage4_sampling.ckpt`

### B7. Train OEM teacher

```bash
PYTHONPATH=. python -m train.train_teacher \
  -c config/teacher/unet_oem.py
```

**Data:** `data/openearthmap_teacher/`
**Output:** `model_weights/teacher/teacher.ckpt`

### B8. Export teacher checkpoint

```bash
PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
```

**Requires:** Teacher checkpoint from B7
**Output:** `pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth`

### B9. Stage 5 -- Knowledge distillation

```bash
PYTHONPATH=. python -m train.train_kd \
  -c config/biodiversity/stage5_kd.py
```

**Data:** `data/biodiversity_split/train_rep/`
**Requires:** Stage 4 checkpoint, exported teacher weights (B8)
**Output:** `model_weights/biodiversity/stage5_kd/stage5_kd.ckpt`

---

## C. Evaluation

### C1. Validation set (all stage checkpoints)

```bash
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split val \
  --base-dir model_weights/biodiversity \
  --data-root data/biodiversity_split/val
```

**Output:** `evaluation/evaluation_results/val/<stage>/` containing
`metrics.json`, `confusion_matrix.{csv,npy,png}`, bar charts, and report.

### C2. Held-out test set (final model only)

```bash
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split test \
  --base-dir model_weights/biodiversity/stage5_kd \
  --data-root data/biodiversity_split/test
```

**Output:** `evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/`

### C3. Validation summary

After C1 completes, a summary is written to:

```
evaluation/evaluation_results/val/metrics_summary.txt
```

---

## D. Supplementary analyses (A1--A6)

All analyses are derived from saved evaluation outputs and
`artifacts/stage4_sampling_weights.tsv`. No retraining required.

```bash
PYTHONPATH=. python scripts/analysis/a1_minority_recall.py
PYTHONPATH=. python scripts/analysis/a2_symmetric_confusion.py
PYTHONPATH=. python scripts/analysis/a3_stage4_weight_uplift.py
PYTHONPATH=. python scripts/analysis/a4_val_test_gap.py
PYTHONPATH=. python scripts/analysis/a5_majority_stability.py
PYTHONPATH=. python scripts/analysis/a6_weight_gini.py
```

**Inputs:**
- `evaluation/evaluation_results/val/stage*/confusion_matrix.csv` (A1, A2, A5)
- `evaluation/evaluation_results/val/stage*/metrics.json` (A4, A5)
- `evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/metrics.json` (A4)
- `artifacts/stage4_sampling_weights.tsv` (A3, A6)
- `artifacts/train_augmentation_list.json` (A3)

**Output:** printed tables matching [docs/robustness_analyses.md](robustness_analyses.md).

---

## E. Paper figures

### All figures at once

```bash
python scripts/figures/build_all_figures.py --device cuda
```

### Individual figures

| Fig | Command | Key dependency |
|-----|---------|----------------|
| 1 | `python scripts/figures/Figure01.py` | `data/biodiversity_raw/` |
| 2 | `jupyter nbconvert --to notebook --execute scripts/figures/Figure02.ipynb` | `data/openearthmap_filtered/` |
| 3 | *Manually produced vector diagram (no script)* | -- |
| 4 | `python scripts/figures/Figure04.py` | `artifacts/stage4_sampling_weights.tsv` |
| 5 | `python scripts/figures/Figure05.py` | `artifacts/stage4_sampling_weights.tsv`, Stage 3b ckpt |
| 6 | `python scripts/figures/Figure06.py` | `data/openearthmap_relabelled/`, `data/openearthmap_raw/` |
| 7 | `jupyter nbconvert --to notebook --execute scripts/figures/Figure07.ipynb` | `data/biodiversity_raw/masks/` |
| 8 | `python scripts/figures/Figure08.py` | All stage checkpoints, `data/biodiversity_split/val/` |
| 9 | `python scripts/figures/Figure09.py` | `evaluation/evaluation_results/val/` (confusion matrices) |
| 10 | `python scripts/figures/Figure10.py` | `evaluation/evaluation_results/val/` (metrics.json) |
| 11 | `python scripts/figures/Figure11.py` | Stage 1 + Stage 5 checkpoints, `data/biodiversity_split/val/` |

All outputs are written to `figures/`.
See [FIGURE_MAP.md](../FIGURE_MAP.md) for full per-figure dependency lists.

---

## Dependency graph (summary)

```
Raw data (Biodiversity + OEM)
 |
 +-- A1-A3: split, analyse, replicate  -->  biodiversity_split/{train,train_rep,val,test}
 +-- A4-A6: filter + relabel OEM       -->  openearthmap_relabelled_filtered/
 +-- A7:    combine Bio + OEM          -->  biodiversity_oem_combined/
 +-- A8:    OEM teacher split           -->  openearthmap_teacher/
 |
 +-- B1-B4: Stages 1-3b training
 +-- B5:    build sampling weights     -->  artifacts/stage4_sampling_weights.tsv
 +-- B6:    Stage 4 training
 +-- B7-B8: teacher train + export     -->  pretrain_weights/*.pth
 +-- B9:    Stage 5 KD training
 |
 +-- C1-C2: evaluation                 -->  evaluation/evaluation_results/
 |
 +-- D:     supplementary analyses (A1-A6, from saved eval outputs)
 +-- E:     paper figures (from data, checkpoints, eval outputs)
```