# Reproducibility Runbook

From-scratch instructions for reproducing all results in the manuscript
*Addressing Severe Class Imbalance in Rural Image Segmentation through Data Curation and Cross-Dataset Knowledge Transfer*.

The ablation is **4-stage and replication-free**:

| Stage | Mechanism added | Config |
|-------|-----------------|--------|
| 1 | Baseline (supervised) | `stage1_baseline.py` |
| 2 | OEM transfer (2a pre-train on Bio+OEM → 2b finetune on Bio) | `stage2a_oem_pretrain.py`, `stage2b_oem_finetune.py` |
| 3 | Hard × minority sampler | `stage3_sampler.py` |
| 4 | Knowledge distillation (KD-B, grounded mapping) | `stage4_kd.py` |

The **teacher is built upstream** of the student lineage, because the OEM→student mappings are
*derived from the teacher's measured confusion* (teacher → confusion → grounded relabel → student).
See `docs/KD_MAPPING_GROUNDING.md`.

All commands assume you are in the repository root with the `ClassImbalance` conda environment active.

```bash
conda env create -f environment.yaml
conda activate ClassImbalance
```

---

## Running the pipeline

To reproduce **everything** end-to-end in a single command:

```bash
bash RUNBOOK.sh
```

To resume from a specific stage:

```bash
bash RUNBOOK.sh --from B1   # resume from Stage 1 training onward
```

Optional attribution null controls (off by default):

```bash
RUN_NULL_CONTROLS=1 bash RUNBOOK.sh
```

Run the student lineage at a different seed (the teacher stays fixed at seed 42):

```bash
SEED=1 bash RUNBOOK.sh --from B1
```

Valid stages: `A0` (taxonomy check), `A1`–`A10` (data prep + teacher build), `B1`–`B6` (student
training), `N3`–`N4` (optional null controls), `C1`–`C4` (evaluation), `D` (analyses), `E` (figures).

**Warning:** This overwrites all derived outputs in-place — checkpoints, sampler weights, evaluation
results, and figures. Raw data (`data/biodiversity_raw/`, `data/openearthmap_raw/`) is never modified.

---

## Prerequisites

| Asset | Expected location | Source |
|-------|-------------------|--------|
| Biodiversity imagery + masks | `data/biodiversity_raw/` | ODOS Technologies (licensed) |
| OpenEarthMap tiles | `data/openearthmap_raw/` | [open-earth-map.org](https://open-earth-map.org) |

Both datasets are **required**. The pipeline aborts at startup if either is missing. The OEM raw root
must contain `OpenEarthMap/OpenEarthMap_wo_xBD/<region>/{images,labels}/`. The Biodiversity dataset is
not publicly redistributable; users with licensed access should place raw files before proceeding.

---

## A. Data preparation + teacher build (A1–A10)

The teacher (A4–A6) and its confusion (A7) come **before** the OEM relabel (A8), because the grounded
OEM→student mapping is the argmax of that confusion.

### A1. Split Biodiversity into train / val / test
```bash
PYTHONPATH=. python scripts/data_prep/split_biodiversity_dataset.py \
  --in-root data/biodiversity_raw --out-root data/biodiversity_split \
  --seed 42 --mode copy --overwrite
```
**Output:** `data/biodiversity_split/{train,val,test}/{images,masks}/`

### A2. Identify minority-rich tiles
```bash
PYTHONPATH=. python scripts/data_prep/analyze_class_distribution.py \
  --data-root data/biodiversity_split/train \
  --out artifacts/train_augmentation_list.json --overwrite
```
**Output:** `artifacts/train_augmentation_list.json` (consumed by the D-stage sampler-uplift analysis).

### A3. Filter OEM (pre-mapping, rural tiles only)
```bash
PYTHONPATH=. python scripts/data_prep/filter_oem_rural.py \
  --raw-root data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD \
  --out-root data/openearthmap_filtered --overwrite
```
**Output:** `data/openearthmap_filtered/{images,masks}/`

### A4. Prepare OEM teacher training split (full OEM, native 9-class)
```bash
PYTHONPATH=. python scripts/data_prep/prepare_oem_teacher_data.py \
  --raw-root data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD \
  --out-root data/openearthmap_teacher --official-split --overwrite
```
The teacher trains on the **full, native OEM** (labels 0–8) — NOT the rural-filtered/relabelled 6-class
set, which would train the teacher on the wrong taxonomy and silently break KD.
**Output:** `data/openearthmap_teacher/{train,val}/{images,masks}/`

### A5. Train OEM teacher (seed fixed at 42)
```bash
PYTHONPATH=. python -m train.train_teacher -c config/teacher/unet_oem.py --force
```
The teacher is a **build-once, seed-invariant** artifact (held fixed across the seed campaign), so it
is not reseeded. First run downloads ImageNet-pretrained EfficientNet-B4 weights (needs internet).
**Output:** `model_weights/teacher/teacher.ckpt`

### A6. Export teacher checkpoint (+ verify native-A channels)
```bash
PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth --force
```
**Output:** `pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth`

### A7. Measure teacher→GT confusion (grounds the mappings)
```bash
PYTHONPATH=. python scripts/analysis/teacher_oem_to_gt_confusion.py
```
**Output:** `artifacts/teacher_oem_gt_confusion.npz` (committed; the grounded pre-train map in
`geoseg/taxonomy.py` is its argmax and the KD-B map is its row-normalised soft form — A0 asserts both).

### A8. Relabel OEM to the 6-class taxonomy (grounded argmax mapping)
```bash
PYTHONPATH=. python scripts/data_prep/relabel_oem_taxonomy.py \
  --in-root data/openearthmap_filtered --out-root data/openearthmap_relabelled --overwrite
```
**Output:** `data/openearthmap_relabelled/{images,masks}/` (PNG, 6-class IDs)

### A9. Filter OEM (post-mapping, remove settlement-dominant)
```bash
PYTHONPATH=. python scripts/data_prep/filter_oem_settlement_postmap.py \
  --in-root data/openearthmap_relabelled --out-root data/openearthmap_relabelled_filtered --overwrite
```
**Output:** `data/openearthmap_relabelled_filtered/{images,masks}/`

### A10. Create combined Biodiversity + OEM dataset (Stage 2a pool)
```bash
PYTHONPATH=. python scripts/data_prep/create_biodiversity_oem_combined.py \
  --bio-root data/biodiversity_split --oem-root data/openearthmap_relabelled_filtered \
  --out-root data/biodiversity_oem_combined --overwrite
```
**Output:** `data/biodiversity_oem_combined/{train,val,test}/{images,masks}/`

---

## B. Student lineage (B1–B6)

Seed-varying; honours `$SEED` (default 42). Each stage warm-starts from the previous checkpoint.

### B1. Stage 1 — Baseline
```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage1_baseline.py --force
```
**Data:** `data/biodiversity_split/train/` · **Output:** `model_weights/biodiversity/stage1_baseline/`

### B2. Stage 2a — OEM pre-training (combined)
```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2a_oem_pretrain.py --force
```
**Data:** `data/biodiversity_oem_combined/train/` · **Output:** `model_weights/biodiversity/stage2a_oem_pretrain/`

### B3. Stage 2b — OEM-transfer finetune (init from 2a)
```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2b_oem_finetune.py --force
```
**Data:** `data/biodiversity_split/train/` · **Requires:** Stage 2a ckpt · **Output:** `model_weights/biodiversity/stage2b_oem_finetune/`

### B4. Build hard × minority sampler weights (offline)
```bash
PYTHONPATH=. python scripts/data_prep/build_sampler_weights.py \
  --ckpt      model_weights/biodiversity/stage2b_oem_finetune/stage2b_oem_finetune.ckpt \
  --out       artifacts/sampler_weights.tsv \
  --data_root data/biodiversity_split/train --batch_size 2 --num_workers 4 --force
```
**Requires:** Stage 2b ckpt · **Output:** `artifacts/sampler_weights.tsv` (base-keyed, 1,846 tiles)

### B5. Stage 3 — Hard × minority sampling
```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_sampler.py --force
```
**Data:** `data/biodiversity_split/train/` · **Requires:** Stage 2b ckpt, `sampler_weights.tsv`
**Output:** `model_weights/biodiversity/stage3_sampler/`

### B6. Stage 4 — Knowledge distillation (KD-B)
```bash
PYTHONPATH=. python -m train.train_kd -c config/biodiversity/stage4_kd.py --force
```
**Data:** `data/biodiversity_split/train/` · **Requires:** Stage 3 ckpt, exported teacher weights (A6)
**Output:** `model_weights/biodiversity/stage4_kd/` (deployed final model, paper "Stage 4")

### N3 / N4 — Attribution null controls (optional)
`RUN_NULL_CONTROLS=1`. N3 = Stage 2b + uniform draws (no weighting) → `Stage3 − N3` = sampler effect.
N4 = Stage 3 without KD → `Stage4 − N4` = KD effect (mandatory per-seed control in the campaign).

---

## C. Evaluation

### C1. Validation set (all stage checkpoints)
```bash
PYTHONPATH=. python evaluation/compute_metrics.py --split val \
  --base-dir model_weights/biodiversity --data-root data/biodiversity_split/val \
  --out-dir evaluation/evaluation_results/val --force
```
**Output:** `evaluation/evaluation_results/val/<stage>/` (metrics.json, confusion matrices, reports).

### C2. Held-out test set (baseline + final model)
```bash
PYTHONPATH=. python evaluation/compute_metrics.py --split test \
  --base-dir model_weights/biodiversity/stage1_baseline --data-root data/biodiversity_split/test \
  --out-dir evaluation/evaluation_results/test --force
PYTHONPATH=. python evaluation/compute_metrics.py --split test \
  --base-dir model_weights/biodiversity/stage4_kd --data-root data/biodiversity_split/test \
  --out-dir evaluation/evaluation_results/test --force
```
**Output:** `evaluation/evaluation_results/test/{stage1_baseline,stage4_kd}/`

### C3. Validation summary
```bash
PYTHONPATH=. python evaluation/aggregate_metrics.py \
  --eval-root evaluation/evaluation_results/val \
  --out-file  evaluation/evaluation_results/val/metrics_summary.txt
```

### C4. Export test-set LaTeX table
```bash
python evaluation/export_final_test_table.py
```
**Input:** `evaluation/evaluation_results/test/{stage1_baseline,stage4_kd}/metrics.json`
**Output:** `evaluation/evaluation_results/final_test_table.tex`

---

## D. Supplementary analyses (A1–A6)

All analyses are derived from saved evaluation outputs and `artifacts/sampler_weights.tsv`. No retraining.

```bash
PYTHONPATH=. python scripts/analysis/a1_minority_recall.py
PYTHONPATH=. python scripts/analysis/a2_symmetric_confusion.py
PYTHONPATH=. python scripts/analysis/a3_stage4_weight_uplift.py
PYTHONPATH=. python scripts/analysis/a4_val_test_gap.py
PYTHONPATH=. python scripts/analysis/a5_majority_stability.py
PYTHONPATH=. python scripts/analysis/a6_weight_gini.py
PYTHONPATH=. python scripts/analysis/bootstrap_metrics.py --device cuda --force
```

`bootstrap_metrics.py` runs per-tile resampling for the Figure 10 confidence intervals. It requires the
trained stage checkpoints under `model_weights/biodiversity/` and writes `analysis/bootstrap_results.md`
plus cached per-tile confusion matrices in `analysis/per_tile_cms/`.

**Inputs:**
- `evaluation/evaluation_results/val/stage*/confusion_matrix.csv` (A1, A2, A5)
- `evaluation/evaluation_results/val/stage*/metrics.json` (A4, A5)
- `evaluation/evaluation_results/test/stage4_kd/metrics.json` (A4)
- `artifacts/sampler_weights.tsv` (A3, A6)
- `artifacts/train_augmentation_list.json` (A3)

---

## E. Paper figures

```bash
python scripts/figures/build_all_figures.py --device cuda
```

| Fig | Command | Key dependency |
|-----|---------|----------------|
| 1 | `pdflatex scripts/figures/Figure01.tex` (staged pipeline flowchart, TikZ) | — |
| 2 | `pdflatex scripts/figures/Figure02.tex` (two-axes mitigation schematic, TikZ) | — |
| 3 | `python scripts/figures/Figure03.py` | `data/biodiversity_raw/` |
| 4 | `python scripts/figures/Figure04.py` (OpenEarthMap taxonomy-harmonisation example) | `data/openearthmap_raw/`, `data/openearthmap_relabelled/` |
| 5 | `python scripts/figures/Figure05.py` (dataset class-distribution comparison) | `data/biodiversity_raw/masks/`, `data/openearthmap_filtered/masks/` |
| 6 | `python scripts/figures/Figure06.py` (hard × minority sampling-weight distribution) | `artifacts/sampler_weights.tsv` |
| 7 | `python scripts/figures/Figure07.py` (low/high-weight example tiles) | `artifacts/sampler_weights.tsv`, Stage 2b ckpt |
| 8 | `python scripts/figures/Figure08.py` (4-stage qualitative comparison) | All stage checkpoints, `data/biodiversity_split/val/` |
| 9 | `python scripts/figures/Figure09.py` | `evaluation/evaluation_results/val/` (confusion matrices) |
| 10 | `python scripts/figures/Figure10.py` | `evaluation/evaluation_results/val/` (metrics.json) |
| 11 | `python scripts/figures/Figure11.py` | Stage 1 + Stage 4 checkpoints, `data/biodiversity_split/val/` |

(For Figures 1–2, `build_all_figures.py` compiles the `.tex` and copies the PDF into `figures/`.)
All outputs go to `figures/`. See `docs/FIGURE_MAP.md` for full per-figure dependency lists.

---

## Dependency graph (summary)

```
Raw data (Biodiversity + OEM)
 |
 +-- A1:     split Biodiversity            -->  biodiversity_split/{train,val,test}
 +-- A2:     identify minority tiles       -->  artifacts/train_augmentation_list.json
 +-- A3:     filter OEM (rural)            -->  openearthmap_filtered/
 +-- A4:     OEM teacher split (raw OEM)   -->  openearthmap_teacher/   (native 9-class)
 +-- A5-A6:  train + export teacher        -->  pretrain_weights/*.pth
 +-- A7:     teacher->GT confusion         -->  artifacts/teacher_oem_gt_confusion.npz
 +-- A8-A9:  grounded relabel + filter OEM -->  openearthmap_relabelled_filtered/
 +-- A10:    combine Bio + OEM             -->  biodiversity_oem_combined/
 |
 +-- B1:     Stage 1 baseline
 +-- B2-B3:  Stage 2 OEM transfer (2a pre-train -> 2b finetune)
 +-- B4:     build sampler weights         -->  artifacts/sampler_weights.tsv
 +-- B5:     Stage 3 sampler
 +-- B6:     Stage 4 KD-B (uses teacher .pth + grounded mapping)
 |
 +-- C1-C2:  evaluation                    -->  evaluation/evaluation_results/
 +-- D:      supplementary analyses (from saved eval outputs)
 +-- E:      paper figures
```
