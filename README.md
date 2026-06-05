# Addressing Severe Class Imbalance in Rural Image Segmentation

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Lightning 2.3](https://img.shields.io/badge/Lightning-2.3-792EE5?logo=lightning&logoColor=white)
![Rasterio](https://img.shields.io/badge/Rasterio-1.4-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

Code accompanying the manuscript *Addressing Severe Class Imbalance in Rural Image Segmentation through Data Curation and Cross-Dataset Knowledge Transfer* — a staged cumulative ablation applying
minority replication, OEM pre-training, hard×minority sampling, and knowledge
distillation to high-resolution Pléiades satellite imagery with FT-UNetFormer.

This repository contains the complete training pipeline, evaluation scripts,
supplementary analyses, and figure generation used in the manuscript. A single
shell script (`RUNBOOK.sh`) reproduces all results end-to-end from raw data.

---

## Setup

```bash
conda env create -f environment.yaml
conda activate ClassImbalance
```

For a complete walkthrough covering data preparation, training, evaluation,
supplementary analyses, and figure generation, see
[RUNBOOK.md](RUNBOOK.md).

---

## Train

Full pipeline (all stages in order):

```bash
bash RUNBOOK.sh                # everything from scratch
bash RUNBOOK.sh --from B1      # resume from training onward
bash RUNBOOK.sh --from C1      # resume from evaluation onward
```

Individual stages:

```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage1_baseline.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2_replication.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3a_pretrain.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3b_finetune.py

# Build Stage 4 sampling weights from Stage 3b checkpoint:
PYTHONPATH=. python scripts/data_prep/build_stage4_weights.py \
  --ckpt model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt \
  --out  artifacts/stage4_sampling_weights.tsv

PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4_sampling.py
PYTHONPATH=. python -m train.train_teacher    -c config/teacher/unet_oem.py
PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
PYTHONPATH=. python -m train.train_kd         -c config/biodiversity/stage5_kd.py
```

---

## Evaluate

```bash
# Validation set (all checkpoints under model_weights/):
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split val \
  --base-dir model_weights/biodiversity \
  --data-root data/biodiversity_split/val

# Held-out test set (final model only):
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split test \
  --base-dir model_weights/biodiversity/stage5_kd \
  --data-root data/biodiversity_split/test
```

Results are written to `evaluation/evaluation_results/`.

---

## Reproduce paper figures

All figures from repo root:

```bash
python scripts/figures/build_all_figures.py --device cuda
```

Individual figures:

```bash
python scripts/figures/Figure03.py          # RGB tile examples (data only)
python scripts/figures/Figure04.py          # Dataset class-distribution comparison (data only)
python scripts/figures/Figure09.py          # Confusion matrices (artifacts only)
python scripts/figures/Figure10.py          # Per-class IoU (metrics only)
python scripts/figures/Figure11.py          # KD pixel transitions
```

Figures 1 (staged-pipeline flowchart) and 2 (two-axes mitigation schematic) are TikZ,
compiled from `scripts/figures/Figure01.tex` and `Figure02.tex` to `figures/`
(`build_all_figures.py` does this automatically).

---

## Supplementary analyses

All derived from saved evaluation outputs and sampling artefacts — no retraining required.
Scripts in `scripts/analysis/` (`a1_minority_recall.py` through `a6_weight_gini.py`) reproduce
robustness analyses A1–A6 (minority recall, symmetric confusion, weight uplift, val–test gap,
majority stability, Gini coefficient).

---

## Data availability

The Biodiversity dataset used in this study is proprietary and not publicly
available. It was acquired under licence from ODOS Technologies and cannot be
redistributed. The OpenEarthMap dataset is publicly available at
[https://open-earth-map.org](https://open-earth-map.org).

Pre-trained model checkpoints are not redistributed.

Users with licensed access should place files as follows:

| Asset | Location |
|-------|----------|
| Biodiversity imagery & masks | `data/biodiversity_raw/` |
| Biodiversity train/val/test split | `data/biodiversity_split/` |
| OpenEarthMap raw tiles | `data/openearthmap_raw/` |
| OEM relabelled (6-class) | `data/openearthmap_relabelled/` |
| OEM filtered subset | `data/openearthmap_filtered/` |
| Stage checkpoints | `model_weights/biodiversity/<stage>/` |
| OEM teacher weights | `pretrain_weights/` |
| Stage 4 sampling weights | `artifacts/stage4_sampling_weights.tsv` |
| Pre-computed evaluation outputs | `evaluation/evaluation_results/` |

---

## Manuscript reproducibility

- Scripts in `scripts/analysis/` (`a1_minority_recall.py` through `a6_weight_gini.py`) reproduce A1–A6 from saved evaluation outputs and sampling artefacts — no retraining required.
- Data preparation scripts (split, filter, relabel, combine, replicate, build weights, export checkpoint) are in `scripts/data_prep/`.

---

## Citation

Citation details will be added upon publication.
