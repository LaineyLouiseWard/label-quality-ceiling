# Diagnosing a Label-Quality Ceiling in Imbalanced Rural Land-Cover Segmentation

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Lightning 2.3](https://img.shields.io/badge/Lightning-2.3-792EE5?logo=lightning&logoColor=white)
![Rasterio](https://img.shields.io/badge/Rasterio-1.4-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

![Graphical abstract: satellite imagery, the FT-UNetFormer segmentation map, and the boundary-localised residual error that points to a label-quality ceiling.](assets/graphical_abstract.png)

Code accompanying the manuscript *Diagnosing a Label-Quality Ceiling in Imbalanced Rural Land-Cover Segmentation*.
Using high-resolution Pléiades satellite imagery and a fixed FT-UNetFormer, two off-the-shelf data-curation levers
(cross-dataset transfer from OpenEarthMap, taxonomy-harmonised, and a class-balanced sampler) are evaluated in a
2×2 factorial over ten seeds. Cross-dataset transfer is the dominant lever and improves every class on validation, while the
class-balanced sampler is largely redundant once transfer is applied. The residual error is diagnosed as a
label-ambiguity ceiling concentrated at class boundaries, rather than a limit of model capacity or the imbalance method.

This repository contains the complete training pipeline, evaluation scripts,
supplementary analyses, and figure generation used in the manuscript. A single
shell script (`RUNBOOK.sh`) reproduces all results end-to-end from raw data.

---

## Setup

```bash
conda env create -f environment.yaml
conda activate label-quality-ceiling
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

Individual stages (3-stage, replication-free; teacher is built upstream to ground the
OEM→student mapping, see `RUNBOOK.md`):

```bash
# Teacher (built once, fixed across seeds). Grounds the OEM->student taxonomy mapping:
PYTHONPATH=. python -m train.train_teacher -c config/teacher/unet_oem.py
PYTHONPATH=. python -m scripts.data_prep.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
PYTHONPATH=. python scripts/analysis/teacher_oem_to_gt_confusion.py   # grounds the OEM->student mappings

# Student lineage:
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage1_baseline.py     # Stage 1: baseline
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2a_oem_pretrain.py # Stage 2a: OEM pre-train
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2b_oem_finetune.py # Stage 2b: finetune on Bio

# Build the class-balanced (clsbal) sampler weights (frequency-only, Kang et al. 2020):
PYTHONPATH=. python scripts/data_prep/build_clsbal_sampler.py        # -> artifacts/sampler_weights_clsbal.tsv

PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_clsbal.py        # Stage 3: clsbal (final model)
```

---

## Evaluate

```bash
# Validation set (all checkpoints under model_weights/):
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split val \
  --base-dir model_weights/biodiversity \
  --data-root data/biodiversity_split/val

# Held-out test set (final model only, Stage 3 clsbal; add --tta for the reported TTA number):
PYTHONPATH=. python evaluation/compute_metrics.py \
  --split test \
  --base-dir model_weights/biodiversity/stage3_clsbal \
  --data-root data/biodiversity_split/test
```

Results are written to `evaluation/evaluation_results/`.

---

## Reproduce paper figures

All figures from repo root:

```bash
python scripts/figures/build_all_figures.py
```

Individual figures use descriptive script names (see [docs/FIGURES.md](docs/FIGURES.md) for the full map):

```bash
python scripts/figures/study_area.py           # study-area map
python scripts/figures/class_distributions.py  # dataset class-distribution comparison
python scripts/figures/confusion_matrices.py   # confusion matrices
python scripts/figures/factorial_effects.py    # per-class factorial main effects
```

The pipeline/factorial-design and two-axes mitigation schematics are TikZ, compiled from their
`.tex` sources in `scripts/figures/` to `figures/` by `build_all_figures.py`.

---

## Supplementary analyses

All derived from saved evaluation outputs and sampling artefacts; no retraining required.
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
| Stage 3 sampler weights (clsbal) | `artifacts/sampler_weights_clsbal.tsv` |
| Pre-computed evaluation outputs | `evaluation/evaluation_results/` |

---

## Manuscript reproducibility

- Scripts in `scripts/analysis/` (`a1_minority_recall.py` through `a6_weight_gini.py`) reproduce A1–A6 from saved evaluation outputs and sampling artefacts; no retraining required.
- Data preparation scripts (split, filter, relabel, combine, build clsbal sampler weights, export teacher checkpoint) are in `scripts/data_prep/`.
- The RGB+NIR 4-channel ablation (the near-infrared null result discussed in the paper) is kept on the `experiment/rgb-nir` branch: `config/biodiversity/stage3_clsbal_rgbnir.py` with 4-channel support in the dataset and model. It is held on a branch rather than `main` because the 4-channel data path is not backward-compatible with the RGB pipeline used for the reported results.

---

## Acknowledgements

The FT-UNetFormer implementation derives from ODOS Technologies' [GeoSeg-Biodiversity](https://github.com/odostech/GeoSeg-Biodiversity). The proprietary Biodiversity dataset was provided by ODOS Technologies under licence. The underlying Pléiades satellite imagery is © CNES 2021, distribution Airbus DS; it is proprietary and is not distributed in this repository, which shares only code and imagery-free derived outputs.

## Citation

Citation details will be added upon publication.
