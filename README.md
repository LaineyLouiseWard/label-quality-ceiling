# Addressing Severe Class Imbalance in Rural Image Segmentation

Code for *Ward et al., Remote Sensing 2026* — a staged cumulative ablation applying
minority replication, OEM pre-training, hard×minority sampling, and knowledge
distillation to high-resolution Pléiades satellite imagery with FT-UNetFormer.

---

## Setup

```bash
conda env create -f environment.yaml
conda activate ClassImbalance
```

---

## Train

Full pipeline (all stages in order):

```bash
bash train_pipeline.sh
```

Individual stages:

```bash
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage1_baseline.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage2_replication.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3a_pretrain.py
PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3b_finetune.py

# Build Stage 4 sampling weights from Stage 3b checkpoint:
PYTHONPATH=. python scripts/build_stage4_weights.py \
  --ckpt model_weights/biodiversity/stage3b_finetune/stage3b_finetune.ckpt \
  --out  artifacts/stage4_sampling_weights.tsv

PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage4_sampling.py
PYTHONPATH=. python -m train.train_teacher    -c config/teacher/unet_oem.py
PYTHONPATH=. python -m scripts.export_teacher_checkpoint \
  --ckpt model_weights/teacher/teacher.ckpt \
  --out  pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
PYTHONPATH=. python -m train.train_kd         -c config/biodiversity/stage6_kd.py
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
  --base-dir model_weights/biodiversity/stage6_kd \
  --data-root data/biodiversity_split/test
```

Results are written to `evaluation/evaluation_results/`.

---

## Reproduce paper figures

All figures from repo root:

```bash
python scripts/figures/build_all_figures.py --device cuda
```

Including supplementary weight-distribution figure:

```bash
python scripts/figures/build_all_figures.py --device cuda --include-supplementary
```

Individual figures:

```bash
python scripts/figures/Figure01.py          # Biodiversity tile examples (data only)
python scripts/figures/Figure08.py          # Confusion matrices (artifacts only)
python scripts/figures/Figure09.py          # Per-class IoU (metrics only)
python scripts/figures/Figure10.py          # KD pixel transitions
jupyter nbconvert --to notebook --execute scripts/figures/Figure02.ipynb
```

See [FIGURE_MAP.md](FIGURE_MAP.md) for per-figure dependencies and required assets.
Figure 3 is a manually produced vector diagram with no associated script.

---

## Supplementary analyses

All derived from saved evaluation outputs and sampling artefacts — no retraining required.

| File | Content |
|------|---------|
| [docs/majority_stability.txt](docs/majority_stability.txt) | Formalised bound on majority-class IoU decline across stage transitions (max 0.90 pp) |
| [docs/minority_recall_progression.md](docs/minority_recall_progression.md) | Per-stage recall for Settlement and Semi-natural Grassland (Stages 1–5) |
| [docs/symmetric_confusion_disclosure.tex](docs/symmetric_confusion_disclosure.tex) | Confirms minority IoU gains reflect genuine class separation (both confusion directions) |
| [docs/val_test_gap_disclosure.tex](docs/val_test_gap_disclosure.tex) | Per-class val–test IoU gap; largest for semi-natural (17.9 pp) and cropland (12.4 pp) |
| [docs/stage4_weight_uplift.md](docs/stage4_weight_uplift.md) | Minority tile sampling uplift: 1.26× (settlement), 3.08× (semi-natural) |
| [docs/stage4_weight_gini.tex](docs/stage4_weight_gini.tex) | Gini coefficient of Stage 4 weight distribution (0.36, moderate concentration) |
| [docs/paper_code_consistency_audit.md](docs/paper_code_consistency_audit.md) | Full paper–code audit; all numerical claims verified against saved outputs |

---

## Data & checkpoint availability

The Biodiversity dataset is not publicly redistributable (ODOS Technologies licence).
Pre-trained checkpoints are not redistributed.

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
