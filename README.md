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

For a complete walkthrough covering data preparation, training, evaluation,
supplementary analyses, and figure generation, see
[RUNBOOK.md](RUNBOOK.md).

---

## Train

Full pipeline (all stages in order):

```bash
bash RUNBOOK.sh
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
python scripts/figures/Figure01.py          # Biodiversity tile examples (data only)
python scripts/figures/Figure09.py          # Confusion matrices (artifacts only)
python scripts/figures/Figure10.py          # Per-class IoU (metrics only)
python scripts/figures/Figure11.py          # KD pixel transitions
jupyter nbconvert --to notebook --execute scripts/figures/Figure02.ipynb
```

See [FIGURE_MAP.md](FIGURE_MAP.md) for per-figure dependencies and required assets.
Figure 3 is a manually produced vector diagram with no associated script.

---

## Supplementary analyses

All derived from saved evaluation outputs and sampling artefacts — no retraining required.

| File | Content |
|------|---------|
| [docs/robustness_analyses.md](docs/robustness_analyses.md) | All supplementary robustness analyses (A1–A6): minority recall, symmetric confusion, weight uplift, val–test gap, majority stability, Gini coefficient |
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

---

## Manuscript reproducibility

- `main_proposed.tex` is the current submission-ready manuscript with A1–A6 robustness insertions applied.
- `docs/robustness_analyses.md` contains the structured A1–A6 analyses (authoritative source for all inserted values).
- `docs/paper_code_consistency_audit_proposed.md` verifies all 123 numerical claims in `main_proposed.tex` against saved artefacts.
- `docs/analysis_index.md` maps every reported metric to its source artefact on disk.
- Scripts in `scripts/analysis/` (`a1_minority_recall.py` through `a6_weight_gini.py`) reproduce A1–A6 from saved evaluation outputs and sampling artefacts — no retraining required.
- Data preparation scripts (split, filter, relabel, combine, replicate, build weights, export checkpoint) are in `scripts/data_prep/`.
