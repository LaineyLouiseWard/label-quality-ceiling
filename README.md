# Addressing Severe Class Imbalance in Rural Image Segmentation

Code for *Ward et al., Remote Sensing 2026* — a 5-stage cumulative ablation study
applying minority replication, OEM pre-training, hard×minority sampling, and knowledge
distillation to high-resolution Pléiades satellite imagery.

---

## Setup

```bash
conda env create -f environment.yaml
conda activate ClassImbalance
```

---

## Reproducing figures

All paper figures can be rebuilt from repo root:

```bash
python scripts/figures/build_all_figures.py --device cuda
```

Individual figures:

```bash
python scripts/figures/Figure01.py          # data only
python scripts/figures/Figure08.py          # pre-computed artifacts only
python scripts/figures/Figure09.py          # pre-computed metrics only
jupyter nbconvert --to notebook --execute scripts/figures/Figure02.ipynb
```

See [FIGURE_MAP.md](FIGURE_MAP.md) for per-figure dependencies.

---

## Repository structure

```
scripts/figures/   Figure01.py … Figure10.py + build_all_figures.py
figures/           Figure01.pdf … Figure10.pdf/png  (paper outputs)
geoseg/            Model and dataset code
config/            Training configs
evaluation/        Pre-computed metrics and confusion matrices
```

---

## Data & Checkpoint Availability

The Biodiversity dataset used in this work is not publicly redistributable due to
licence restrictions. Pre-trained model checkpoints are similarly not redistributed.

Users with licensed access to the dataset should place files in the following
locations (the scripts will raise clear errors if anything is missing):

| Asset | Location |
|-------|----------|
| Biodiversity imagery & masks | `data/biodiversity_raw/` |
| Biodiversity train/val/test split | `data/biodiversity_split/` |
| OpenEarthMap (OEM) raw tiles | `data/openearthmap_raw/` |
| OEM relabelled to 6-class taxonomy | `data/openearthmap_relabelled/` |
| OEM filtered subset | `data/openearthmap_filtered/` |
| Stage checkpoints | `model_weights/biodiversity/<stage>/` |
| OEM pre-train weights | `pretrain_weights/` |
| Stage 4 sampling weights TSV | `artifacts/stage4_sampling_weights.tsv` |
| Pre-computed evaluation artefacts | `evaluation/evaluation_results/` |
