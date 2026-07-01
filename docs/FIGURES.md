# Figures

Every manuscript figure is generated from a script in `scripts/figures/` or `scripts/analysis/`
and written to `figures/`. Script and output names are descriptive and stable; the printed figure
numbers are assigned by LaTeX, so the table below maps by content rather than by number.

## Build

```bash
python scripts/figures/build_all_figures.py --device cuda
```

This renders the core figure set into `figures/`. The TikZ figures are compiled with `pdflatex`
and copied in; the matplotlib figures render directly and use `text.usetex` (Latin Modern /
Computer Modern), so a LaTeX toolchain is required — see the figure prerequisites in
[RUNBOOK.md](RUNBOOK.md), section E.

The uncertainty and boundary figures are produced by their scripts under `scripts/analysis/`,
because they depend on saved per-tile evaluation outputs rather than the raw imagery.

## Map

| Figure content | Source script | Output |
|----------------|---------------|--------|
| Staged pipeline / factorial design | `scripts/figures/workflow_pipeline.tex`, `factorial_design.tex` | `figures/workflow_pipeline.pdf`, `factorial_design.pdf` |
| Two-axes mitigation schematic | `scripts/figures/mitigation_axes.tex` | `figures/mitigation_axes.pdf` |
| OpenEarthMap ↔ Biodiversity taxonomy mapping | `scripts/figures/oem_mapping.tex` | `figures/oem_mapping.pdf` |
| Study area | `scripts/figures/study_area.py` | `figures/study_area.pdf` |
| Dataset class-distribution comparison | `scripts/figures/class_distributions.py` | `figures/class_distributions.pdf` |
| Sampler example tiles | `scripts/figures/sampler_tiles.py` | `figures/sampler_tiles.pdf` |
| Ablation qualitative comparison | `scripts/figures/ablation_qualitative.py` | `figures/ablation_qualitative.pdf` |
| Confusion matrices | `scripts/figures/confusion_matrices.py` | `figures/confusion_matrices.pdf` |
| Per-class factorial main effects | `scripts/figures/factorial_effects.py` | `figures/factorial_effects.pdf` |
| Frequency vs difficulty | `scripts/figures/frequency_vs_difficulty.py` | `figures/frequency_vs_difficulty.pdf` |
| Reliability / ECE | `scripts/figures/reliability_ece.py` | `figures/reliability_ece.pdf` |
| Uncertainty quality | `scripts/figures/uncertainty_quality.py` | `figures/uncertainty_quality.pdf` |
| Boundary-limited error | `scripts/figures/boundary_limited_error.py` | `figures/boundary_limited_error.pdf` |
| Uncertainty overlay | `scripts/analysis/draft_boundary_overlay.py` | `figures/uncertainty_overlay.pdf` |
| Class-pair boundary matrix | `scripts/analysis/class_pair_boundary.py` | `figures/class_pair_boundary.pdf` |

Per-figure inputs (data paths, checkpoints, evaluation outputs) are documented stage-by-stage in
[RUNBOOK.md](RUNBOOK.md). Figures that draw on the proprietary Biodiversity imagery cannot be
regenerated without licensed access to that dataset — see the data-availability note in the
[README](../README.md).
