# Figures

The manuscript uses fourteen figures. Each is generated from a script in `scripts/figures/`
or `scripts/analysis/` and written to `figures/`. Script and output names are descriptive and
stable; the printed figure numbers are assigned by LaTeX, so the table maps by content.

## Build

```bash
python scripts/figures/build_all_figures.py --device cuda
```

This builds the twelve figures that render directly to `figures/`: it compiles the TikZ `.tex`
figures with `pdflatex` and runs the matplotlib figures (which use `text.usetex`, Latin Modern /
Computer Modern — a LaTeX toolchain is required; see [RUNBOOK.md](RUNBOOK.md), section E).

The remaining two figures — the uncertainty overlay and the class-pair boundary matrix — are
produced by their scripts under `scripts/analysis/`, because they depend on saved per-tile
evaluation outputs. The figure PDFs are copied into the submission bundle (`manuscript/Figures/`),
which is where the manuscript reads them.

## Map

| Figure content | Source script | Output |
|----------------|---------------|--------|
| Staged pipeline | `scripts/figures/workflow_pipeline.tex` | `workflow_pipeline.pdf` |
| Two-axes mitigation schematic | `scripts/figures/mitigation_axes.tex` | `mitigation_axes.pdf` |
| OpenEarthMap ↔ Biodiversity taxonomy mapping | `scripts/figures/oem_mapping.tex` | `oem_mapping.pdf` |
| Study area | `scripts/figures/study_area.py` | `study_area.pdf` |
| Dataset class-distribution comparison | `scripts/figures/class_distributions.py` | `class_distributions.pdf` |
| Ablation qualitative comparison | `scripts/figures/ablation_qualitative.py` | `ablation_qualitative.pdf` |
| Confusion matrices | `scripts/figures/confusion_matrices.py` | `confusion_matrices.pdf` |
| Per-class factorial main effects | `scripts/figures/factorial_effects.py` | `factorial_effects.pdf` |
| Frequency vs difficulty | `scripts/figures/frequency_vs_difficulty.py` | `frequency_vs_difficulty.pdf` |
| Reliability / ECE | `scripts/figures/reliability_ece.py` | `reliability_ece.pdf` |
| Uncertainty quality | `scripts/figures/uncertainty_quality.py` | `uncertainty_quality.pdf` |
| Boundary-limited error | `scripts/figures/boundary_limited_error.py` | `boundary_limited_error.pdf` |
| Uncertainty overlay | `scripts/analysis/draft_boundary_overlay.py` | `uncertainty_overlay.pdf` |
| Class-pair boundary matrix | `scripts/analysis/class_pair_boundary.py` | `class_pair_boundary.pdf` |

The graphical abstract is built separately from `scripts/figures/graphical_abstract_panels.py`
(three raster panels) and `graphical_abstract_tikz.tex` (assembly); the final image ships with the
submission bundle.

Per-figure inputs (data paths, checkpoints, evaluation outputs) are documented stage-by-stage in
[RUNBOOK.md](RUNBOOK.md). Figures that draw on the proprietary Biodiversity imagery cannot be
regenerated without licensed access to that dataset — see the data-availability note in the
[README](../README.md).
