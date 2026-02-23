#!/usr/bin/env python3
"""
scripts/figures/FigureXX.py

Supplementary analysis: Stage 4 sampling weight distribution.

Loads the pre-computed Stage 4 hardness × minority-aware sampling weights
from artifacts/stage4_sampling_weights.tsv, computes summary statistics,
plots a histogram, and writes a numeric summary.

Outputs:
  figures/FigureXX.pdf
  docs/stage4_weight_summary.md

Run:
  python scripts/figures/FigureXX.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Repo root detection
# ---------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    for p in [start.resolve(), *start.resolve().parents]:
        if (p / "artifacts").is_dir() and (p / "geoseg").is_dir():
            return p
    raise FileNotFoundError(f"Could not find repo root from {start}")


repo_root = find_repo_root(Path(__file__).parent)

WEIGHTS_TSV = repo_root / "artifacts" / "stage4_sampling_weights.tsv"
OUT_PDF     = repo_root / "figures" / "FigureXX.pdf"
OUT_MD      = repo_root / "docs"    / "stage4_weight_summary.md"

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style — match existing figure scripts
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})


# ---------------------------------------------------------------------------
# Load weights
# ---------------------------------------------------------------------------
img_ids: list[str] = []
weights: list[float] = []

with open(WEIGHTS_TSV, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_id, w = line.split("\t")
        img_ids.append(img_id)
        weights.append(float(w))

w = np.array(weights, dtype=np.float64)
n = len(w)

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
stats = {
    "n":    n,
    "min":  float(np.min(w)),
    "max":  float(np.max(w)),
    "mean": float(np.mean(w)),
    "std":  float(np.std(w)),
    "median":  float(np.median(w)),
    "p25":  float(np.percentile(w, 25)),
    "p75":  float(np.percentile(w, 75)),
    "p95":  float(np.percentile(w, 95)),
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)

ax.hist(
    w,
    bins=60,
    color="#3B8DF7",   # Settlement blue from palette
    edgecolor="white",
    linewidth=0.4,
    alpha=0.85,
)

ax.axvline(stats["mean"],   color="#E05A00", linewidth=1.6, linestyle="-",  label=f'Mean = {stats["mean"]:.3f}')
ax.axvline(stats["median"], color="#222222", linewidth=1.4, linestyle="--", label=f'Median = {stats["median"]:.3f}')
ax.axvline(stats["p95"],    color="#888888", linewidth=1.2, linestyle=":",  label=f'95th pct = {stats["p95"]:.3f}')

ax.set_xlabel("Sampling weight")
ax.set_ylabel("Number of tiles")

ax.legend(frameon=True, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Write markdown summary
# ---------------------------------------------------------------------------
md_lines = [
    "# Stage 4 Sampling Weight Summary",
    "",
    f"Source: `{WEIGHTS_TSV.relative_to(repo_root)}`",
    "",
    "## Statistics",
    "",
    f"| Statistic | Value |",
    f"|-----------|-------|",
    f"| N (tiles) | {stats['n']} |",
    f"| Min       | {stats['min']:.6f} |",
    f"| Max       | {stats['max']:.6f} |",
    f"| Mean      | {stats['mean']:.6f} |",
    f"| Std       | {stats['std']:.6f} |",
    f"| Median    | {stats['median']:.6f} |",
    f"| 25th pct  | {stats['p25']:.6f} |",
    f"| 75th pct  | {stats['p75']:.6f} |",
    f"| 95th pct  | {stats['p95']:.6f} |",
    "",
    "## Notes",
    "",
    "- Weights are computed offline from the Stage 3b checkpoint using pixel-wise",
    "  error rate (hardness) and minority class pixel fraction (richness).",
    "- Parameters: β=0.5, γ=1.0, α_mix=0.5, clip [5th, 95th], ε=1e-6.",
    "- Mean ≈ 1.0 by construction (weights are normalised then mixed with uniform).",
    "- Replicated tiles inherit the base-tile weight.",
]

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
print(f"Weights loaded:  {WEIGHTS_TSV}")
print(f"N tiles:         {stats['n']}")
print(f"Min / Max:       {stats['min']:.4f} / {stats['max']:.4f}")
print(f"Mean ± Std:      {stats['mean']:.4f} ± {stats['std']:.4f}")
print(f"Median:          {stats['median']:.4f}")
print(f"25th / 75th pct: {stats['p25']:.4f} / {stats['p75']:.4f}")
print(f"95th pct:        {stats['p95']:.4f}")
print(f"\nFigure saved:    {OUT_PDF}")
print(f"Summary saved:   {OUT_MD}")
