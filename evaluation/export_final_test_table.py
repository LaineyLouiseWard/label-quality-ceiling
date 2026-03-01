#!/usr/bin/env python3
"""
Export a LaTeX tabular snippet summarising test-set performance of the final model.

Reads:
  evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/metrics.json

Writes:
  evaluation/evaluation_results/final_test_table.tex

The output is a bare tabularx environment (no \\begin{table} float) so it can be
\\input{} inside any table/figure environment in the manuscript.

Usage (from repo root):
  python evaluation/export_final_test_table.py
"""

from __future__ import annotations

import json
from pathlib import Path


# ── Paths (relative to repo root) ──────────────────────────────────────────

METRICS_PATH = Path("evaluation/evaluation_results/test/stage5_final_kd_ftunetformer/metrics.json")
OUTPUT_PATH = Path("evaluation/evaluation_results/final_test_table.tex")

# Class names matching compute_metrics.py (foreground only)
CLASS_NAMES_5 = ["Forest land", "Grassland", "Cropland", "Settlement", "Seminatural Grassland"]

# Display names matching Table 2 in the manuscript
DISPLAY_NAMES = ["Forest", "Grassland", "Cropland", "Settlement", "Semi-natural"]


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_PATH}\n"
            "Run evaluation/compute_metrics.py --split test first."
        )

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Extract values
    oa = m["OA"] * 100
    miou = m["mIoU_excluding_bg"] * 100
    mf1 = m["mF1_excluding_bg"] * 100
    per_class_iou = {c: m["per_class_iou"][c] * 100 for c in CLASS_NAMES_5}

    # Build LaTeX tabularx (bare — no \begin{table} wrapper)
    lines = [
        r"\begin{tabularx}{\textwidth}{",
        r"    >{\centering\arraybackslash}p{1.6cm}",
        r"    >{\centering\arraybackslash}p{1.6cm}",
        r"    >{\centering\arraybackslash}p{1.6cm}",
        r"    >{\centering\arraybackslash}p{1.8cm}",
        r"    >{\centering\arraybackslash}p{2.0cm}",
        r"    >{\centering\arraybackslash}p{1.2cm}",
        r"    >{\centering\arraybackslash}p{1.2cm}",
        r"    >{\centering\arraybackslash}X",
        r"}",
        r"\toprule",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Forest} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Grassland} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Cropland} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Settlement} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{Semi-natural} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{mIoU} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{mF1} \\ (\%) \end{tabular} &",
        r"\begin{tabular}[c]{@{}c@{}} \textbf{OA} \\ (\%) \end{tabular}",
        r"\\",
        r"\midrule",
    ]

    # Data row — 1 decimal place to match Tables 1–2
    iou_vals = [per_class_iou[c] for c in CLASS_NAMES_5]
    data_cells = " & ".join(f"{v:.1f}" for v in iou_vals)
    data_cells += f" & {miou:.1f} & {mf1:.1f} & {oa:.1f}"
    lines.append(data_cells + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")

    # Write
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Summary to stdout
    print(f"Source: {METRICS_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Test-set metrics (Stage 5 KD, final model):")
    for name, key in zip(DISPLAY_NAMES, CLASS_NAMES_5):
        print(f"  {name:14s} IoU = {per_class_iou[key]:.1f}%")
    print(f"  {'mIoU':14s}     = {miou:.1f}%")
    print(f"  {'mF1':14s}     = {mf1:.1f}%")
    print(f"  {'OA':14s}     = {oa:.1f}%")


if __name__ == "__main__":
    main()
