#!/usr/bin/env python3
"""
scripts/figures/build_all_figures.py

Single entry point to reproduce all manuscript figures.

Usage:
  python scripts/figures/build_all_figures.py [--device cuda|cpu] [--skip <figN> ...]

Figures use stable DESCRIPTIVE names (decoupled from printed numbers; see CLAUDE.md).

Figure types:
  - factorial_design, mitigation_axes, oem_mapping are TikZ (.tex), compiled with
    pdflatex and copied to figures/.
  - rgb_tiles, oem_harmonisation, class_distributions, sampler_tiles,
    ablation_qualitative, confusion_matrices, factorial_effects are Python scripts
    that write directly to figures/.
  - The paper's last three figures (boundary_distance, uncertainty_overlay,
    class_pair_boundary) are produced by scripts/analysis/* and copied in, not here.

ablation_qualitative takes a --device argument (model inference); the rest are data-
or artifact-only. Pass --skip <name> to skip by the descriptive key.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent
FIGURES_DIR = REPO_ROOT / "figures"


def run_py(script: Path, extra_args: list[str], device: str) -> bool:
    if not script.exists():
        print(f"\nERROR: script not found: {script}")
        return False
    cmd = [sys.executable, str(script), f"--device={device}", *extra_args]
    print(f"\n{'='*60}\nRunning: {script.name}\n{'='*60}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode == 0


def run_py_no_device(script: Path) -> bool:
    """For scripts that do not accept a --device argument."""
    if not script.exists():
        print(f"\nERROR: script not found: {script}")
        return False
    cmd = [sys.executable, str(script)]
    print(f"\n{'='*60}\nRunning: {script.name}\n{'='*60}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode == 0


def run_tex(tex: Path) -> bool:
    """Compile a standalone TikZ figure and copy the PDF into figures/."""
    if not tex.exists():
        print(f"\nERROR: source not found: {tex}")
        return False
    print(f"\n{'='*60}\nCompiling: {tex.name}\n{'='*60}")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name]
    ok = subprocess.run(cmd, cwd=str(SCRIPTS_DIR)).returncode == 0
    produced = SCRIPTS_DIR / (tex.stem + ".pdf")
    if not ok or not produced.exists():
        return False
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, FIGURES_DIR / produced.name)
    # tidy LaTeX side products
    for ext in (".aux", ".log"):
        (SCRIPTS_DIR / (tex.stem + ext)).unlink(missing_ok=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build all manuscript figures.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Device for scripts that run model inference (default: cuda).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Figure numbers to skip, e.g. --skip 03 08 11")
    args = ap.parse_args()

    skip = set(args.skip)

    # Canonical manuscript figures (ordered)
    # "mapping" = the grounded OEM<->Bio mapping schematic (oem_mapping.tex -> final Fig 5).
    # Its numeric content is generated reproducibly by _gen_mapping_values.py from the frozen
    # teacher confusion; no campaign dependency, so it builds with the rest of the TikZ set.
    # NOTE: the graphical abstract (graphical_abstract_tikz.tex) is intentionally NOT wired here
    # yet — it needs the median-seed prediction PNGs (biodiversity_1310_stage{1,3,4}*.png) which
    # do not exist until the seed campaign finishes; wiring it now would make this build fail.
    figures: list[tuple[str, callable]] = [
        # 2026-06-25 figure audit: figures now use stable DESCRIPTIVE names (decoupled from
        # printed numbers, which LaTeX assigns) so reordering during revision needs no renames.
        # The cut figures (clsbal weight histogram, recovered-vs-lost transitions, the three
        # supplementary uncertainty figures) were removed from the final figure set.
        # The paper's last three figures (boundary_distance, uncertainty_overlay,
        # class_pair_boundary) are produced by scripts/analysis/* and copied in, not built here.
        ("factorial_design",    lambda: run_tex(SCRIPTS_DIR / "factorial_design.tex")),
        ("mitigation_axes",     lambda: run_tex(SCRIPTS_DIR / "mitigation_axes.tex")),
        ("oem_mapping",         lambda: run_tex(SCRIPTS_DIR / "oem_mapping.tex")),
        ("rgb_tiles",           lambda: run_py_no_device(SCRIPTS_DIR / "rgb_tiles.py")),
        ("oem_harmonisation",   lambda: run_py_no_device(SCRIPTS_DIR / "oem_harmonisation.py")),
        ("class_distributions", lambda: run_py_no_device(SCRIPTS_DIR / "class_distributions.py")),
        ("sampler_tiles",       lambda: run_py_no_device(SCRIPTS_DIR / "sampler_tiles.py")),
        ("ablation_qualitative",lambda: run_py(SCRIPTS_DIR / "ablation_qualitative.py", [], args.device)),
        ("confusion_matrices",  lambda: run_py_no_device(SCRIPTS_DIR / "confusion_matrices.py")),
        ("factorial_effects",   lambda: run_py_no_device(SCRIPTS_DIR / "factorial_effects.py")),
    ]

    results: dict[str, bool | str] = {}
    for fig_num, runner in figures:
        if fig_num in skip:
            results[fig_num] = "skipped"
            print(f"\nSkipping Figure {fig_num} (--skip).")
            continue
        results[fig_num] = runner()

    print(f"\n{'='*60}\nBuild summary\n{'='*60}")
    for fig_num, status in results.items():
        marker = "SKIP" if status == "skipped" else ("OK  " if status is True else "FAIL")
        print(f"  Figure {fig_num}: {marker}")

    failed = [n for n, s in results.items() if s is False]
    if failed:
        print(f"\n{len(failed)} figure(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll requested figures built successfully.")


if __name__ == "__main__":
    main()
