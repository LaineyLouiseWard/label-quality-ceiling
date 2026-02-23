#!/usr/bin/env python3
"""
scripts/figures/build_all_figures.py

Single entry point to reproduce all manuscript figures.

Usage:
  python scripts/figures/build_all_figures.py [--device cuda|cpu] [--skip <figN> ...]
                                               [--include-supplementary]

Figures that require model checkpoints (04, 07, 10) will fail loudly if
checkpoints are missing; all others depend only on data or saved artifacts.
Figure 03 has no script (manually produced vector diagram).

Notebooks (Figure02, Figure06) are executed via jupyter nbconvert.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent.parent


def run_py(script: Path, extra_args: list[str], device: str) -> bool:
    if not script.exists():
        print(f"\nERROR: script not found: {script}")
        return False
    cmd = [sys.executable, str(script), f"--device={device}", *extra_args]
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def run_py_no_device(script: Path) -> bool:
    """For scripts that do not accept a --device argument."""
    if not script.exists():
        print(f"\nERROR: script not found: {script}")
        return False
    cmd = [sys.executable, str(script)]
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def run_nb(notebook: Path) -> bool:
    if not notebook.exists():
        print(f"\nERROR: notebook not found: {notebook}")
        return False
    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(notebook),
    ]
    print(f"\n{'='*60}")
    print(f"Running notebook: {notebook.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build all manuscript figures.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Device for scripts that run model inference (default: cuda).")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Figure numbers to skip, e.g. --skip 04 07 10")
    ap.add_argument("--include-supplementary", action="store_true",
                    help="Also run supplementary figure scripts (FigureXX.py).")
    args = ap.parse_args()

    skip = set(args.skip)

    # Canonical manuscript figures (ordered)
    # Each entry: (label, runner_callable)
    figures: list[tuple[str, callable]] = [
        ("01", lambda: run_py(SCRIPTS_DIR / "Figure01.py", [], args.device)),
        ("02", lambda: run_nb(SCRIPTS_DIR / "Figure02.ipynb")),
        # 03: no script — manually produced vector diagram
        ("04", lambda: run_py(SCRIPTS_DIR / "Figure04.py", [], args.device)),
        ("05", lambda: run_py(SCRIPTS_DIR / "Figure05.py", [], args.device)),
        ("06", lambda: run_nb(SCRIPTS_DIR / "Figure06.ipynb")),
        ("07", lambda: run_py(SCRIPTS_DIR / "Figure07.py", [], args.device)),
        ("08", lambda: run_py(SCRIPTS_DIR / "Figure08.py", [], args.device)),
        ("09", lambda: run_py(SCRIPTS_DIR / "Figure09.py", [], args.device)),
        ("10", lambda: run_py(SCRIPTS_DIR / "Figure10.py", [], args.device)),
    ]

    # Supplementary figures (no model inference; run only with --include-supplementary)
    supplementary: list[tuple[str, callable]] = [
        ("XX (weight dist.)", lambda: run_py_no_device(SCRIPTS_DIR / "FigureXX.py")),
    ]

    results: dict[str, bool | str] = {}

    for fig_num, runner in figures:
        if fig_num in skip:
            results[fig_num] = "skipped"
            print(f"\nSkipping Figure {fig_num} (--skip).")
            continue
        results[fig_num] = runner()

    if args.include_supplementary:
        for label, runner in supplementary:
            results[label] = runner()

    print(f"\n{'='*60}")
    print("Build summary")
    print(f"{'='*60}")
    for fig_num, status in results.items():
        if status == "skipped":
            marker = "SKIP"
        elif status is True:
            marker = "OK  "
        else:
            marker = "FAIL"
        print(f"  Figure {fig_num}: {marker}")

    print(f"  Figure 03: no script (manually produced)")

    failed = [n for n, s in results.items() if s is False]
    if failed:
        print(f"\n{len(failed)} figure(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll requested figures built successfully.")


if __name__ == "__main__":
    main()
