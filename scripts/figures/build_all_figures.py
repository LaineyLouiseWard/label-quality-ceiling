#!/usr/bin/env python3
"""
scripts/figures/build_all_figures.py

Single entry point to reproduce all manuscript figures.

Usage:
  python scripts/figures/build_all_figures.py [--device cuda|cpu] [--skip <figN> ...]

Figures that require model checkpoints (04, 07, 09, 10) will fail if checkpoints
are missing; all others (01, 02, 05, 06, 08) depend only on data/artifacts.
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
    cmd = [sys.executable, str(script), f"--device={device}", *extra_args]
    print(f"\n{'='*60}")
    print(f"Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def run_nb(notebook: Path) -> bool:
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
    args = ap.parse_args()

    skip = set(args.skip)

    # Ordered list: (figure_number_str, callable)
    figures: list[tuple[str, callable]] = [
        ("01", lambda: run_py(SCRIPTS_DIR / "Figure01.py", [], args.device)),
        ("02", lambda: run_nb(SCRIPTS_DIR / "Figure02.ipynb")),
        # 03: no script — manually produced
        ("04", lambda: run_py(SCRIPTS_DIR / "Figure04.py", [], args.device)),
        ("05", lambda: run_py(SCRIPTS_DIR / "Figure05.py", [], args.device)),
        ("06", lambda: run_nb(SCRIPTS_DIR / "Figure06.ipynb")),
        ("07", lambda: run_py(SCRIPTS_DIR / "Figure07.py", [], args.device)),
        ("08", lambda: run_py(SCRIPTS_DIR / "Figure08.py", [], args.device)),
        ("09", lambda: run_py(SCRIPTS_DIR / "Figure09.py", [], args.device)),
        ("10", lambda: run_py(SCRIPTS_DIR / "Figure10.py", [], args.device)),
    ]

    results: dict[str, bool | str] = {}
    for fig_num, runner in figures:
        if fig_num in skip:
            results[fig_num] = "skipped"
            print(f"\nSkipping Figure {fig_num} (--skip).")
            continue
        ok = runner()
        results[fig_num] = ok

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

    # Also note Fig 03 (no script)
    print(f"  Figure 03: no script (manually produced)")

    failed = [n for n, s in results.items() if s is False]
    if failed:
        print(f"\n{len(failed)} figure(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll figures built successfully.")


if __name__ == "__main__":
    main()
