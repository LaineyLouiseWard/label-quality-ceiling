#!/usr/bin/env python3
"""
Aggregate per-seed evaluation metrics across the multi-seed campaign and estimate the
2x2 factorial ablation effects.

The final campaign is a 2x2 factorial over {OEM transfer} x {clsbal sampler}, run for
10 seeds (42..51). The four cells are:
  baseline       stage1_baseline       (-transfer, -sampler)
  transfer-only  stage2b_oem_finetune  (+transfer, -sampler)
  sampler-only   stage_sampler_only    (-transfer, +sampler)
  full           stage3_clsbal         (+transfer, +sampler)  -- the final shipped model

Each seed is an independent draw of the student pipeline (init + sampler/aug RNG) run
through all four cells. This script reads every seed's ``metrics.json`` and reports:

  * Per-cell mean +/- *sample* std (ddof=1) per class -- the reproducibility statistic.
  * The three factorial effects as **within-seed paired contrasts across the seeds**, each
    with a 95 % paired-t confidence interval and a paired significance test (this is the
    statistic that quantifies "the increment exceeds training-stochasticity noise"; a sample
    std band does NOT):
      OEM transfer (main)   = mean(+transfer) - mean(-transfer) , paired per seed
                            = 0.5*[(full + transfer-only) - (sampler-only + baseline)]
      clsbal sampler (main) = mean(+sampler) - mean(-sampler)   , paired per seed
                            = 0.5*[(full + sampler-only) - (transfer-only + baseline)]
      transfer x sampler    = (full - transfer-only) - (sampler-only - baseline), per seed
    plus the cumulative full - baseline as a secondary "total pipeline gain".

Outputs (all under analysis/seed_aggregate/, gitignored -- proprietary-data-derived):
  summary.json              machine-readable mean/std/n + factorial effects (mean, CI, p)
  per_seed_metrics.csv      tidy provenance: every seed x split x cell x metric value
  report.md                 human-readable cell tables + factorial-effect attribution
  ablation_<split>.tex      manuscript-ready bare tabularx (mean +/- std, 1 dp)
  figure10_iou_<split>.csv  per-class IoU mean/std/CI per cell (Figure 10 input)

Run from anywhere (the repo is located via find_repo_root):
  PYTHONPATH=. python scripts/analysis/aggregate_seeds.py                 # local worktree layout
  PYTHONPATH=. python scripts/analysis/aggregate_seeds.py \\
      --results-dir /path/to/final_results                                # flat Sonic drop (val)
  ... --seeds 42 43 44 45 46 47 48 49 50 51 --root-seed 42 [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── repo root ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.utils import find_repo_root  # noqa: E402

REPO = find_repo_root()

# ── the 2x2 factorial cells ───────────────────────────────────────────────────
# Defined LOCALLY on purpose: do NOT fold these into utils.STAGES, which the a1..a6
# cumulative-ablation scripts iterate as the 3-stage chain (baseline -> transfer -> full).
# Adding the 4th (sampler-only) cell there would inject a spurious row into those tables.
#            label            folder                  transfer  sampler
CELLS = [
    ("baseline",      "stage1_baseline",       False, False),
    ("transfer-only", "stage2b_oem_finetune",  True,  False),
    ("sampler-only",  "stage_sampler_only",    False, True),
    ("full",          "stage3_clsbal",         True,  True),
]
CELL_FOLDERS = [c[1] for c in CELLS]
STAGE_VIEW = [(label, folder) for label, folder, *_ in CELLS]  # for the per-cell tables

F_BASELINE = "stage1_baseline"
F_TRANSFER = "stage2b_oem_finetune"
F_SAMPLER = "stage_sampler_only"
F_FULL = "stage3_clsbal"

# Factorial effects as per-seed linear contrasts over the four cells (folder-keyed).
EFFECT_ORDER = ["transfer", "sampler", "interaction", "total"]
EFFECT_LABELS = {
    "transfer": "OEM transfer (main effect)",
    "sampler": "clsbal sampler (main effect)",
    "interaction": "transfer x sampler (interaction)",
    "total": "full - baseline (cumulative)",
}


def effect_contrasts(cv: dict[str, float]) -> dict[str, float]:
    """Per-seed factorial contrasts from one seed's four cell values (folder -> value)."""
    b, t, s, f = cv[F_BASELINE], cv[F_TRANSFER], cv[F_SAMPLER], cv[F_FULL]
    return {
        "transfer": 0.5 * (t + f) - 0.5 * (b + s),     # main effect of OEM transfer
        "sampler": 0.5 * (s + f) - 0.5 * (b + t),      # main effect of clsbal sampler
        # Standard 2^2 factorial (Montgomery): interaction = 1/2[(ab - b) - (a - (1))],
        # on the SAME scale as the main effects so transfer + sampler == full - baseline.
        "interaction": 0.5 * ((f - s) - (t - b)),      # transfer x sampler interaction
        "total": f - b,                                # cumulative gain (secondary)
    }


SPLITS = ("val", "test")

# Foreground classes, keyed exactly as compute_metrics.py writes them.
FG_CLASSES = ["Forest land", "Grassland", "Cropland", "Settlement", "Seminatural Grassland"]
DISPLAY = {
    "Forest land": "Forest",
    "Grassland": "Grassland",
    "Cropland": "Cropland",
    "Settlement": "Settlement",
    "Seminatural Grassland": "Semi-natural",
}
SCALAR_METRICS = ["mIoU_excluding_bg", "mF1_excluding_bg", "OA"]
SCALAR_LABELS = {"mIoU_excluding_bg": "mIoU", "mF1_excluding_bg": "mF1", "OA": "OA"}

# Metrics the factorial effects are reported for: headline mIoU + the two minority classes.
EFFECT_METRICS = [
    ("mIoU", "mIoU_excluding_bg", None),
    ("Settlement", None, "Settlement"),
    ("Semi-natural", None, "Seminatural Grassland"),
]


# ── seed → repo path ─────────────────────────────────────────────────────────

def seed_repo(seed: int, root_seed: int) -> Path:
    """Root seed = this repo; the others = sibling worktree ``<repo>_seed<N>``."""
    if seed == root_seed:
        return REPO
    return REPO.parent / f"{REPO.name}_seed{seed}"


def metrics_path(seed: int, root_seed: int, split: str, folder: str) -> Path:
    return seed_repo(seed, root_seed) / "evaluation" / "evaluation_results" / split / folder / "metrics.json"


def flat_path(results_dir: Path, seed: int, folder: str) -> Path:
    """Flat Sonic drop written by the campaign slurm: ``seed<N>_<cell>.json`` (val cells)."""
    return results_dir / f"seed{seed}_{folder}.json"


# ── stats helpers ────────────────────────────────────────────────────────────

def mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean and *sample* std (ddof=1); std is NaN for n<2."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    return mean, std, n


def paired_stats(diffs: list[float], alpha: float = 0.05) -> dict:
    """Paired across-seed inference on a per-seed contrast.

    Returns mean, sample std, 95% paired-t CI (mean +/- t_{n-1} * se), the two-sided
    one-sample t-test p-value (H0: effect = 0), and the Wilcoxon signed-rank p-value as a
    distribution-free check. Falls back to a normal approximation if scipy is unavailable.
    """
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    if n < 2:
        return {"mean": mean, "lo": float("nan"), "hi": float("nan"), "sd": float("nan"),
                "se": float("nan"), "p_t": float("nan"), "p_wilcoxon": float("nan"),
                "n": n, "method": "n<2"}
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    p_t = float("nan")
    p_w = float("nan")
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        p_t = float(stats.ttest_1samp(arr, 0.0).pvalue)
        if np.any(arr != 0):
            p_w = float(stats.wilcoxon(arr).pvalue)
        method = f"paired-t (t_{{{n - 1}}})"
    except Exception:
        tcrit = 1.959963984540054  # z_{0.975} normal approximation
        method = "normal-approx (scipy unavailable)"
    return {"mean": mean, "lo": mean - tcrit * se, "hi": mean + tcrit * se, "sd": sd,
            "se": se, "p_t": p_t, "p_wilcoxon": p_w, "n": n, "method": method}


def mean_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """95% CI half-width of the across-seed mean (mean +/- t_{n-1}*se). NaN for n<2."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n < 2:
        return float("nan"), float("nan")
    se = float(arr.std(ddof=1)) / np.sqrt(n)
    try:
        from scipy import stats
        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        tcrit = 1.959963984540054
    return float(arr.mean()), tcrit * se


def is_nan(x: float) -> bool:
    return isinstance(x, float) and x != x


def json_safe(obj):
    """Recursively replace NaN floats with None so summary.json is valid JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return None if is_nan(obj) else obj


def fmt_pm(mean: float, std: float, dp: int = 1, pm: str = "±") -> str:
    if is_nan(mean):
        return "--"
    if is_nan(std):
        return f"{mean:.{dp}f}"
    return f"{mean:.{dp}f} {pm} {std:.{dp}f}"


def fmt_ci(d: dict, dp: int = 2) -> str:
    """'mean [lo, hi]' for a paired-effect dict."""
    if is_nan(d["mean"]):
        return "--"
    if is_nan(d["lo"]):
        return f"{d['mean']:.{dp}f}"
    return f"{d['mean']:.{dp}f} [{d['lo']:.{dp}f}, {d['hi']:.{dp}f}]"


def fmt_p(p: float) -> str:
    if is_nan(p):
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


# ── loading ──────────────────────────────────────────────────────────────────

def load_metrics_file(path: Path) -> dict:
    """Read one metrics.json → simplified dict with everything in PERCENT."""
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    iou, f1 = m["per_class_iou"], m["per_class_f1"]
    return {
        "mIoU_excluding_bg": m["mIoU_excluding_bg"] * 100,
        "mF1_excluding_bg": m["mF1_excluding_bg"] * 100,
        "OA": m["OA"] * 100,
        "iou": {c: iou[c] * 100 for c in FG_CLASSES},
        "f1": {c: f1[c] * 100 for c in FG_CLASSES},
        "checkpoint": m.get("checkpoint", ""),
    }


def get_metric(simplified: dict, scalar_key: str | None, class_key: str | None) -> float:
    return simplified[scalar_key] if scalar_key is not None else simplified["iou"][class_key]


# ── aggregation ──────────────────────────────────────────────────────────────

def aggregate_stage(seedmap: dict[int, dict]) -> dict:
    seeds = sorted(seedmap)
    out = {"n_seeds": len(seeds), "seeds": seeds, "scalars": {}, "per_class_iou": {}, "per_class_f1": {}}
    for metric in SCALAR_METRICS:
        vals = {s: seedmap[s][metric] for s in seeds}
        mean, std, n = mean_std(list(vals.values()))
        out["scalars"][metric] = {"mean": mean, "std": std, "n": n, "values": vals}
    for c in FG_CLASSES:
        vi = {s: seedmap[s]["iou"][c] for s in seeds}
        mi, si, ni = mean_std(list(vi.values()))
        _, hw = mean_ci(list(vi.values()))
        out["per_class_iou"][c] = {"mean": mi, "std": si, "ci95_halfwidth": hw, "n": ni, "values": vi}
        vf = {s: seedmap[s]["f1"][c] for s in seeds}
        mf, sf, nf = mean_std(list(vf.values()))
        out["per_class_f1"][c] = {"mean": mf, "std": sf, "n": nf, "values": vf}
    return out


def compute_effects(data_split: dict) -> dict:
    """The three factorial effects + cumulative, paired by seed across all four cells."""
    seedmaps = {fl: data_split.get(fl, {}).get("seedmap", {}) for fl in CELL_FOLDERS}
    if not all(seedmaps[fl] for fl in CELL_FOLDERS):
        return {"seeds": [], "metrics": {}}  # the factorial needs all four cells
    common = sorted(set.intersection(*[set(seedmaps[fl]) for fl in CELL_FOLDERS]))
    metrics = {}
    for name, sk, ck in EFFECT_METRICS:
        per_effect = {e: [] for e in EFFECT_ORDER}
        for s in common:
            cv = {fl: get_metric(seedmaps[fl][s], sk, ck) for fl in CELL_FOLDERS}
            contr = effect_contrasts(cv)
            for e in EFFECT_ORDER:
                per_effect[e].append(contr[e])
        metrics[name] = {e: paired_stats(per_effect[e]) for e in EFFECT_ORDER}
    return {"seeds": common, "metrics": metrics}


# ── output writers ───────────────────────────────────────────────────────────

def write_per_seed_csv(path: Path, data: dict) -> None:
    rows = []
    for split in SPLITS:
        for label, folder in STAGE_VIEW:
            seedmap = data[split].get(folder, {}).get("seedmap", {})
            for seed in sorted(seedmap):
                m = seedmap[seed]
                for metric in SCALAR_METRICS:
                    rows.append([seed, split, label, folder, SCALAR_LABELS[metric], "", f"{m[metric]:.4f}"])
                for c in FG_CLASSES:
                    rows.append([seed, split, label, folder, "IoU", DISPLAY[c], f"{m['iou'][c]:.4f}"])
                    rows.append([seed, split, label, folder, "F1", DISPLAY[c], f"{m['f1'][c]:.4f}"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "split", "cell_label", "cell_folder", "metric", "class", "value_pct"])
        w.writerows(rows)


def write_figure10_csv(path: Path, agg_split: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_label", "cell_folder", "class", "iou_mean_pct", "iou_std_pct",
                    "iou_ci95_halfwidth_pct", "n_seeds"])
        for label, folder in STAGE_VIEW:
            if folder not in agg_split:
                continue
            a = agg_split[folder]
            for c in FG_CLASSES:
                d = a["per_class_iou"][c]
                std = "" if is_nan(d["std"]) else f"{d['std']:.4f}"
                hw = "" if is_nan(d["ci95_halfwidth"]) else f"{d['ci95_halfwidth']:.4f}"
                w.writerow([label, folder, DISPLAY[c], f"{d['mean']:.4f}", std, hw, d["n"]])


def write_ablation_tex(path: Path, agg_split: dict, n_max: int) -> None:
    lines = [
        f"% Per-cell ablation (2x2 factorial), mean $\\pm$ std over up to {n_max} seeds. Values in %.",
        "% Bare tabularx (no \\begin{table} wrapper) — \\input{} inside a table float.",
        r"\begin{tabularx}{\textwidth}{l *{8}{>{\centering\arraybackslash}X}}",
        r"\toprule",
        r"\textbf{Cell} & \textbf{Forest} & \textbf{Grassland} & \textbf{Cropland} & "
        r"\textbf{Settlement} & \textbf{Semi-natural} & \textbf{mIoU} & \textbf{mF1} & \textbf{OA} \\",
        r"\midrule",
    ]
    for label, folder in STAGE_VIEW:
        if folder not in agg_split:
            continue
        a = agg_split[folder]
        cells = [label]
        cells += [fmt_pm(a["per_class_iou"][c]["mean"], a["per_class_iou"][c]["std"], pm=r"$\pm$") for c in FG_CLASSES]
        cells += [fmt_pm(a["scalars"][m]["mean"], a["scalars"][m]["std"], pm=r"$\pm$") for m in SCALAR_METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(agg_split: dict) -> list[str]:
    header = ["Cell", "Forest", "Grassland", "Cropland", "Settlement", "Semi-natural", "mIoU", "mF1", "OA"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for label, folder in STAGE_VIEW:
        if folder not in agg_split:
            continue
        a = agg_split[folder]
        cells = [label]
        cells += [fmt_pm(a["per_class_iou"][c]["mean"], a["per_class_iou"][c]["std"]) for c in FG_CLASSES]
        cells += [fmt_pm(a["scalars"][m]["mean"], a["scalars"][m]["std"]) for m in SCALAR_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def md_effects(effects: dict) -> list[str]:
    """Factorial effects: rows = effect, columns = headline metrics, cell = mean [CI] (p)."""
    if not effects.get("seeds"):
        return ["_Factorial effects unavailable — not all four cells present for the seeds._"]
    metric_names = [m[0] for m in EFFECT_METRICS]
    header = ["Effect (pp)"] + [f"{m} (95% CI; p)" for m in metric_names]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for e in EFFECT_ORDER:
        cells = [EFFECT_LABELS[e]]
        for m in metric_names:
            d = effects["metrics"][m][e]
            cells.append(f"{fmt_ci(d)} (p={fmt_p(d['p_t'])})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"_Paired across {len(effects['seeds'])} seeds {effects['seeds']}; "
                 f"95% paired-t CI; p = two-sided one-sample t (H0: effect=0); "
                 f"Wilcoxon p in summary.json. CI excluding 0 ⇒ effect above seed noise._")
    return lines


def write_report(path: Path, aggregates: dict, all_effects: dict, meta: dict) -> None:
    lines = [
        "# Multi-seed campaign aggregate (2x2 factorial)",
        "",
        f"Seeds requested: {meta['seeds']} (root = {meta['root_seed']}). "
        f"Per-cell tables show mean ± sample std (ddof=1, n ≥ 2). Factorial effects are paired "
        f"within-seed contrasts with 95% paired-t CIs. All values in %.",
        "",
    ]
    for split in SPLITS:
        agg = aggregates[split]
        if not agg:
            lines += [f"## {split} — no metrics found", ""]
            continue
        ns = sorted({s for st in agg.values() for s in st["seeds"]})
        lines += [f"## {split} (seeds present: {ns})", ""]
        lines += md_table(agg)
        lines += ["", "### Factorial effects (paired across seeds)", ""]
        lines += md_effects(all_effects[split])
        lines += [""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── coverage check ───────────────────────────────────────────────────────────

def check_coverage(seeds: list[int], root_seed: int, results_dir: Path | None) -> tuple[list, list]:
    """Return (missing_seed_sources, missing_cells) for the four factorial cells (val)."""
    missing_src = []
    missing_cells = []
    for s in seeds:
        if results_dir is not None:
            present = any(flat_path(results_dir, s, fl).exists() for _l, fl, *_ in CELLS)
            if not present:
                missing_src.append(s)
        elif not seed_repo(s, root_seed).is_dir():
            missing_src.append(s)
    for _label, folder, *_ in CELLS:
        for s in seeds:
            if results_dir is not None:
                if not flat_path(results_dir, s, folder).exists():
                    missing_cells.append(("val", folder, s))
            elif seed_repo(s, root_seed).is_dir() and not metrics_path(s, root_seed, "val", folder).exists():
                missing_cells.append(("val", folder, s))
    return missing_src, missing_cells


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-seed metrics + estimate 2x2 factorial effects.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)),
                    help="Seed set (LOCKED final-campaign default = 42..51).")
    ap.add_argument("--root-seed", type=int, default=42,
                    help="Seed that lives in this repo (the others in <repo>_seed<N> worktrees).")
    ap.add_argument("--results-dir", type=str, default=None,
                    help="Flat campaign drop of seed<N>_<cell>.json (val cells). If set, val is "
                         "read from here instead of the per-seed worktree layout.")
    ap.add_argument("--test-results-dir", type=str, default=None,
                    help="Flat campaign drop of seed<N>_<cell>.json (test cells). If set, test is "
                         "read from here instead of the per-seed worktree layout. Use the CURRENT "
                         "final_results_test drop so no stale per-seed eval can be read.")
    ap.add_argument("--out-dir", type=str, default=str(REPO / "analysis" / "seed_aggregate"))
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any requested seed source or any factorial cell (val) is missing.")
    args = ap.parse_args()

    seeds, root_seed = args.seeds, args.root_seed
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None
    test_results_dir = Path(args.test_results_dir) if args.test_results_dir else None

    missing_src, missing_cells = check_coverage(seeds, root_seed, results_dir)
    if missing_src:
        print(f"[warn] seed sources not found: {missing_src}")
    if missing_cells:
        print(f"[warn] {len(missing_cells)} factorial-cell metrics.json missing "
              f"(e.g. {missing_cells[:3]}{' ...' if len(missing_cells) > 3 else ''})")
    if args.strict and (missing_src or missing_cells):
        raise SystemExit("[strict] incomplete campaign — aborting.")

    # Collect every (split, cell, seed) that exists. Both splits read from the CURRENT flat
    # campaign drops when given (--results-dir for val, --test-results-dir for test); the
    # per-seed worktree layout is only a legacy fallback when no flat drop is supplied.
    flat_for = {"val": results_dir, "test": test_results_dir}
    data = {split: {} for split in SPLITS}
    for split in SPLITS:
        for label, folder in STAGE_VIEW:
            seedmap = {}
            for s in seeds:
                if flat_for[split] is not None:
                    p = flat_path(flat_for[split], s, folder)
                else:
                    p = metrics_path(s, root_seed, split, folder)
                if p.exists():
                    seedmap[s] = load_metrics_file(p)
            if seedmap:
                data[split][folder] = {"label": label, "seedmap": seedmap}

    if not any(data[split] for split in SPLITS):
        raise SystemExit(
            f"No metrics.json found for seeds {seeds}. Run the campaign "
            "(sonic/10_submit_final_campaign.slurm, or RUNBOOK.sh per seed) first, "
            "or point --results-dir at the fetched flat drop."
        )

    aggregates = {split: {f: aggregate_stage(d["seedmap"]) for f, d in data[split].items()} for split in SPLITS}
    all_effects = {split: compute_effects(data[split]) for split in SPLITS}

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_from": {"seeds": seeds, "root_seed": root_seed, "repo": str(REPO),
                            "results_dir": str(results_dir) if results_dir else None},
        "design": "2x2 factorial over {OEM transfer} x {clsbal sampler}",
        "cells": {label: {"folder": folder, "transfer": tr, "sampler": sm}
                  for label, folder, tr, sm in CELLS},
        "note": ("Per-cell values in percent, mean + sample std (ddof=1, n<2 -> null). "
                 "Factorial effects are within-seed paired contrasts with 95% paired-t CI "
                 "(p_t = one-sample t vs 0; p_wilcoxon = signed-rank)."),
        "splits": {
            split: {
                "cells": {
                    folder: {"label": data[split][folder]["label"], **aggregates[split][folder]}
                    for folder in aggregates[split]
                },
                "factorial_effects": all_effects[split],
            }
            for split in SPLITS
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    write_per_seed_csv(out_dir / "per_seed_metrics.csv", data)
    write_report(out_dir / "report.md", aggregates, all_effects, {"seeds": seeds, "root_seed": root_seed})
    for split in SPLITS:
        if aggregates[split]:
            write_ablation_tex(out_dir / f"ablation_{split}.tex", aggregates[split], len(seeds))
            write_figure10_csv(out_dir / f"figure10_iou_{split}.csv", aggregates[split])

    # ── stdout summary ───────────────────────────────────────────────────────
    print(f"\nAggregated seeds {seeds} → {out_dir}")
    for split in SPLITS:
        agg = aggregates[split]
        if not agg:
            continue
        print(f"\n[{split}] per-cell mIoU (mean ± std %):")
        for label, folder in STAGE_VIEW:
            if folder in agg:
                d = agg[folder]["scalars"]["mIoU_excluding_bg"]
                print(f"  {label:14s} {folder:24s} {fmt_pm(d['mean'], d['std'])}  (n={d['n']})")
        eff = all_effects[split]
        if eff.get("seeds"):
            print(f"  factorial effects on mIoU (pp, 95% CI, n={len(eff['seeds'])}):")
            for e in EFFECT_ORDER:
                d = eff["metrics"]["mIoU"][e]
                print(f"    {EFFECT_LABELS[e]:34s} {fmt_ci(d)}  (p={fmt_p(d['p_t'])})")


if __name__ == "__main__":
    main()
