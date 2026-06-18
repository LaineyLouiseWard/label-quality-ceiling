#!/usr/bin/env python3
"""
Aggregate per-seed evaluation metrics across the multi-seed campaign.

Each seed is an independent draw of the student pipeline (init + sampler/aug RNG)
run through the four ablation stages. The root repo holds the root seed; the other
seeds live in sibling git worktrees ``<repo>_seed<N>`` (see docs/MULTISEED_PLAN.md).
This script reads every seed's ``metrics.json`` and reports **mean ± std** (sample
std, ddof=1) per stage × class for both the val and test splits — the reproducibility
statistic the campaign exists to produce.

Outputs (all under analysis/seed_aggregate/, gitignored — proprietary-data-derived):
  summary.json              machine-readable mean/std/n (percent), incl. paired deltas
  per_seed_metrics.csv      tidy provenance: every seed × split × stage × metric value
  report.md                 human-readable tables + paired-delta attribution
  ablation_<split>.tex      manuscript-ready bare tabularx (mean ± std, 1 dp)
  figure10_iou_<split>.csv  per-class IoU mean/std per stage (Figure 10 input)

Paired deltas (per-seed differences, then mean ± std over the seeds where BOTH arms
exist) attribute each stage's gain to the right cause:
  Stage 4 − Stage 3       KD increment over the sampler
  Stage 4 − no-KD control isolates KD from "just 45 more epochs"  [mandatory claim]
  Stage 3 − no-sampler    isolates the sampler                    [if that null was run]
  Stage 4 − Stage 1       cumulative ablation effect

Run from anywhere (the repo is located via find_repo_root):
  PYTHONPATH=. python scripts/analysis/aggregate_seeds.py
  ... --seeds 42 43 44 45 46 --root-seed 42 [--strict]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── repo root + shared stage map ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.utils import STAGES as MAIN_STAGES, find_repo_root  # noqa: E402

REPO = find_repo_root()

# Built-in per-seed null controls (RUN_NULL_CONTROLS=1). Optional: only aggregated
# when present, but used for the attribution deltas below.
NULL_STAGES = [
    ("3-null", "stage3null_nosampler"),
    ("4-null", "stage4null_nokd"),
]
ALL_STAGES = MAIN_STAGES + NULL_STAGES

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

# Comparisons reported with paired per-seed deltas. (id, label, minuend, subtrahend)
DELTA_SPECS = [
    ("kd_vs_sampler", "Stage 4 (KD) − Stage 3 (sampler)", "stage4_kd", "stage3_sampler"),
    ("kd_vs_nokd_control", "Stage 4 (KD) − no-KD control", "stage4_kd", "stage4null_nokd"),
    ("sampler_vs_nosampler_null", "Stage 3 (sampler) − no-sampler null", "stage3_sampler", "stage3null_nosampler"),
    ("cumulative_1_to_4", "Stage 4 (KD) − Stage 1 (baseline)", "stage4_kd", "stage1_baseline"),
]
# Metrics the deltas are reported for: headline mIoU + the two minority classes.
DELTA_METRICS = [
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


# ── stats helpers ────────────────────────────────────────────────────────────

def mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean and *sample* std (ddof=1); std is NaN for n<2."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    return mean, std, n


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
        out["per_class_iou"][c] = {"mean": mi, "std": si, "n": ni, "values": vi}
        vf = {s: seedmap[s]["f1"][c] for s in seeds}
        mf, sf, nf = mean_std(list(vf.values()))
        out["per_class_f1"][c] = {"mean": mf, "std": sf, "n": nf, "values": vf}
    return out


def compute_delta(data_split: dict, folder_a: str, folder_b: str) -> dict:
    a = data_split.get(folder_a, {}).get("seedmap", {})
    b = data_split.get(folder_b, {}).get("seedmap", {})
    common = sorted(set(a) & set(b))
    metrics = {}
    for name, sk, ck in DELTA_METRICS:
        diffs = [get_metric(a[s], sk, ck) - get_metric(b[s], sk, ck) for s in common]
        mean, std, n = mean_std(diffs)
        metrics[name] = {"mean": mean, "std": std, "n": n}
    return {"seeds": common, "metrics": metrics}


# ── output writers ───────────────────────────────────────────────────────────

def write_per_seed_csv(path: Path, data: dict, root_seed: int) -> None:
    rows = []
    for split in SPLITS:
        for label, folder in ALL_STAGES:
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
        w.writerow(["seed", "split", "stage_label", "stage_folder", "metric", "class", "value_pct"])
        w.writerows(rows)


def write_figure10_csv(path: Path, agg_split: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage_label", "stage_folder", "class", "iou_mean_pct", "iou_std_pct", "n_seeds"])
        for label, folder in MAIN_STAGES:
            if folder not in agg_split:
                continue
            a = agg_split[folder]
            for c in FG_CLASSES:
                d = a["per_class_iou"][c]
                std = "" if is_nan(d["std"]) else f"{d['std']:.4f}"
                w.writerow([label, folder, DISPLAY[c], f"{d['mean']:.4f}", std, d["n"]])


def write_ablation_tex(path: Path, agg_split: dict, n_max: int) -> None:
    lines = [
        f"% Per-stage ablation, mean $\\pm$ std over up to {n_max} seeds. Values in %.",
        "% Bare tabularx (no \\begin{table} wrapper) — \\input{} inside a table float.",
        r"\begin{tabularx}{\textwidth}{l *{8}{>{\centering\arraybackslash}X}}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Forest} & \textbf{Grassland} & \textbf{Cropland} & "
        r"\textbf{Settlement} & \textbf{Semi-natural} & \textbf{mIoU} & \textbf{mF1} & \textbf{OA} \\",
        r"\midrule",
    ]
    for label, folder in MAIN_STAGES:
        if folder not in agg_split:
            continue
        a = agg_split[folder]
        cells = [f"Stage {label}"]
        cells += [fmt_pm(a["per_class_iou"][c]["mean"], a["per_class_iou"][c]["std"], pm=r"$\pm$") for c in FG_CLASSES]
        cells += [fmt_pm(a["scalars"][m]["mean"], a["scalars"][m]["std"], pm=r"$\pm$") for m in SCALAR_METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md_table(agg_split: dict) -> list[str]:
    header = ["Stage", "Forest", "Grassland", "Cropland", "Settlement", "Semi-natural", "mIoU", "mF1", "OA"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for label, folder in MAIN_STAGES:
        if folder not in agg_split:
            continue
        a = agg_split[folder]
        cells = [f"Stage {label}"]
        cells += [fmt_pm(a["per_class_iou"][c]["mean"], a["per_class_iou"][c]["std"]) for c in FG_CLASSES]
        cells += [fmt_pm(a["scalars"][m]["mean"], a["scalars"][m]["std"]) for m in SCALAR_METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def md_deltas(deltas: dict) -> list[str]:
    lines = ["| Comparison | mIoU (pp) | Settlement IoU (pp) | Semi-nat. IoU (pp) | n |",
             "|---|---|---|---|---|"]
    for _id, label, fa, fb in DELTA_SPECS:
        d = deltas.get(_id)
        if not d or not d["seeds"]:
            lines.append(f"| {label} | -- | -- | -- | 0 |")
            continue
        m = d["metrics"]
        cells = [fmt_pm(m[k]["mean"], m[k]["std"]) for k in ("mIoU", "Settlement", "Semi-natural")]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {len(d['seeds'])} |")
    return lines


def write_report(path: Path, data: dict, aggregates: dict, all_deltas: dict, meta: dict) -> None:
    lines = [
        "# Multi-seed campaign aggregate",
        "",
        f"Seeds requested: {meta['seeds']} (root = {meta['root_seed']}). "
        f"Sample std (ddof=1); std shown only where n ≥ 2. All values in %.",
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
        lines += ["", "### Attribution (paired per-seed deltas)", ""]
        lines += md_deltas(all_deltas[split])
        lines += [""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── coverage check ───────────────────────────────────────────────────────────

def check_coverage(seeds: list[int], root_seed: int) -> tuple[list, list]:
    """Return (missing_seed_dirs, missing_main_cells)."""
    missing_dirs = [s for s in seeds if not seed_repo(s, root_seed).is_dir()]
    missing_cells = []
    for split in SPLITS:
        for _label, folder in MAIN_STAGES:
            for s in seeds:
                if seed_repo(s, root_seed).is_dir() and not metrics_path(s, root_seed, split, folder).exists():
                    missing_cells.append((split, folder, s))
    return missing_dirs, missing_cells


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-seed evaluation metrics (mean ± std).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="Seed set (LOCKED campaign default = 42 43 44 45 46).")
    ap.add_argument("--root-seed", type=int, default=42,
                    help="Seed that lives in this repo (the others in <repo>_seed<N> worktrees).")
    ap.add_argument("--out-dir", type=str, default=str(REPO / "analysis" / "seed_aggregate"))
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any requested seed dir or any main-stage cell is missing.")
    args = ap.parse_args()

    seeds, root_seed = args.seeds, args.root_seed
    out_dir = Path(args.out_dir)

    missing_dirs, missing_cells = check_coverage(seeds, root_seed)
    if missing_dirs:
        print(f"[warn] seed dirs not found: {missing_dirs}")
    if missing_cells:
        print(f"[warn] {len(missing_cells)} main-stage metrics.json missing "
              f"(e.g. {missing_cells[:3]}{' ...' if len(missing_cells) > 3 else ''})")
    if args.strict and (missing_dirs or missing_cells):
        raise SystemExit("[strict] incomplete campaign — aborting.")

    # Collect every (split, stage, seed) that exists.
    data = {split: {} for split in SPLITS}
    for split in SPLITS:
        for label, folder in ALL_STAGES:
            seedmap = {}
            for s in seeds:
                p = metrics_path(s, root_seed, split, folder)
                if p.exists():
                    seedmap[s] = load_metrics_file(p)
            if seedmap:
                data[split][folder] = {"label": label, "seedmap": seedmap}

    if not any(data[split] for split in SPLITS):
        raise SystemExit(
            f"No metrics.json found for seeds {seeds}. Run the campaign "
            "(RUNBOOK.sh --from B1 per seed) and C-stage evaluation first."
        )

    # Mandatory-claim guard: every KD seed should have a no-KD control.
    for split in SPLITS:
        kd = set(data[split].get("stage4_kd", {}).get("seedmap", {}))
        nokd = set(data[split].get("stage4null_nokd", {}).get("seedmap", {}))
        if kd - nokd:
            print(f"[warn] {split}: no-KD control missing for seed(s) {sorted(kd - nokd)} "
                  "→ KD-vs-no-KD claim will use fewer seeds (run with RUN_NULL_CONTROLS=1).")

    aggregates = {split: {f: aggregate_stage(d["seedmap"]) for f, d in data[split].items()} for split in SPLITS}
    all_deltas = {
        split: {_id: compute_delta(data[split], fa, fb) for _id, _label, fa, fb in DELTA_SPECS}
        for split in SPLITS
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_from": {"seeds": seeds, "root_seed": root_seed, "repo": str(REPO)},
        "note": "All values in percent. Sample std (ddof=1); std null when n<2.",
        "splits": {
            split: {
                "stages": {
                    folder: {"label": data[split][folder]["label"], **aggregates[split][folder]}
                    for folder in aggregates[split]
                },
                "deltas": all_deltas[split],
            }
            for split in SPLITS
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    write_per_seed_csv(out_dir / "per_seed_metrics.csv", data, root_seed)
    write_report(out_dir / "report.md", data, aggregates, all_deltas, {"seeds": seeds, "root_seed": root_seed})
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
        print(f"\n[{split}] mIoU (mean ± std %):")
        for label, folder in MAIN_STAGES:
            if folder in agg:
                d = agg[folder]["scalars"]["mIoU_excluding_bg"]
                print(f"  Stage {label} {folder:24s} {fmt_pm(d['mean'], d['std'])}  (n={d['n']})")
        kd = all_deltas[split]["kd_vs_nokd_control"]["metrics"]["mIoU"]
        nseed = len(all_deltas[split]["kd_vs_nokd_control"]["seeds"])
        if nseed:
            print(f"  KD vs no-KD control: ΔmIoU = {fmt_pm(kd['mean'], kd['std'])} pp  (n={nseed})")


if __name__ == "__main__":
    main()
