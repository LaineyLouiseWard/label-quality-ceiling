#!/usr/bin/env python3
"""
evaluation/tile_number_metrics.py

Counts dataset tile totals for the *current ClassImbalance data tree*.

Raw OpenEarthMap layout (your download):
data/openearthmap_raw/OpenEarthMap/OpenEarthMap_wo_xBD/<region>/{images,labels}/*.tif
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set


REP_SUFFIX_RE = re.compile(r"_rep\d+$", re.IGNORECASE)


# -------------------------
# helpers
# -------------------------
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data").exists() and (p / "evaluation").exists():
            return p
    return start


def list_files(dir_path: Path, exts: Sequence[str], recursive: bool = False) -> List[Path]:
    if not dir_path.exists():
        return []
    files: List[Path] = []
    if recursive:
        for ext in exts:
            files.extend(dir_path.rglob(f"*{ext}"))
    else:
        for ext in exts:
            files.extend(dir_path.glob(f"*{ext}"))
    return sorted(files)


def stems(paths: Iterable[Path]) -> Set[str]:
    return {p.stem for p in paths}


def stem_no_rep(stem: str) -> str:
    return REP_SUFFIX_RE.sub("", stem)


def fmt(n: int) -> str:
    return f"{n:,}"


@dataclass
class DirCounts:
    name: str
    root: str
    images: int
    masks: int
    paired: int
    unique_ids: int
    replicas: int


def count_paired_dir(
    name: str,
    root: Path,
    img_dir: str = "images",
    mask_dir: str = "masks",
    img_exts: Sequence[str] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    mask_exts: Sequence[str] = (".png",),
) -> DirCounts:
    img_p = root / img_dir
    msk_p = root / mask_dir

    imgs = list_files(img_p, img_exts, recursive=False)
    msks = list_files(msk_p, mask_exts, recursive=False)

    img_ids = stems(imgs)
    msk_ids = stems(msks)
    paired_ids = img_ids & msk_ids

    uniq = {stem_no_rep(s) for s in paired_ids}
    replicas = len(paired_ids) - len(uniq)

    return DirCounts(
        name=name,
        root=str(root),
        images=len(imgs),
        masks=len(msks),
        paired=len(paired_ids),
        unique_ids=len(uniq),
        replicas=replicas,
    )


def merge_counts(name: str, root: Path, counts: List[DirCounts]) -> DirCounts:
    images = sum(c.images for c in counts)
    masks = sum(c.masks for c in counts)
    paired = sum(c.paired for c in counts)
    # unique_ids/replicas are not meaningful across heterogeneous region stems (they’re already unique)
    # but we can compute them as: unique_ids = paired - replicas, and replicas=0
    return DirCounts(
        name=name,
        root=str(root),
        images=images,
        masks=masks,
        paired=paired,
        unique_ids=paired,
        replicas=0,
    )


def count_oem_raw_by_regions(oem_root: Path) -> DirCounts:
    """
    Counts raw OEM where regions are immediate subfolders:
      oem_root/<region>/{images,labels}/*.tif
    """
    region_counts: List[DirCounts] = []
    if not oem_root.exists():
        return DirCounts("OpenEarthMap raw", str(oem_root), 0, 0, 0, 0, 0)

    for region in sorted([p for p in oem_root.iterdir() if p.is_dir()]):
        # only treat as region if it has images/ and labels/
        if not (region / "images").is_dir() or not (region / "labels").is_dir():
            continue
        c = count_paired_dir(
            name=f"raw:{region.name}",
            root=region,
            img_dir="images",
            mask_dir="labels",
            img_exts=(".tif", ".tiff"),
            mask_exts=(".tif", ".tiff"),
        )
        region_counts.append(c)

    return merge_counts("OpenEarthMap raw", oem_root, region_counts)


def print_block(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def print_counts_row(c: DirCounts) -> None:
    print(
        f"{c.name:<36} paired={fmt(c.paired):>6}  "
        f"(images={fmt(c.images):>6}, masks={fmt(c.masks):>6}, "
        f"unique={fmt(c.unique_ids):>6}, reps={fmt(c.replicas):>5})"
    )


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="Relative to repo root")
    ap.add_argument("--out-json", default="", help="Optional JSON output path, e.g. artifacts/tile_number_metrics.json")
    args = ap.parse_args()

    repo_root = find_repo_root(Path.cwd())
    data_root = (repo_root / args.data_root).resolve()

    # ---- A) Raw dataset sizes (provenance) ----
    biodiv_raw = count_paired_dir(
        name="Biodiversity raw",
        root=data_root / "biodiversity_raw",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        mask_exts=(".png",),
    )

    oem_raw_root = data_root / "openearthmap_raw" / "OpenEarthMap" / "OpenEarthMap_wo_xBD"
    oem_raw = count_oem_raw_by_regions(oem_raw_root)

    # ---- B) OEM processing (student-side) ----
    oem_filtered = count_paired_dir(
        name="OEM rural-filtered",
        root=data_root / "openearthmap_filtered",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff"),
        mask_exts=(".tif", ".tiff"),
    )

    oem_relabelled = count_paired_dir(
        name="OEM relabelled (→Biodiv taxonomy)",
        root=data_root / "openearthmap_relabelled",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff"),
        mask_exts=(".png",),
    )

    oem_relabelled_filtered = count_paired_dir(
        name="OEM relabelled + settlement-filtered",
        root=data_root / "openearthmap_relabelled_filtered",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff"),
        mask_exts=(".png",),
    )

    # ---- C) OEM teacher dataset ----
    oem_teacher_train = count_paired_dir(
        name="OEM teacher train",
        root=data_root / "openearthmap_teacher" / "train",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff"),
        mask_exts=(".tif", ".tiff"),
    )
    oem_teacher_val = count_paired_dir(
        name="OEM teacher val",
        root=data_root / "openearthmap_teacher" / "val",
        img_dir="images",
        mask_dir="masks",
        img_exts=(".tif", ".tiff"),
        mask_exts=(".tif", ".tiff"),
    )

    # ---- D) Biodiversity split + replication ----
    biodiv_split_root = data_root / "biodiversity_split"
    biodiv_train = count_paired_dir(name="Biodiversity train", root=biodiv_split_root / "train")
    biodiv_train_rep = count_paired_dir(name="Biodiversity train_rep", root=biodiv_split_root / "train_rep")
    biodiv_val = count_paired_dir(name="Biodiversity val", root=biodiv_split_root / "val")
    biodiv_test = count_paired_dir(name="Biodiversity test", root=biodiv_split_root / "test")

    # ---- E) Combined pretraining pool (Biodiversity + OEM) ----
    combined_root = data_root / "biodiversity_oem_combined"
    combined_train = count_paired_dir(name="Combined pretrain train", root=combined_root / "train")
    combined_val = count_paired_dir(name="Combined pretrain val", root=combined_root / "val")
    combined_test = count_paired_dir(name="Combined pretrain test", root=combined_root / "test")

    # ---- Print report ----
    print(f"Repo root: {repo_root}")
    print(f"Data root: {data_root}")

    print_block("A) Raw dataset sizes (provenance)")
    print_counts_row(biodiv_raw)
    print_counts_row(oem_raw)

    print_block("B) OpenEarthMap processing (student-side)")
    print_counts_row(oem_filtered)
    print_counts_row(oem_relabelled)
    print_counts_row(oem_relabelled_filtered)

    print_block("C) OpenEarthMap teacher dataset")
    print_counts_row(oem_teacher_train)
    print_counts_row(oem_teacher_val)

    print_block("D) Biodiversity splits (training / eval)")
    print_counts_row(biodiv_train)
    print_counts_row(biodiv_train_rep)
    print_counts_row(biodiv_val)
    print_counts_row(biodiv_test)

    print_block("E) Combined pretraining pool (Biodiversity + OEM)")
    print_counts_row(combined_train)
    print_counts_row(combined_val)
    print_counts_row(combined_test)

    print_block("F) Stage-wise pools (paper-facing)")
    print(f"{'Stage 1–2':<12} Biodiversity train/train_rep pool: {fmt(biodiv_train.paired)} / {fmt(biodiv_train_rep.paired)} paired tiles")
    print(f"{'Stage 3–4':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles (same pool; different sampling/cropping)")
    print(f"{'Stage 5 pre':<12} Combined pretrain train pool:     {fmt(combined_train.paired)} paired tiles")
    print(f"{'Stage 5 ft':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles")
    print(f"{'Stage 5':<12} Biodiversity train_rep pool:       {fmt(biodiv_train_rep.paired)} paired tiles (teacher supervision)")
    print(f"{'Eval':<12} Biodiversity val/test pools:       {fmt(biodiv_val.paired)} / {fmt(biodiv_test.paired)} paired tiles")

    if args.out_json:
        out_path = (repo_root / args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "counts": {
                "biodiversity_raw": asdict(biodiv_raw),
                "openearthmap_raw": asdict(oem_raw),
                "openearthmap_filtered": asdict(oem_filtered),
                "openearthmap_relabelled": asdict(oem_relabelled),
                "openearthmap_relabelled_filtered": asdict(oem_relabelled_filtered),
                "openearthmap_teacher_train": asdict(oem_teacher_train),
                "openearthmap_teacher_val": asdict(oem_teacher_val),
                "biodiversity_train": asdict(biodiv_train),
                "biodiversity_train_rep": asdict(biodiv_train_rep),
                "biodiversity_val": asdict(biodiv_val),
                "biodiversity_test": asdict(biodiv_test),
                "combined_train": asdict(combined_train),
                "combined_val": asdict(combined_val),
                "combined_test": asdict(combined_test),
            },
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {out_path}")


if __name__ == "__main__":
    main()
