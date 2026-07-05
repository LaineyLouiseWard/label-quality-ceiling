# geoseg/datasets/biodiversity_dataset.py
from __future__ import annotations

"""
Biodiversity TIFF segmentation dataset (6-class).

- Background = class 0 (REAL class, but ignored by loss/metrics via ignore_index=0 in training configs)
- No void class in masks
- SmartCropV1 may use 255 internally for padding (we always map 255 -> 0 before returning)

Stage usage:
- Stage 1/2/3 (hierarchical experiment): train_aug_random
"""

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import albumentations as albu

# geometry-aware transforms (mask-aligned)
from geoseg.datasets.transform import Compose, RandomScale, SmartCropV1, SmartCropV2


# -----------------------------------------------------------------------------
# Constants / taxonomy
# -----------------------------------------------------------------------------

ORIGIN_IMG_SIZE: Tuple[int, int] = (512, 512)

# Canonical student taxonomy — single source of truth in geoseg/taxonomy.py.
from geoseg.taxonomy import STUDENT_CLASSES, STUDENT_PALETTE

CLASSES = STUDENT_CLASSES
PALETTE = STUDENT_PALETTE


# -----------------------------------------------------------------------------
# TIFF image reader (shared everywhere)
# -----------------------------------------------------------------------------

try:
    import rasterio  # type: ignore
except Exception:
    rasterio = None


def _normalize_percentile(img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        band = img[:, :, c].astype(np.float32)
        valid = band[(band != 0) & ~np.isnan(band)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 > p2:
                band = np.clip(band, p2, p98)
                band = (band - p2) / (p98 - p2)
        out[:, :, c] = band
    return out


def _read_tif_as_rgb_uint8(path: str) -> Image.Image:
    if rasterio is not None:
        with rasterio.open(path) as src:
            data = src.read()  # (C,H,W)
            data = np.transpose(data, (1, 2, 0))
            data = np.where(np.isnan(data), 0, data)
            data = _normalize_percentile(data)
            data = (data * 255).clip(0, 255).astype(np.uint8)
            if data.shape[2] >= 3:
                data = data[:, :, :3]
            else:
                data = np.repeat(data, 3, axis=2)
            return Image.fromarray(data)

    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# -----------------------------------------------------------------------------
# Albumentations (pixel-level)
# -----------------------------------------------------------------------------

def _train_tf():
    return albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomBrightnessContrast(0.25, 0.25, p=0.25),
            albu.Normalize(),
        ]
    )


def _val_tf():
    return albu.Compose([albu.Normalize()])


# -----------------------------------------------------------------------------
# Lever 1 — Settlement copy-paste augmentation (behind a flag; default OFF)
# -----------------------------------------------------------------------------
# Hard-mask composite (MSAug Eqs 14-16, Gong 2024; Ghiasi 2021): one binary mask drives
# everything, the IMAGE is copied and the LABEL is set HARD to Settlement (never blended /
# never interpolated). Settlement is pasted only onto Background (dst==0). Photometric jitter
# is applied LATER, image-only and globally, by _train_tf() on the composited tile.
SETTLEMENT_INDEX = 4
assert STUDENT_CLASSES[SETTLEMENT_INDEX] == "Settlement", (
    f"copy-paste donor class id {SETTLEMENT_INDEX} is not Settlement: "
    f"{STUDENT_CLASSES[SETTLEMENT_INDEX]!r}"
)

_COPYPASTE_CFG: dict | None = None
_COPYPASTE_DONORS: dict[str, list[str]] = {}  # donor_root -> list of Settlement-bearing img_ids
_COPYPASTE_DONOR_W: dict = {}  # cache: (donor_root, quality_tsv, temp) -> (ids, prob)


def configure_settlement_copypaste(
    enabled: bool,
    donor_root: str = "data/biodiversity_split/train",
    prob: float = 0.5,
    n_donors: int = 1,
    flip: bool = True,
    target_class: int = SETTLEMENT_INDEX,
    targeted: bool = False,
    quality_tsv: str = "artifacts/donor_quality_settlement.tsv",
    quality_temp: float = 1.0,
    paste_onto=(0,),
):
    """Enable/disable Settlement copy-paste in train_aug_random (Lever 1). Call from an arm config.

    targeted=True selects donors weighted by Stage-2b Settlement CONFIDENCE (MSAug-style: paste clean,
    well-segmented instances, down-weight noisy/ambiguous Settlement), read from quality_tsv (built by
    scripts/data_prep/build_donor_quality.py). targeted=False = uniform-random donor (the baseline).
    quality_temp sharpens the confidence weighting (prob ∝ confidence**temp).
    """
    global _COPYPASTE_CFG
    if not enabled:
        _COPYPASTE_CFG = None
        return
    _COPYPASTE_CFG = dict(
        donor_root=donor_root, prob=float(prob), n_donors=int(n_donors),
        flip=bool(flip), target_class=int(target_class),
        targeted=bool(targeted), quality_tsv=str(quality_tsv), quality_temp=float(quality_temp),
        paste_onto=tuple(int(c) for c in paste_onto),
    )


def _donor_ids(donor_root: str, target_class: int) -> list[str]:
    """Lazily index donor tiles that contain the target class (scan mask PNGs once)."""
    if donor_root in _COPYPASTE_DONORS:
        return _COPYPASTE_DONORS[donor_root]
    mask_dir = osp.join(donor_root, "masks")
    ids = []
    for f in sorted(os.listdir(mask_dir)):
        if not f.endswith(".png"):
            continue
        m = np.array(Image.open(osp.join(mask_dir, f)).convert("L"))
        if (m == target_class).any():
            ids.append(osp.splitext(f)[0])
    _COPYPASTE_DONORS[donor_root] = ids
    return ids


def _donor_pool(cfg):
    """Return (ids, prob). prob is None for uniform-random; a confidence-weighted distribution
    (over the Settlement-bearing donors that have a quality score) when targeted."""
    base_ids = _donor_ids(cfg["donor_root"], cfg["target_class"])
    if not cfg.get("targeted"):
        return base_ids, None
    key = (cfg["donor_root"], cfg["quality_tsv"], cfg["quality_temp"])
    if key in _COPYPASTE_DONOR_W:
        return _COPYPASTE_DONOR_W[key]
    qpath = cfg["quality_tsv"]
    if not osp.isabs(qpath):
        # resolve relative to the repo root (the dir containing artifacts/)
        here = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
        qpath = osp.join(here, cfg["quality_tsv"])
    conf = {}
    with open(qpath) as f:
        header = f.readline()  # img_id\tconfidence\trecall\tarea
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            conf[parts[0]] = float(parts[1])
    base_set = set(base_ids)
    ids = [i for i in conf if i in base_set]
    if not ids:  # fallback to uniform if the quality file doesn't align
        _COPYPASTE_DONOR_W[key] = (base_ids, None)
        return base_ids, None
    w = np.array([conf[i] for i in ids], dtype=np.float64) ** cfg["quality_temp"]
    p = w / w.sum()
    _COPYPASTE_DONOR_W[key] = (ids, p)
    return ids, p


def _compose_settlement(dst_img, dst_mask, src_img, src_mask, target_class=SETTLEMENT_INDEX,
                        paste_onto=(0,)):
    """Pure hard composite. Returns (out_img, out_mask, paste, alpha).

    alpha = donor target blob (src_mask==target_class); paste = alpha & (dst_mask ∈ paste_onto). The
    label is set HARD (out_mask[paste]=target_class) — never blended, never NEAREST-of-a-fractional.

    paste_onto = destination class ids the Settlement blob may overwrite:
      (0,)      ORIGINAL Background-only rule. Label-safe but a near-no-op on this data: 63% of train
                tiles have ZERO Background (median 0%), so the blob∩Background is empty and nothing is
                deposited. NON-standard vs Ghiasi 2021 (random
                placement + occlusion).
      (0,2,3)   TARGETED: Background+Grassland+Cropland = open land where rural Settlement plausibly
                sits. Deposits on ~every tile, lands at the Grassland boundary where Settlement is
                actually confused (§15.1), and never overwrites Forest/Settlement/Semi-natural.
    """
    assert dst_img.shape[:2] == dst_mask.shape, (dst_img.shape, dst_mask.shape)
    assert src_img.shape[:2] == src_mask.shape, (src_img.shape, src_mask.shape)
    assert dst_img.shape == src_img.shape, (dst_img.shape, src_img.shape)
    alpha = src_mask == target_class
    paste = alpha & np.isin(dst_mask, np.asarray(paste_onto))
    out_img = dst_img.copy()
    out_mask = dst_mask.copy()
    out_img[paste] = src_img[paste]          # boolean mask keeps the channel axis -> pixel-locked
    out_mask[paste] = target_class           # HARD label
    return out_img, out_mask, paste, alpha


def _load_donor(donor_root: str, img_id: str, size_hw: tuple[int, int], flip: bool):
    """Load a donor tile and resize it to the destination size (img BICUBIC, mask NEAREST)."""
    h, w = size_hw
    img = _read_tif_as_rgb_uint8(osp.join(donor_root, "images", img_id + ".tif")).resize(
        (w, h), Image.BICUBIC
    )
    mask = Image.open(osp.join(donor_root, "masks", img_id + ".png")).convert("L").resize(
        (w, h), Image.NEAREST
    )
    src_img = np.array(img)
    src_mask = np.array(mask)
    if flip:
        if np.random.rand() < 0.5:  # horizontal — exact (no interpolation) for img AND mask
            src_img = src_img[:, ::-1].copy()
            src_mask = src_mask[:, ::-1].copy()
        if np.random.rand() < 0.5:  # vertical
            src_img = src_img[::-1, :].copy()
            src_mask = src_mask[::-1, :].copy()
    return src_img, src_mask


def _apply_settlement_copypaste(img_np, mask_np):
    """Composite 1..n_donors Settlement blobs onto Background of (img_np, mask_np). Flag-gated.
    Donor is uniform-random (baseline) or confidence-weighted (targeted=True)."""
    cfg = _COPYPASTE_CFG
    if cfg is None or np.random.rand() >= cfg["prob"]:
        return img_np, mask_np
    ids, p = _donor_pool(cfg)
    if not ids:
        return img_np, mask_np
    out_img, out_mask = img_np, mask_np
    for _ in range(cfg["n_donors"]):
        j = np.random.randint(len(ids)) if p is None else int(np.random.choice(len(ids), p=p))
        src_img, src_mask = _load_donor(cfg["donor_root"], ids[j], out_mask.shape, cfg["flip"])
        out_img, out_mask, _, _ = _compose_settlement(
            out_img, out_mask, src_img, src_mask, cfg["target_class"], cfg.get("paste_onto", (0,))
        )
    return out_img, out_mask


# -----------------------------------------------------------------------------
# Geometry-aware augmentation (mask-aligned)
# -----------------------------------------------------------------------------

def train_aug_random(img: Image.Image, mask: Image.Image):
    """Stage 1/2/3 default: RandomScale + SmartCropV1 -> 512x512.

    If Settlement copy-paste is configured (Lever 1), the hard composite is applied AFTER the
    crop + 255->0 safety and BEFORE _train_tf() — so the global, image-only photometric jitter in
    _train_tf() lands on the whole composited tile and the label stays hard.
    """
    crop_aug = Compose(
        [
            RandomScale([0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=0, nopad=False),
        ]
    )

    img, mask = crop_aug(img, mask)
    img_np, mask_np = np.array(img), np.array(mask)

    # Safety: remove SmartCrop padding label if present
    mask_np[mask_np == 255] = 0

    if _COPYPASTE_CFG is not None:
        img_np, mask_np = _apply_settlement_copypaste(img_np, mask_np)

    out = _train_tf()(image=img_np, mask=mask_np)
    return out["image"], out["mask"]


def val_aug(img: Image.Image, mask: Image.Image):
    img = img.resize((512, 512), Image.BICUBIC)
    mask = mask.resize((512, 512), Image.NEAREST)
    img_np, mask_np = np.array(img), np.array(mask)

    # safety
    mask_np[mask_np == 255] = 0

    out = _val_tf()(image=img_np, mask=mask_np)
    return out["image"], out["mask"]

def train_aug_minority(img: Image.Image, mask: Image.Image):
    """
    Minority-aware crop policy:
    - biases crops to include rare classes Settlement(4) and Seminatural(5)
    - still uses same pixel-level aug + normalize as train_aug_random
    """
    crop_aug = Compose(
        [
            RandomScale([0.75, 1.0, 1.25, 1.5], mode="value"),
            SmartCropV2(
                crop_size=512,
                num_classes=6,
                class_interest=[4, 5],
                class_ratio=[0.01, 0.01],
                max_ratio=0.75,
                ignore_index=0,
                nopad=False,
            ),
        ]
    )

    img, mask = crop_aug(img, mask)
    img_np, mask_np = np.array(img), np.array(mask)

    # Safety: remove SmartCrop padding label if present
    mask_np[mask_np == 255] = 0

    out = _train_tf()(image=img_np, mask=mask_np)
    return out["image"], out["mask"]

# -----------------------------------------------------------------------------
# Dataset base
# -----------------------------------------------------------------------------

class _BiodiversitySegDataset(Dataset):
    """
    Expects:
      data_root/
        images/*.tif
        masks/*.png
    """

    def __init__(
        self,
        data_root: str,
        transform=None,
        img_suffix: str = ".tif",
        mask_suffix: str = ".png",
        mosaic_ratio: float = 0.0,
    ):
        self.data_root = data_root
        self.transform = transform
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        # Mosaic augmentation: fraction of training items composed as a 2x2 splice
        # of 4 tiles BEFORE the normal transform. 0.0 = off (default; val/test stay off).
        self.mosaic_ratio = float(mosaic_ratio)

        self.img_dir = osp.join(data_root, "images")
        self.mask_dir = osp.join(data_root, "masks")

        self.img_ids = self._get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_pil(self, idx: int):
        """Read one tile as (RGB PIL uint8, L PIL mask), sizes matched."""
        img_id = self.img_ids[idx]
        img = _read_tif_as_rgb_uint8(osp.join(self.img_dir, img_id + self.img_suffix))
        mask = Image.open(osp.join(self.mask_dir, img_id + self.mask_suffix)).convert("L")
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)
        return img, mask

    def _make_mosaic(self, idx: int):
        """2x2 mosaic of 4 tiles (this idx + 3 random) onto a fixed canvas.

        Robust by construction: each source is resized to the full canvas (image
        bilinear, mask NEAREST) so any quadrant crop is valid regardless of the
        source tile's native size; masks never interpolate, so output classes stay
        a subset of the inputs' {0..5}. Uses np.random (seeded per-worker by
        pl.seed_everything(workers=True)) for reproducibility.
        """
        H, W = ORIGIN_IMG_SIZE  # (512, 512)
        idxs = [idx] + [int(np.random.randint(0, len(self.img_ids))) for _ in range(3)]
        # splice centre kept away from edges so every quadrant is >= H/4 x W/4
        cx = int(np.random.randint(W // 4, W - W // 4 + 1))
        cy = int(np.random.randint(H // 4, H - H // 4 + 1))
        quad = [(cy, cx), (cy, W - cx), (H - cy, cx), (H - cy, W - cx)]  # (h, w) per pane
        pos = [(0, 0), (0, cx), (cy, 0), (cy, cx)]                       # top-left (row, col)
        img_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        mask_canvas = np.zeros((H, W), dtype=np.uint8)
        for k in range(4):
            ip, mp = self._load_pil(idxs[k])
            ia = np.asarray(ip.resize((W, H), Image.BILINEAR), dtype=np.uint8)
            ma = np.asarray(mp.resize((W, H), Image.NEAREST), dtype=np.uint8)
            qh, qw = quad[k]
            y0 = int(np.random.randint(0, H - qh + 1))
            x0 = int(np.random.randint(0, W - qw + 1))
            r, c = pos[k]
            img_canvas[r:r + qh, c:c + qw] = ia[y0:y0 + qh, x0:x0 + qw]
            mask_canvas[r:r + qh, c:c + qw] = ma[y0:y0 + qh, x0:x0 + qw]
        return Image.fromarray(img_canvas), Image.fromarray(mask_canvas, mode="L")

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        use_mosaic = (
            self.transform is not None
            and self.mosaic_ratio > 0.0
            and np.random.random() < self.mosaic_ratio
        )
        if use_mosaic:
            img, mask = self._make_mosaic(idx)
        else:
            img, mask = self._load_pil(idx)

        if self.transform:
            img_np, mask_np = self.transform(img, mask)
        else:
            img_np, mask_np = np.array(img), np.array(mask)

        # Enforce label invariant (prevents CUDA asserts)
        if np.any((mask_np < 0) | (mask_np > 5)):
            bad = np.unique(mask_np[(mask_np < 0) | (mask_np > 5)]).tolist()
            raise ValueError(f"Invalid mask labels {bad} for img_id={img_id}")

        img_t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
        return {"img": img_t, "gt_semantic_seg": mask_t, "img_id": img_id}

    def _get_img_ids(self) -> List[str]:
        imgs = sorted(f for f in os.listdir(self.img_dir) if f.endswith(self.img_suffix))
        masks = set(f for f in os.listdir(self.mask_dir) if f.endswith(self.mask_suffix))
        return [
            osp.splitext(f)[0]
            for f in imgs
            if f.replace(self.img_suffix, self.mask_suffix) in masks
        ]


# -----------------------------------------------------------------------------
# Public datasets
# -----------------------------------------------------------------------------

class BiodiversityTrainDataset(_BiodiversitySegDataset):
    def __init__(self, data_root, transform=None, mosaic_ratio: float = 0.0):
        super().__init__(
            data_root=data_root,
            transform=(transform or train_aug_random),
            mosaic_ratio=mosaic_ratio,
        )


class BiodiversityValDataset(_BiodiversitySegDataset):
    def __init__(self, data_root="data/biodiversity_split/val", transform=None, **kw):
        super().__init__(data_root, transform=(transform or val_aug), **kw)


class BiodiversityTestDataset(Dataset):
    def __init__(self, data_root="data/biodiversity_split/test"):
        self.img_dir = osp.join(data_root, "images")
        self.img_ids = sorted(osp.splitext(f)[0] for f in os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = _read_tif_as_rgb_uint8(osp.join(self.img_dir, img_id + ".tif"))
        img_np = _val_tf()(image=np.array(img))["image"]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return {"img": img_t, "img_id": img_id}


class BiodiversityTestWithMasksDataset(_BiodiversitySegDataset):
    """Test split WITH masks, for evaluation only."""
    def __init__(self, data_root="data/biodiversity_split/test", transform=None, **kw):
        super().__init__(data_root=data_root, transform=(transform or val_aug), **kw)
