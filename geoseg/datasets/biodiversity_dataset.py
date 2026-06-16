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
# Geometry-aware augmentation (mask-aligned)
# -----------------------------------------------------------------------------

def train_aug_random(img: Image.Image, mask: Image.Image):
    """Stage 1/2/3 default: RandomScale + SmartCropV1 -> 512x512."""
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
    ):
        self.data_root = data_root
        self.transform = transform
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        self.img_dir = osp.join(data_root, "images")
        self.mask_dir = osp.join(data_root, "masks")

        self.img_ids = self._get_img_ids()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img = _read_tif_as_rgb_uint8(osp.join(self.img_dir, img_id + self.img_suffix))
        mask = Image.open(osp.join(self.mask_dir, img_id + self.mask_suffix)).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

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
    def __init__(self, data_root, transform=None):
        super().__init__(data_root=data_root, transform=(transform or train_aug_random))


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
