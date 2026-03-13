# geoseg/datasets/biodiversity_oem_dataset.py
from __future__ import annotations

"""
Combined Biodiversity + OpenEarthMap (OEM) dataset for Stage 3a OEM pretraining.

IMPORTANT:
- This file defines DATASETS ONLY.
- NO sampling logic
- NO weights
- NO KD
- NO stage-specific behaviour

Used ONLY in:
- Stage 3a: OEM pretraining (supervised)

All masks are assumed to be harmonised to the 6-class Biodiversity taxonomy:
  0 Background
  1 Forest
  2 Grassland
  3 Cropland
  4 Settlement
  5 Seminatural
"""

import os
import os.path as osp
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from geoseg.datasets.biodiversity_dataset import (
    _read_tif_as_rgb_uint8,
    train_aug_random,
    val_aug,
)


# -----------------------------------------------------------------------------
# Base dataset
# -----------------------------------------------------------------------------

class _OEMSegDataset(Dataset):
    """
    Expects:
      data_root/
        images/*.tif
        masks/*.png
    """

    def __init__(self, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform

        self.img_dir = osp.join(data_root, "images")
        self.mask_dir = osp.join(data_root, "masks")

        self.img_ids = self._get_img_ids()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = _read_tif_as_rgb_uint8(osp.join(self.img_dir, img_id + ".tif"))
        mask = Image.open(osp.join(self.mask_dir, img_id + ".png")).convert("L")

        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        if self.transform is not None:
            img_np, mask_np = self.transform(img, mask)
        else:
            img_np = np.array(img)
            mask_np = np.array(mask)

        # Safety: enforce valid class range
        if np.any((mask_np < 0) | (mask_np > 5)):
            bad = np.unique(mask_np[(mask_np < 0) | (mask_np > 5)]).tolist()
            raise ValueError(f"Invalid labels {bad} in {img_id}")

        img_t = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()

        return {
            "img": img_t,
            "gt_semantic_seg": mask_t,
            "img_id": img_id,
        }

    def _get_img_ids(self) -> List[str]:
        imgs = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".tif"))
        masks = set(f for f in os.listdir(self.mask_dir) if f.endswith(".png"))

        return [
            osp.splitext(f)[0]
            for f in imgs
            if f.replace(".tif", ".png") in masks
        ]


# -----------------------------------------------------------------------------
# Public datasets
# -----------------------------------------------------------------------------

class BiodiversityOEMTrainDataset(_OEMSegDataset):
    """
    Training split for Stage 3a OEM pretraining.
    Uses standard random crop + flip augmentation.
    """
    def __init__(self, data_root: str, transform=None):
        super().__init__(
            data_root=data_root,
            transform=transform or train_aug_random,
        )


class BiodiversityOEMValDataset(_OEMSegDataset):
    """
    Optional validation split (rarely used).
    """
    def __init__(self, data_root: str, transform=None):
        super().__init__(
            data_root=data_root,
            transform=transform or val_aug,
        )
