"""
Teacher training: EfficientNet-B4 U-Net on OpenEarthMap (native taxonomy 0..8).

- Uses your openearthmap_teacher split created by prepare_oem_teacher_data.py
- Uses SAME TIFF->RGB conversion as Biodiversity (_read_tif_as_rgb_uint8)
- Output checkpoint: pretrain_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth
"""

from torch.utils.data import DataLoader
import torch

from geoseg.losses import *
from geoseg.models.unet import TeacherUNet
from geoseg.utils.optim import Lookahead

from geoseg.datasets.openearthmap_dataset import (
    OEM_CLASSES_9,
    OpenEarthMapTeacherTrainDataset,
    OpenEarthMapTeacherValDataset,
    oem_train_aug,
    oem_val_aug,
)


# -------------------------
# training hparams
# -------------------------
max_epoch = 45
ignore_index = 255

train_batch_size = 4
val_batch_size = 4

lr = 3e-4 #changed from 6e-4 to match. 
weight_decay = 2.5e-4

num_classes = 9
classes = OEM_CLASSES_9

# -------------------------
# logging / ckpt
# -------------------------
# --- checkpoints ---
weights_path = "model_weights/teacher"
weights_name = "teacher"
test_weights_name = weights_name
log_name = "teacher_oem/u-efficientnet-b4"

monitor = "val_mIoU"
monitor_mode = "max"
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1

pretrained_ckpt_path = None
resume_ckpt_path = None
gpus = "auto"

# -------------------------
# network / loss
# -------------------------
net = TeacherUNet(num_classes=num_classes, pretrained=True)

# Teacher loss: smooth_factor=0.0 and smooth=0.0 are deliberate.
# Student stages use eps=0.05 (paper Section 2.7); teacher training does not.
loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.0, ignore_index=ignore_index),
    DiceLoss(smooth=0.0, ignore_index=ignore_index),
    1.0,
    1.0,
)
use_aux_loss = False

# -------------------------
# datasets
# -------------------------
train_dataset = OpenEarthMapTeacherTrainDataset(
    data_root="data/openearthmap_teacher/train",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".tif",
    mask_suffix=".png",
)

val_dataset = OpenEarthMapTeacherValDataset(
    data_root="data/openearthmap_teacher/val",
    img_dir="images",
    mask_dir="masks",
    img_suffix=".tif",
    mask_suffix=".png",
)

# Sanity check: fail fast if teacher data is missing
print(f"[unet_oem] train: {train_dataset.data_root} ({len(train_dataset)} samples)")
print(f"[unet_oem] val:   {val_dataset.data_root} ({len(val_dataset)} samples)")
if len(train_dataset) == 0:
    raise RuntimeError(
        f"Teacher train dataset is empty (data_root={train_dataset.data_root}).\n"
        "  Run A8 (prepare_oem_teacher_data.py) first, or check the path."
    )



train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

# -------------------------
# optimizer
# -------------------------
base_optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
