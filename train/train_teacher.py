"""
Train a teacher segmentation model on OpenEarthMap (OEM).

This script performs standard supervised training and saves
Lightning checkpoints (.ckpt). Export to a plain .pth state_dict
is handled separately by export_teacher_checkpoint.py.
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from geoseg.utils.cfg import py2cfg
from geoseg.utils.metric import Evaluator


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config_path", type=Path, required=True)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing checkpoint without prompting.")
    return p.parse_args()


# ---------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------
class TeacherTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = config.net
        self.loss = config.loss
        self.classes = config.classes

        ignore_index = getattr(config, "ignore_index", None)
        self.train_evaluator = Evaluator(len(self.classes), ignore_index=ignore_index)
        self.val_evaluator = Evaluator(len(self.classes), ignore_index=ignore_index)

        self._train_cache = []
        self._val_cache = []

    def forward(self, x):
        return self.net(x)

    # ---------------- TRAIN ----------------
    def on_train_epoch_start(self):
        self.train_evaluator.reset()
        self._train_cache.clear()

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        logits = self(img)
        loss = self.loss(logits, mask)
        pred = torch.argmax(logits, dim=1)

        self._train_cache.append(
            {
                "pred": pred.detach().cpu().numpy(),
                "target": mask.detach().cpu().numpy(),
            }
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        for o in self._train_cache:
            self.train_evaluator.add_batch(o["target"], o["pred"])

        iou = self.train_evaluator.Intersection_over_Union()
        f1 = self.train_evaluator.F1()
        oa = self.train_evaluator.OA()

        self.log("train_mIoU", np.nanmean(iou), prog_bar=True)
        self.log("train_F1", np.nanmean(f1), prog_bar=True)
        self.log("train_OA", oa, prog_bar=True)

        print("\ntrain:", {"mIoU": np.nanmean(iou), "F1": np.nanmean(f1), "OA": oa})

    # ---------------- VAL ----------------
    def on_validation_epoch_start(self):
        self.val_evaluator.reset()
        self._val_cache.clear()

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        logits = self(img)
        loss = self.loss(logits, mask)
        pred = torch.argmax(logits, dim=1)

        self._val_cache.append(
            {
                "pred": pred.detach().cpu().numpy(),
                "target": mask.detach().cpu().numpy(),
            }
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        for o in self._val_cache:
            self.val_evaluator.add_batch(o["target"], o["pred"])

        iou = self.val_evaluator.Intersection_over_Union()
        f1 = self.val_evaluator.F1()
        oa = self.val_evaluator.OA()

        self.log("val_mIoU", np.nanmean(iou), prog_bar=True)
        self.log("val_F1", np.nanmean(f1), prog_bar=True)
        self.log("val_OA", oa, prog_bar=True)

        print("\nval:", {"mIoU": np.nanmean(iou), "F1": np.nanmean(f1), "OA": oa})

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = get_args()
    config = py2cfg(args.config_path)

    seed_everything(42)
    pl.seed_everything(42, workers=True)

    g = torch.Generator().manual_seed(42)
    if hasattr(config, "train_loader") and config.train_loader is not None:
        config.train_loader.worker_init_fn = seed_worker
        config.train_loader.generator = g
    if hasattr(config, "val_loader") and config.val_loader is not None:
        config.val_loader.worker_init_fn = seed_worker
        config.val_loader.generator = g

    model = TeacherTrain(config)

    os.makedirs(config.weights_path, exist_ok=True)

    ckpt_out = Path(config.weights_path) / f"{config.weights_name}.ckpt"
    if ckpt_out.exists() and not args.force:
        raise FileExistsError(
            f"Checkpoint already exists: {ckpt_out}\n"
            "Pass --force to overwrite."
        )
    if ckpt_out.exists() and args.force:
        ckpt_out.unlink()
        logging.info(f"Removed existing checkpoint: {ckpt_out}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=config.weights_name,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=config.save_last,
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_cb],
        logger=[
            CSVLogger("lightning_logs", name=config.log_name),
            TensorBoardLogger("lightning_logs", name=config.log_name),
        ],
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
    )

    _accum = trainer.accumulate_grad_batches
    _phys = config.train_batch_size
    print(f"[teacher] physical_batch={_phys}  accumulate_grad_batches={_accum}  effective_batch={_phys * _accum}")

    trainer.fit(model, config.train_loader, config.val_loader)


if __name__ == "__main__":
    main()
