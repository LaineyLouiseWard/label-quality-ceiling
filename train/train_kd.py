"""
Knowledge Distillation (KD) training script (Stage 5).

Trains a student segmentation model using:
- Ground-truth labels
- A frozen teacher model (logits supervision)

Everything (models, loaders, losses, optimizer, scheduler) is defined in the config.

Supports:
- Loading student weights from a Lightning checkpoint (config.pretrained_ckpt_path)
- Optional KD disablement if config.kd_enabled == False or config.teacher is None
  (then config.loss must accept (student_logits, targets))
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
    p.add_argument("-c", "--config_path", type=Path, required=True, help="Path to KD config file")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing checkpoint without prompting.")
    return p.parse_args()


# ---------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------
class KDTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.net = config.net
        # NOTE: torch.compile of the student is applied in main() AFTER the pretrained
        # (Stage 4) weights are loaded. Compiling here prefixes params with "_orig_mod." so
        # the un-prefixed checkpoint would silently fail to load and the student would train
        # from random init. Teacher is never compiled. See main() and docs/TORCH_COMPILE_SPEEDUP_TODO.md.
        self.loss = config.loss
        self.classes = config.classes

        # KD toggles:
        # - if config.kd_enabled is provided, obey it
        # - else KD is enabled iff teacher exists
        self.teacher = getattr(config, "teacher", None)
        self.kd_enabled = bool(getattr(config, "kd_enabled", self.teacher is not None))

        # Freeze teacher if present
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        ignore_index = getattr(config, "ignore_index", None)
        self.train_evaluator = Evaluator(len(self.classes), ignore_index=ignore_index)
        self.val_evaluator = Evaluator(len(self.classes), ignore_index=ignore_index)

    def forward(self, x):
        return self.net(x)

    def on_fit_start(self):
        # Ensure teacher is on same device as student (if used)
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()

    # ---------------- TRAIN ----------------
    def on_train_epoch_start(self):
        self.train_evaluator.reset()

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        student_logits = self(img)

        if self.kd_enabled and self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(img)
            loss = self.loss(student_logits, mask, teacher_logits)
        else:
            # loss must accept (student_logits, targets)
            loss = self.loss(student_logits, mask)

        pred = torch.argmax(student_logits, dim=1)

        self.train_evaluator.add_batch(
            mask.detach().cpu().numpy(),
            pred.detach().cpu().numpy()
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=img.size(0),
        )
        return loss

    def on_train_epoch_end(self):
        iou = self.train_evaluator.Intersection_over_Union()
        f1 = self.train_evaluator.F1()
        oa = self.train_evaluator.OA()

        self.log("train_mIoU", np.nanmean(iou), prog_bar=True)
        self.log("train_F1", np.nanmean(f1), prog_bar=True)
        self.log("train_OA", oa, prog_bar=True)

        print("\ntrain:", {"mIoU": np.nanmean(iou), "F1": np.nanmean(f1), "OA": oa})
        print({self.classes[i]: iou[i] for i in range(len(self.classes))})

    # ---------------- VAL ----------------
    def on_validation_epoch_start(self):
        self.val_evaluator.reset()

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        mask = batch["gt_semantic_seg"]

        student_logits = self(img)

        if self.kd_enabled and self.teacher is not None:
            with torch.no_grad():
                teacher_logits = self.teacher(img)
            loss = self.loss(student_logits, mask, teacher_logits)
        else:
            loss = self.loss(student_logits, mask)

        pred = torch.argmax(student_logits, dim=1)

        self.val_evaluator.add_batch(
            mask.detach().cpu().numpy(),
            pred.detach().cpu().numpy()
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=img.size(0))
        return loss

    def on_validation_epoch_end(self):
        iou = self.val_evaluator.Intersection_over_Union()
        f1 = self.val_evaluator.F1()
        oa = self.val_evaluator.OA()

        # Foreground-only (exclude background class 0) — matches reported metric.
        miou_fg = float(np.nanmean(iou[1:]))
        self.log("val_mIoU", miou_fg, prog_bar=True)
        self.log("val_F1", np.nanmean(f1), prog_bar=True)
        self.log("val_OA", oa, prog_bar=True)

        print("\nval:", {"mIoU": miou_fg, "F1": np.nanmean(f1), "OA": oa})
        print({self.classes[i]: iou[i] for i in range(len(self.classes))})

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]


# ---------------------------------------------------------------------
# Helper: load student weights only
# ---------------------------------------------------------------------
def _load_student_weights_from_pl_ckpt(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Checkpoint does not look like a Lightning .ckpt: {ckpt_path}")

    state_dict = ckpt["state_dict"]

    # Prefer "net.*" (LightningModule style), else fall back to direct keys.
    net_sd = {k.replace("net.", "", 1): v for k, v in state_dict.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = state_dict

    # Checkpoints saved from a torch.compile'd model carry an "_orig_mod." prefix on every key.
    # Strip it so they match the (uncompiled) target module — otherwise the load silently fails
    # (strict=False) and the student trains from random init. See compile note in main().
    net_sd = {k.replace("_orig_mod.", "", 1): v for k, v in net_sd.items()}

    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    print("Loaded student weights.")
    if missing:
        print("Missing (non-fatal):", missing[:10])
    if unexpected:
        print("Unexpected (non-fatal):", unexpected[:10])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = get_args()

    # Student lineage seed for the 5-seed campaign (default 42). Set SEED=<n> in the environment
    # to reseed the full student lineage; the teacher is built once and held fixed (not reseeded).
    # Seed BEFORE py2cfg: the model is constructed at config-parse time (e.g.
    # ft_unetformer(pretrained=False)), so seeding afterwards would leave the fresh-init stages'
    # weight initialisation entropy-seeded and not reproducible. (KD inits from a checkpoint, but
    # keep the ordering identical to train_supervision for consistency.)
    seed = int(os.environ.get("SEED", "42"))
    seed_everything(seed)
    pl.seed_everything(seed, workers=True)

    config = py2cfg(args.config_path)

    g = torch.Generator().manual_seed(seed)
    if hasattr(config, "train_loader") and config.train_loader is not None:
        config.train_loader.worker_init_fn = seed_worker
        config.train_loader.generator = g
    if hasattr(config, "val_loader") and config.val_loader is not None:
        config.val_loader.worker_init_fn = seed_worker
        config.val_loader.generator = g

    model = KDTrain(config)

    # --- Resume / fresh-run logic -------------------------------------------------------------
    # An interrupted run leaves a resumable "last.ckpt" (full model + optimizer + scheduler + epoch
    # state). Re-running the SAME command picks up from where it stopped automatically; --force wipes
    # both checkpoints and starts fresh. NOTE: resume MUST use the same TORCH_COMPILE setting as the
    # interrupted run (compile prefixes param names with "_orig_mod.", so the saved keys must match).
    weights_dir = Path(config.weights_path)
    best_ckpt = weights_dir / f"{config.weights_name}.ckpt"
    last_ckpt = weights_dir / "last.ckpt"

    if args.force:
        for p in (best_ckpt, last_ckpt):
            if p.exists():
                p.unlink()
                print(f"[force] Removed existing checkpoint: {p}")

    resume_from = str(last_ckpt) if (last_ckpt.exists() and not args.force) else None
    if resume_from:
        print(f"[resume] Resuming from {last_ckpt} (epoch/optimizer/scheduler state restored).")
    elif best_ckpt.exists() and not args.force:
        raise FileExistsError(
            f"Completed checkpoint exists but no resumable last.ckpt: {best_ckpt}\n"
            "Pass --force to restart from scratch."
        )

    # Load pretrained student weights ONLY for a fresh run. On resume, Lightning restores the full
    # model state from last.ckpt, so loading pretrained over it would corrupt the resume.
    if resume_from is None and getattr(config, "pretrained_ckpt_path", None):
        ckpt_path = Path(config.pretrained_ckpt_path)
        assert ckpt_path.exists(), f"Missing ckpt: {ckpt_path}"
        print(f"Loading student weights from: {ckpt_path}")
        _load_student_weights_from_pl_ckpt(model.net, str(ckpt_path))

    # Opt-in torch.compile of the STUDENT only (set TORCH_COMPILE=1), AFTER the pretrained load
    # so checkpoint keys match (compile prefixes params with "_orig_mod."). Teacher left uncompiled.
    # ~10-20% speedup from epoch 2. If it OOMs/graph-breaks it fails in epoch 1 -> unset the var.
    if os.environ.get("TORCH_COMPILE") == "1":
        model.net = torch.compile(model.net)
        print("[torch.compile] student compiled (post-load).")

    checkpoint_cb = ModelCheckpoint(
        dirpath=config.weights_path,
        filename=config.weights_name,
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=True,   # always write last.ckpt so an interrupted run can resume
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
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
    )

    # Fail fast: checkpoint must monitor the foreground-only mIoU logged above.
    assert config.monitor == "val_mIoU", (
        f"Checkpoint monitor must be 'val_mIoU' (foreground-only mIoU). Got: '{config.monitor}'"
    )

    trainer.fit(model, config.train_loader, config.val_loader, ckpt_path=resume_from)


if __name__ == "__main__":
    main()
