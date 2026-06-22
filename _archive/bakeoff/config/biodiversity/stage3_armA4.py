"""Arm A4 — A1 + recall-weighted SoftCE (momentum=0.9) + Settlement copy-paste (Levers 1+2+3),
all in ONE fine-tune from Stage 2b (NOT A3-plus-loss). Expected winner. SEED=42 triage.
Run: SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_armA4.py --force"""
from _arm_common import build_arm
globals().update(build_arm("stage3_armA4", recall=True, recall_momentum=0.9, copy_paste=True))
