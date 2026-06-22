"""Arm A3 — A1 + Settlement copy-paste (Levers 2+1). Primary L1. SEED=42 triage.
Run: SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_armA3.py --force"""
from _arm_common import build_arm
globals().update(build_arm("stage3_armA3", copy_paste=True))
