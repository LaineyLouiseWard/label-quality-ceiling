"""Arm A1 — per-class sampler ONLY (Lever 2). Primary L2. SEED=42 triage.
Run: SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_armA1.py --force"""
from _arm_common import build_arm
globals().update(build_arm("stage3_armA1"))
