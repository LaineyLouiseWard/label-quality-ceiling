"""Arm A2 (momentum=0.9, EMA default) — A1 + recall-weighted SoftCE (Levers 2+3). Primary L3.
Run: SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_armA2_m09.py --force"""
from _arm_common import build_arm
globals().update(build_arm("stage3_armA2_m09", recall=True, recall_momentum=0.9))
