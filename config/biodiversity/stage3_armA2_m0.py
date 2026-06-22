"""Arm A2 (momentum=0, Tian-exact control) — A1 + instantaneous recall-weighted SoftCE (Levers 2+3).
Isolates the EMA's effect from the method. SEED=42 triage.
Run: SEED=42 PYTHONPATH=. python -m train.train_supervision -c config/biodiversity/stage3_armA2_m0.py --force"""
from _arm_common import build_arm
globals().update(build_arm("stage3_armA2_m0", recall=True, recall_momentum=0.0))
