"""Self/ensemble-distillation loss (RETIRED — self-distillation DROPPED; kept for reference, not in the current factorial pipeline). In-domain ensemble teacher.

Pairs with `geoseg.models.ensemble_teacher.EnsembleTeacher`. Unlike the cross-taxonomy
OEM-KD path (`kd_utils.KDHelper`, which remaps 9->6 via a mapping matrix), here the teacher
is in-domain (6-class), so the third loss argument is already the ensemble TARGET PROBABILITY
distribution at temperature T — no remap.

Form (textbook Hinton): loss = (1 - alpha) * hard + alpha * T^2 * KL(teacher || student),
with the KL computed per pixel (summed over classes, mean over batch*pixels). The T^2 factor
keeps the soft-target gradient magnitude comparable across temperatures, so `alpha` means the
same thing as you sweep T in {1, 2, 5, 10} (protocol §11.3).

Crucially `alpha` can be HIGH here (default 0.5, protocol §11.3) because the ensemble teacher is
*stronger* than the student (+1.19 pp) — the opposite of the taxonomy-blind OEM teacher that
forced alpha=0.10.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfDistillLoss(nn.Module):
    """(1-alpha)*hard + alpha*T^2*KL(teacher_probs || student) for an in-domain ensemble teacher.

    Args:
        hard_loss: the supervised loss on ground truth, called as hard_loss(student_logits,
            targets) — e.g. JointLoss(SoftCrossEntropyLoss(ignore_index=0), DiceLoss(...)).
        alpha: weight on the distillation term in [0, 1] (default 0.5).
        temperature: T; MUST match the EnsembleTeacher's temperature (the teacher probs are
            already softmaxed at T; the student is tempered here).
        ignore_index: if set, KL is masked out where target == ignore_index (default None =
            distil over all pixels, matching the existing KDHelper convention; background
            dark-knowledge is harmless). Pass 0 to restrict KD to foreground pixels.
    """

    def __init__(self, hard_loss: nn.Module, alpha: float = 0.5, temperature: float = 2.0,
                 ignore_index=None):
        super().__init__()
        self.hard_loss = hard_loss
        self.alpha = float(alpha)
        self.T = float(temperature)
        self.ignore_index = ignore_index

    def forward(self, student_logits, targets, teacher_probs):
        hard = self.hard_loss(student_logits, targets)

        log_student = F.log_softmax(student_logits / self.T, dim=1)
        # KL(teacher || student) per element; teacher_probs is the target distribution.
        kl = F.kl_div(log_student, teacher_probs, reduction="none")  # (N, C, H, W)
        kl = kl.sum(dim=1)                                           # per-pixel KL -> (N, H, W)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            denom = valid.sum().clamp(min=1)
            kd = (kl * valid).sum() / denom
        else:
            kd = kl.mean()

        kd = kd * (self.T ** 2)
        return (1.0 - self.alpha) * hard + self.alpha * kd
