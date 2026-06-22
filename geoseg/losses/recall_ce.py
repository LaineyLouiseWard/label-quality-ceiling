"""
Recall-weighted soft cross-entropy (Tian et al. 2022, "Striking the Right Balance: Recall Loss
for Semantic Segmentation", ICRA 2022, DOI 10.1109/ICRA46639.2022.9811702, arXiv:2106.14917).

Per-class weight = (1 - recall_c) = FNR_c = FN_c / (FN_c + TP_c), applied per-pixel by the pixel's
GROUND-TRUTH class (Tian Eq. 5: Sum_c Sum_{n: y_n=c} (1 - R_c) (-log p_n)). No sum-to-1
normalisation. This ADDS the only per-class CE weight in the pipeline; it does not replace one
(the JointLoss SoftCE term is unweighted today). The Dice term is left unchanged.

recall_c is estimated from the running stream of training batches:
  * momentum = 0   -> instantaneous per-batch recall (Tian-exact control)
  * momentum = 0.9 -> EMA (default; a DELIBERATE stability deviation from Tian, because Settlement
                      recall is noisy per batch at ~3.3% pixels)
The buffer update is GATED on class-present batches (a class absent from the batch targets would
otherwise read recall=0 -> weight~1, systematically biasing that class's weight high) and only runs
in training mode (validation never perturbs the estimate). weight[ignore_index] is pinned to 0.

Indexing invariant (the off-by-one trap, gate G3): the weight vector is LENGTH-6, indexed by the
canonical class id (Background=0 ... Seminatural=5; geoseg/taxonomy.py), weight[0]=0.
"""

from typing import Optional

import torch
from torch import Tensor

from .soft_ce import SoftCrossEntropyLoss

__all__ = ["RecallCrossEntropyLoss"]


class RecallCrossEntropyLoss(SoftCrossEntropyLoss):
    def __init__(
        self,
        num_classes: int = 6,
        ignore_index: Optional[int] = 0,
        smooth_factor: float = 0.0,
        momentum: float = 0.9,
        reduction: str = "mean",
        dim: int = 1,
        init_recall: float = 1.0,
    ):
        # Start from a length-6 zero weight (recall_ema init = 1.0 -> weight 0 until a class is seen).
        super().__init__(
            reduction=reduction,
            smooth_factor=smooth_factor,
            ignore_index=ignore_index,
            dim=dim,
            weight=torch.zeros(num_classes),
        )
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        # Persisted so a resume restores a consistent estimate.
        self.register_buffer("recall_ema", torch.full((num_classes,), float(init_recall)))
        self.register_buffer("recall_init", torch.zeros(num_classes, dtype=torch.bool))
        self._refresh_weight()

    @torch.no_grad()
    def _refresh_weight(self) -> None:
        w = (1.0 - self.recall_ema).clamp(min=0.0)
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            w[self.ignore_index] = 0.0
        self.weight = w  # updates the registered buffer in place (stays on device)

    @torch.no_grad()
    def update_recall(self, logits: Tensor, target: Tensor) -> None:
        pred = logits.detach().argmax(dim=1)  # (N,H,W)
        tgt = target.detach()
        if tgt.dim() == pred.dim() + 1:  # (N,1,H,W) -> (N,H,W)
            tgt = tgt.squeeze(1)
        for c in range(self.num_classes):
            gt_c = tgt == c
            denom = gt_c.sum()
            if denom.item() == 0:
                continue  # class absent from this batch -> do NOT update (gate on class-present)
            tp = (gt_c & (pred == c)).sum().float()
            rec_c = (tp / denom.float()).clamp(0.0, 1.0)
            if (not bool(self.recall_init[c])) or self.momentum == 0.0:
                self.recall_ema[c] = rec_c
            else:
                m = self.momentum
                self.recall_ema[c] = m * self.recall_ema[c] + (1.0 - m) * rec_c
            self.recall_init[c] = True
        self._refresh_weight()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.training:
            self.update_recall(input, target)
        return super().forward(input, target)
