from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    - Optional per-class weight vector applied per-pixel by ground-truth class (used by the
      recall-weighted subclass, RecallCrossEntropyLoss). ``weight=None`` reproduces the
      original unweighted behaviour bit-for-bit.
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1,
                 weight: Optional[Tensor] = None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        # Stored as a buffer so it follows the module to the correct device. None => unweighted.
        self.register_buffer("weight", weight if weight is None else weight.float())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
            weight=self.weight,
        )
