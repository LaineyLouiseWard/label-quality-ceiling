"""In-domain ENSEMBLE teacher for Stage-4 self/ensemble distillation.

RETIRED: self/ensemble distillation was DROPPED (kept only as the Discussion's negative result),
NOT part of the current 2x2 factorial pipeline.

Implements the teacher defined in the Stage-4 protocol: the teacher is the **uniform softmax
mean of N same-architecture FT-UNetFormer seed students** (in-domain, 6-class — so
NO mapping matrix, unlike the cross-taxonomy OEM teacher in `kd_utils.KDHelper`).

`forward(x)` returns the ensemble TARGET DISTRIBUTION (probabilities, already at
temperature T), shape (N, 6, H, W). It therefore drops straight into the existing
`train/train_kd.py` loop — which calls `teacher_out = self.teacher(img)` under no-grad
and passes `teacher_out` as the third argument of the loss — paired with
`geoseg.losses.selfdistill.SelfDistillLoss` (which consumes probabilities, not logits).

The members are frozen and eval-only; the mean is accumulated member-by-member so peak
memory is ~one member's activations, not N (matters for the 10-seed ensemble on one GPU).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleTeacher(nn.Module):
    """Uniform softmax-mean ensemble over N frozen same-arch members.

    Args:
        members: list of already-constructed, weight-loaded student modules. Each must
            output logits of shape (N, C, H, W) in eval mode (aux tuples are tolerated:
            the first element is taken). Dependency-injected so the class is unit-testable
            without checkpoints — use `from_checkpoints(...)` for the production path.
        temperature: distillation temperature T. The returned probabilities are
            mean_i softmax(member_i(x) / T). MUST equal the T used by SelfDistillLoss on
            the student side, or the KL is malformed.
    """

    def __init__(self, members, temperature: float = 2.0):
        super().__init__()
        if not members:
            raise ValueError("EnsembleTeacher needs at least one member.")
        self.members = nn.ModuleList(members)
        self.temperature = float(temperature)
        for m in self.members:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        # Members are permanently frozen/eval even when the LightningModule calls .train().
        super().train(mode)
        for m in self.members:
            m.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        acc = None
        for m in self.members:
            out = m(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = F.softmax(out / self.temperature, dim=1)
            acc = probs if acc is None else acc + probs
        return acc / len(self.members)

    @classmethod
    def from_checkpoints(cls, ckpt_paths, build_member, temperature: float = 2.0,
                         map_location="cpu"):
        """Build the ensemble from a list of Lightning .ckpt paths.

        Args:
            ckpt_paths: iterable of paths to the N seed checkpoints (the shipped Stage-3
                recipe's seeds — the ensemble members).
            build_member: zero-arg callable returning a fresh student module
                (e.g. lambda: ft_unetformer(pretrained=False, num_classes=6,
                decoder_channels=256)).
            temperature: see __init__.
        """
        ckpt_paths = list(ckpt_paths)
        if not ckpt_paths:
            raise ValueError(
                "No ensemble member checkpoints provided. Populate the manifest with the "
                "shipped Stage-3 recipe's per-seed checkpoints first (see the config)."
            )
        members = []
        for p in ckpt_paths:
            m = build_member()
            _load_member_weights(m, str(p), map_location=map_location)
            members.append(m)
        return cls(members, temperature=temperature)


def _load_member_weights(model: nn.Module, ckpt_path: str, map_location="cpu"):
    """Load a student's weights from a Lightning .ckpt — same key handling as
    train/train_kd.py:_load_student_weights_from_pl_ckpt (strip 'net.' and the
    torch.compile '_orig_mod.' prefix). Strict to catch a wrong/empty checkpoint loudly:
    an ensemble member silently loading random weights would corrupt the teacher.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Not a Lightning .ckpt: {ckpt_path}")
    sd = ckpt["state_dict"]
    net_sd = {k.replace("net.", "", 1): v for k, v in sd.items() if k.startswith("net.")}
    if not net_sd:
        net_sd = sd
    net_sd = {k.replace("_orig_mod.", "", 1): v for k, v in net_sd.items()}
    missing, unexpected = model.load_state_dict(net_sd, strict=False)
    # strict=False to tolerate buffer/aux-head naming, but a member that matched almost
    # nothing is a fatal misconfiguration, not a warning.
    loaded = len(model.state_dict()) - len(missing)
    if loaded < 0.5 * len(model.state_dict()):
        raise RuntimeError(
            f"Ensemble member loaded only {loaded}/{len(model.state_dict())} tensors from "
            f"{ckpt_path} — wrong checkpoint? (missing={missing[:6]})"
        )
    if missing:
        print(f"[EnsembleTeacher] {ckpt_path}: {len(missing)} missing (non-fatal):", missing[:5])
    if unexpected:
        print(f"[EnsembleTeacher] {ckpt_path}: {len(unexpected)} unexpected (non-fatal):", unexpected[:5])
