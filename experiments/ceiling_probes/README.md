# Ceiling probes (negative results)

Short fine-tunes on top of Stage 5 KD that did NOT beat a matched "12-more-epochs" control:
- `stage6_confusion.py` — directional minority->Grassland cost-matrix penalty (CONF_LAMBDA sweep {0,2,4}); λ=2/4 ≤ control.
- `stage6_lovasz.py` — Dice→Lovász-Softmax (IoU-direct loss); tied the control.

Not part of the canonical pipeline. Kept for reproducibility of the "what did not help" section.
See docs/CLEANUP_AND_MANUSCRIPT_PLAN_2026-06-16.md and docs/MANUSCRIPT_IMPLICATIONS_NOREP.md.
