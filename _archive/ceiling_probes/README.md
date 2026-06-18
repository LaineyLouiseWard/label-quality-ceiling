# Archived: ceiling probes (negative results)

Short fine-tunes on top of the grounded Stage 4 KD-B model that did **not** beat a matched
"extra-epochs" control. Not part of the canonical pipeline; kept for reproducibility of the
manuscript's "what did not help" section.

- `stage6_confusion.py` — directional minority→Grassland cost-matrix penalty (`CONF_LAMBDA` sweep
  {0, 2, 4}). λ>0 ≤ the λ=0 control on mIoU and Settlement; Semi-natural flat. Clean null.
- `stage6_lovasz.py` — Dice→Lovász-Softmax (IoU-direct loss). Tied the control; no minority gain.

Exact numbers and interpretation (teacher-blindness ceiling — the OEM teacher has no Semi-natural
channel): `docs/results/negative_results.md`.
Grounded KD-B that these sit on top of: `docs/KD_MAPPING_GROUNDING.md`.

These reference the OLD init checkpoint name `stage5_norep` (now `stage4_kd`) and the OLD sampler
artifact `stage4_sampling_weights.tsv` (now `sampler_weights.tsv`); paths were left as-was since the
probes are not re-run. Re-grounding them would require updating those paths.
