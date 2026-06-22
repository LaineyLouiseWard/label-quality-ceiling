# Archived: Stage-4 Knowledge Distillation & Self-Distillation (2026-06-22)

These are the **retired Stage-4 KD experiments**, moved out of the active pipeline so the repo is a clean
**3-stage** mirror (baseline → OEM transfer → clsbal sampler). They are **preserved, not deleted**, so the
methods can be described and (if needed) re-run for the **Discussion** section. Verdict + numbers:
`docs/SELFDISTIL_VERDICT_2026-06-22.md`.

## Why retired
- **OEM EfficientNet-B4 KD teacher** (`stage4_kd*`): the 9→6 cross-taxonomy teacher was too weak to distil
  up into the student (controlled contribution ≈ +0.2 ± 0.2).
- **In-domain self-distillation** (`stage4_selfdistil`): 5-seed binding test vs a step-matched no-KD control
  gave **−0.54 pp mIoU [−0.90, −0.18]** (worse than training the same epochs without KD; Semi-nat −1.51),
  and no T×α grid cell beat the control. Dropped.

## What is here
- `config/` — the 5 Stage-4 configs: `stage4_kd.py`, `stage4_kd_clsbal.py`, `stage4_selfdistil.py`,
  `stage4null_nokd.py`, `stage4null_nokd_clsbal.py`. (`stage4null_nokd_clsbal` is the step-matched no-KD
  control whose 90-epoch run first measured the warm-restart gain that is now folded into the Stage-3
  scheduler — see the verdict doc.)
- `scripts_bakeoff/` — `eval_ensemble.py`, `selfdistil_readout.py`, `s2_sweep_readout.py`.
- `sonic/` — the KD/self-distil SLURM launchers + push/fetch (`8*selfdistil*`, `5_submit_kd_gate`,
  `ensemble_eval*`, `9_push_selfdistil`, `9b_fetch_selfdistil`).

## Kept in place (NOT archived) — needed by the live pipeline or for re-runnability
- `geoseg/losses/selfdistill.py`, `geoseg/models/ensemble_teacher.py` — KD library modules (only imported
  by the archived configs; left in `geoseg/` so the archived configs stay importable/re-runnable).
- `geoseg/taxonomy.py`, `geoseg/utils/kd_utils.py` — **load-bearing for Stage 2** (the dataset and the
  grounded OEM→student relabel use them; `kd_utils` holds the shared mapping-matrix utilities). Do NOT move.
- `train/train_kd.py` — KD training entrypoint (kept so the archived configs can run).
- `scripts/bakeoff/s1_sweep_readout.py`, `config/biodiversity/stage3null_nosampler.py`, `sonic/4*bakeoff*`,
  `sonic/9_submit_s1_sampler_sweep.slurm` — these are **sampler** (Stage-3) artefacts, part of the 3-stage
  story, not KD.

## To re-run an archived KD experiment (for the Discussion)
The library modules and `train_kd.py` are still in place, so e.g. `stage4_selfdistil.py` runs unchanged if
copied back to `config/biodiversity/` (it reads `ENSEMBLE_MANIFEST`, `STUDENT_INIT_CKPT`). The evidence
(`evaluation/evaluation_results/selfdistil/`, `analysis/clsbal_results/`) is retained in the repo.
