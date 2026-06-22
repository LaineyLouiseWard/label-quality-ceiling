# Bake-off archive

Experimental configs, scripts, and artifacts from the minority-strategy **bake-off** that
selected the shipped Stage-3 mechanism (class-balanced `clsbal` sampling, Kang et al. 2020).
None of this is on the shipped 3-stage pipeline; it is kept only as the reproducibility record
for the lever-selection decision. The companion knowledge-distillation negative result is
archived separately under [`../kd_selfdistil/`](../kd_selfdistil/).

These files were moved out of the live tree to keep `config/biodiversity/`, `scripts/`, and
`artifacts/` a clean mirror of the published paper. They reference modules/artifacts by their
**original** repo paths and are not expected to import/run in place from here without restoring
those paths — they are a record, not a runnable tree.

## Contents (original location → here)

| Original location | What it was |
|---|---|
| `config/biodiversity/stage3_armA{1,2_m0,2_m09,3,4}.py`, `_arm_common.py` | S1 sampler-sweep arms (per-class / hardness levers) |
| `config/biodiversity/stage3_copypaste.py`, `stage3_richonly.py` | copy-paste and rich-only sampler variants |
| `config/biodiversity/stage3_clsbal_recall.py`, `stage3_clsbal_recall_cp.py`, `stage3_clsbal_cp.py` | clsbal + recall-CE / copy-paste combinations |
| `scripts/bakeoff/` | bake-off gates, preflight, and read-out scripts |
| `artifacts/sampler_weights_perclass.tsv`, `sampler_weights_richonly.tsv` | sampler weights for the losing arms |

## Modules left in place (not moved)

The loss/sampler modules these configs use — `geoseg/losses/recall_ce.py`,
`geoseg/datasets/per_class_sampler.py` (and the KD pair `geoseg/losses/selfdistill.py`,
`geoseg/models/ensemble_teacher.py`) — remain under `geoseg/` so the archived configs can still
be imported if restored, mirroring how the KD archive was handled. They are inert: nothing on the
shipped path imports them (`geoseg.losses.__init__` still re-exports `recall_ce`, which is
unused by the shipped `JointLoss`).
