# Archived: replication-era pipeline (not used)

These configs and scripts implemented an earlier 5-stage ablation that included a **static
minority-replication** stage (`train_rep`: minority tiles physically duplicated on disk). The
published pipeline is **replication-free** (no-replication, 4-stage): the hard × minority
*sampler* is the single imbalance mechanism, so static duplication was dropped.

Kept only for provenance — **not** part of the reproducible pipeline (`RUNBOOK.sh`) and not
referenced by any figure, table, or analysis. Do not run.

| File | Was | Superseded by |
|---|---|---|
| `stage2_replication.py` | Stage 2: train on replicated `train_rep` | dropped (no replication) |
| `stage3b_finetune.py` | OEM finetune on `train_rep` | `config/biodiversity/stage2b_oem_finetune.py` (on `train`) |
| `stage4_sampling.py` | Sampler on `train_rep` | `config/biodiversity/stage3_sampler.py` (on `train`) |
| `stage5_kd.py` | KD on `train_rep`, name-based mapping | `config/biodiversity/stage4_kd.py` (grounded KD-B, on `train`) |
| `replicate_minority_samples.py` | builds `train_rep` | dropped |
| `compute_replication_exposure.py` | quantified replication vs sampler exposure | dropped (motivated the drop) |
| `compare_norep.py` | compared no-rep arm vs replicated arm | dropped (no-rep is now canonical) |

Rationale for dropping replication: `docs/MANUSCRIPT_IMPLICATIONS_NOREP.md`.
Grounded mappings used by the live pipeline: `docs/KD_MAPPING_GROUNDING.md`.
