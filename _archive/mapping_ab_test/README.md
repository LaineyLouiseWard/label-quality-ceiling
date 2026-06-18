# OEM→target KD mapping A/B test (2026-06-17, archived)

Single-seed test of two grounded KD mappings vs the hand-map control, all from the same Stage 4 checkpoint.
**Result: A≈B within seed noise; B adopted** and folded into the canonical KD config
`config/biodiversity/stage4_kd.py` (`build_mapping_from_confusion("B")`). These two configs are
archived here for reproducibility of the comparison.

- `stage5_mapA.py` — targeted grounding (fix only Bareland/Water/Agriculture rows).
- `stage5_mapB.py` — full data-driven label-transition matrix (adopted).

Result, mechanism, and decision rationale: `docs/KD_MAPPING_GROUNDING.md` and
`docs/results/negative_results.md`. (These archived configs reference the OLD init/artifact names
and are not re-run.)
