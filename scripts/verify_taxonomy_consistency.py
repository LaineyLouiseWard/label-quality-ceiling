#!/usr/bin/env python3
"""
Taxonomy consistency tripwire.

Guards against the class of bug that caused the original KD failure (a stray OEM channel order).
Run as the first pipeline stage (RUNBOOK A0): it aborts in seconds if any class order/index drifts.

It checks three things:
  1. EQUALITY PROOF (behaviour-preserving): the canonical taxonomy.py values equal the exact
     values that were in the codebase before centralisation — so nothing changed.
  2. DERIVATION: the load-bearing modules (biodiversity_dataset, kd_utils, relabel) now derive
     correctly from taxonomy.py, and the KD matrix matches the verified native-A mapping.
  3. DRIFT GUARD: the remaining scattered copies still agree with taxonomy by order/index
     (display-string variants like "Forest land" vs "Forest" are normalised away).

Exit 0 = all consistent. Exit 1 = drift detected (do not run the pipeline).

Usage:  PYTHONPATH=. python scripts/verify_taxonomy_consistency.py
"""
import sys

import geoseg.taxonomy as tax

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _norm(s):
    """Normalise a display name to a canonical key (folds 'Forest land'->'forest', 'Semi-nat.'->'seminatural')."""
    k = s.lower().strip().replace("-", "").replace(".", "").replace("_", "").replace(" ", "")
    return {"forestland": "forest", "seminaturalgrassland": "seminatural", "seminat": "seminatural"}.get(k, k)


CANON_KEYS = [_norm(c) for c in tax.STUDENT_CLASSES]  # background,forest,grassland,cropland,settlement,seminatural

# ---------------------------------------------------------------------------
# 1. EQUALITY PROOF — taxonomy.py == the exact pre-centralisation values (verified this session)
# ---------------------------------------------------------------------------
check(tuple(tax.STUDENT_CLASSES) == ("Background", "Forest", "Grassland", "Cropland", "Settlement", "Seminatural"),
      "STUDENT_CLASSES changed from the verified value")
check(list(map(list, tax.STUDENT_PALETTE)) ==
      [[0, 0, 0], [250, 62, 119], [168, 232, 84], [242, 180, 92], [59, 141, 247], [255, 214, 33]],
      "STUDENT_PALETTE changed from the verified value")
check(tuple(tax.OEM_NATIVE_CLASSES) ==
      ("Unknown", "Bareland", "Rangeland", "Developed", "Road", "Tree", "Water", "Agriculture", "Building"),
      "OEM_NATIVE_CLASSES changed from the verified native-A order")
check(dict(tax.OEM_TO_STUDENT_PRETRAIN) == {0: 0, 1: 0, 2: 2, 3: 4, 4: 4, 5: 1, 6: 0, 7: 3, 8: 4},
      "OEM_TO_STUDENT_PRETRAIN changed from the verified pre-training map")
check(tuple(tax.MINORITY_INDICES) == (4, 5) and tax.BACKGROUND_INDEX == 0,
      "MINORITY_INDICES / BACKGROUND_INDEX changed")

# KD soft map (alpha=0.7) must equal the verified native-A targets
EXPECTED_KD = {0: {0: 1.0}, 1: {5: 1.0}, 2: {2: 0.7, 5: 0.3}, 3: {4: 1.0}, 4: {4: 1.0},
               5: {1: 1.0}, 6: {0: 1.0}, 7: {3: 1.0}, 8: {4: 1.0}}
kd = {o: {s: round(w, 6) for s, w in targets} for o, targets in tax.oem_to_student_kd(0.7).items()}
check(kd == EXPECTED_KD, f"oem_to_student_kd(0.7) changed from verified KD map: {kd}")

# ---------------------------------------------------------------------------
# 2. DERIVATION — load-bearing modules derive from taxonomy; KD matrix matches native-A
# ---------------------------------------------------------------------------
from geoseg.datasets.biodiversity_dataset import CLASSES as BIO_CLASSES, PALETTE as BIO_PALETTE
check(tuple(BIO_CLASSES) == tuple(tax.STUDENT_CLASSES), "biodiversity_dataset.CLASSES != taxonomy")
check(list(map(list, BIO_PALETTE)) == list(map(list, tax.STUDENT_PALETTE)), "biodiversity_dataset.PALETTE != taxonomy")

from geoseg.utils.kd_utils import OEM_CLASSES, NEW_CLASSES, REMAP_OUTPUT_CLASSES, create_mapping_matrix
check(tuple(OEM_CLASSES) == tuple(tax.OEM_NATIVE_CLASSES), "kd_utils.OEM_CLASSES != taxonomy")
check(tuple(NEW_CLASSES) == tuple(tax.STUDENT_CLASSES), "kd_utils.NEW_CLASSES != taxonomy")
check(tuple(REMAP_OUTPUT_CLASSES) == tuple(tax.STUDENT_CLASSES), "kd_utils.REMAP_OUTPUT_CLASSES != student CLASSES")

M = create_mapping_matrix(0.7)
check(tuple(M.shape) == (9, 6), f"KD matrix shape {tuple(M.shape)} != (9,6)")
row_sums = [round(float(M[i].sum()), 6) for i in range(M.shape[0])]
check(all(rs == 1.0 for rs in row_sums), f"KD matrix rows do not sum to 1: {row_sums}")
nonzero = {(i, j): round(float(M[i, j]), 6) for i in range(9) for j in range(6) if float(M[i, j]) != 0.0}
EXPECTED_M = {(0, 0): 1.0, (1, 5): 1.0, (2, 2): 0.7, (2, 5): 0.3, (3, 4): 1.0,
              (4, 4): 1.0, (5, 1): 1.0, (6, 0): 1.0, (7, 3): 1.0, (8, 4): 1.0}
check(nonzero == EXPECTED_M, f"KD matrix entries changed from verified native-A: {nonzero}")

from scripts.data_prep.relabel_oem_taxonomy import OEM_ID_TO_TARGET6
check(dict(OEM_ID_TO_TARGET6) == dict(tax.OEM_TO_STUDENT_PRETRAIN), "relabel OEM_ID_TO_TARGET6 != taxonomy")

# Pretrain and KD maps must agree on every NON-dagger class (all but bareland=1 and rangeland=2)
kd_argmax = {o: max(t, key=lambda kv: kv[1])[0] for o, t in tax.oem_to_student_kd(0.7).items()}
for o in range(9):
    if o in (1, 2):
        continue
    check(tax.OEM_TO_STUDENT_PRETRAIN[o] == kd_argmax[o],
          f"pretrain/KD disagree on non-dagger OEM class {o}")

# ---------------------------------------------------------------------------
# 3. DRIFT GUARD — remaining scattered copies still agree by order/index
# ---------------------------------------------------------------------------
def guard_import(label, fn):
    try:
        fn()
    except ImportError as e:
        print(f"  (skipped {label}: {e})")


def _check_compute_metrics():
    from evaluation.compute_metrics import CLASS_NAMES_6
    check([_norm(x) for x in CLASS_NAMES_6] == CANON_KEYS,
          f"compute_metrics.CLASS_NAMES_6 order/identity drifted: {CLASS_NAMES_6}")


def _check_analysis_utils():
    from scripts.analysis import utils as au
    check([_norm(x) for x in au.CLASS_NAMES] == CANON_KEYS,
          f"analysis/utils.CLASS_NAMES drifted: {au.CLASS_NAMES}")
    check(list(au.MINORITY_INDICES) == list(tax.MINORITY_INDICES),
          f"analysis/utils.MINORITY_INDICES drifted: {au.MINORITY_INDICES}")


guard_import("evaluation.compute_metrics", _check_compute_metrics)
guard_import("scripts.analysis.utils", _check_analysis_utils)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
if FAILURES:
    print(f"TAXONOMY CONSISTENCY: FAIL ({len(FAILURES)} of {CHECKS} checks failed)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"TAXONOMY CONSISTENCY: PASS ({CHECKS} checks, derived == verified values; no drift).")
sys.exit(0)
