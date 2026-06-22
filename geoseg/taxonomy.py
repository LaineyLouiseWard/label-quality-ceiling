"""
Canonical taxonomy — the single source of truth for class names, order, colours, and the
OpenEarthMap(OEM)->Biodiversity mappings used across the whole pipeline.

Every other module should DERIVE its class conventions from here rather than re-declaring them,
so the orders/indices can never silently drift (the original KD bug was a stray OEM channel order).
`scripts/verify_taxonomy_consistency.py` asserts that everything still agrees with this module.

This module is pure-Python (no torch / no project imports) so it can be imported anywhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Student (Biodiversity) taxonomy — 6 classes, background first.
# ---------------------------------------------------------------------------
STUDENT_CLASSES = (
    "Background",   # 0
    "Forest",       # 1
    "Grassland",    # 2
    "Cropland",     # 3
    "Settlement",   # 4
    "Seminatural",  # 5
)

# RGB palette aligned to STUDENT_CLASSES (index -> colour).
STUDENT_PALETTE = [
    [0, 0, 0],         # 0 Background
    [250, 62, 119],    # 1 Forest
    [168, 232, 84],    # 2 Grassland
    [242, 180, 92],    # 3 Cropland
    [59, 141, 247],    # 4 Settlement
    [255, 214, 33],    # 5 Seminatural
]

BACKGROUND_INDEX = 0
# Minority classes of interest (pixel-rare): Settlement, Semi-natural grassland.
MINORITY_INDICES = (4, 5)

# ---------------------------------------------------------------------------
# Native OpenEarthMap taxonomy — 9 output channels (8 land classes + unknown/background).
# Official OEM integer encoding: 0=unknown, 1=bareland, ... 8=building. The teacher trains
# directly on these labels, so teacher output channel i == OEM class i.
# ---------------------------------------------------------------------------
OEM_NATIVE_CLASSES = (
    "Unknown",       # 0
    "Bareland",      # 1
    "Rangeland",     # 2
    "Developed",     # 3
    "Road",          # 4
    "Tree",          # 5
    "Water",         # 6
    "Agriculture",   # 7
    "Building",      # 8
)

# ---------------------------------------------------------------------------
# OEM native index -> student index for PRE-TRAINING (hard labels). Table 1, pre-training column.
# GROUNDED (2026-06-17): argmax of the teacher's empirical OEM->target confusion on the TRAINING set
# (scripts/analysis/teacher_oem_to_gt_confusion.py; docs/KD_MAPPING_GROUNDING.md). Replaces the original
# name-based hand-map, which mis-routed the teacher's domain-confused classes (Irish "Agriculture" is
# pasture = Grassland, not Cropland; teacher "Water"/"Bareland" land on vegetation, not Background).
# Consistent BY CONSTRUCTION with the KD column (= the full soft confusion): pretrain = argmax(confusion),
# KD = full distribution -- so no "dagger" exceptions are needed anymore.
#   Old name-based hand-map was: {0:0, 1:0, 2:2, 3:4, 4:4, 5:1, 6:0, 7:3, 8:4}
# ---------------------------------------------------------------------------
OEM_TO_STUDENT_PRETRAIN = {
    0: 0,  # Unknown      -> Background    (99% GT Background)
    1: 5,  # Bareland     -> Seminatural   [CHANGED from Background] (41% GT Seminatural, plurality)
    2: 2,  # Rangeland    -> Grassland     (54% GT Grassland)
    3: 4,  # Developed    -> Settlement    (59% GT Settlement)
    4: 4,  # Road         -> Settlement    (57% GT Settlement)
    5: 1,  # Tree         -> Forest        (75% GT Forest)
    6: 2,  # Water        -> Grassland     [CHANGED from Background] (58% GT Grassland)
    7: 2,  # Agriculture  -> Grassland     [CHANGED from Cropland]   (82% GT Grassland; Irish ag = pasture)
    8: 4,  # Building     -> Settlement    (85% GT Settlement)
}


# The legacy name-based KD soft map (oem_to_student_kd) was REMOVED 2026-06-19.
# The campaign KD map is grounded in the teacher's empirical confusion:
# geoseg/utils/kd_utils.build_mapping_from_confusion("B"). Pre-training uses
# OEM_TO_STUDENT_PRETRAIN (above) = argmax of that same confusion.


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
NUM_STUDENT_CLASSES = len(STUDENT_CLASSES)   # 6
NUM_OEM_CLASSES = len(OEM_NATIVE_CLASSES)    # 9


def student_name(i: int) -> str:
    return STUDENT_CLASSES[i]


def oem_native_name(i: int) -> str:
    return OEM_NATIVE_CLASSES[i]
