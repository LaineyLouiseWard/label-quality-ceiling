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
# Bareland and Rangeland (the daggers) differ from the KD column below.
# ---------------------------------------------------------------------------
OEM_TO_STUDENT_PRETRAIN = {
    0: 0,  # Unknown      -> Background
    1: 0,  # Bareland     -> Background      (dagger: -> Seminatural under KD)
    2: 2,  # Rangeland    -> Grassland       (dagger: split under KD)
    3: 4,  # Developed    -> Settlement
    4: 4,  # Road         -> Settlement
    5: 1,  # Tree         -> Forest
    6: 0,  # Water        -> Background
    7: 3,  # Agriculture  -> Cropland
    8: 4,  # Building     -> Settlement
}


def oem_to_student_kd(alpha: float = 0.7):
    """OEM native index -> list of (student_index, weight) for KD soft targets.

    Table 1, KD column. `alpha` is the Rangeland->Grassland share (1-alpha -> Seminatural).
    Each row's weights sum to 1.0.
    """
    return {
        0: [(0, 1.0)],                      # Unknown     -> Background
        1: [(5, 1.0)],                      # Bareland    -> Seminatural      (KD-only)
        2: [(2, alpha), (5, 1.0 - alpha)],  # Rangeland   -> Grassland/Seminatural
        3: [(4, 1.0)],                      # Developed   -> Settlement
        4: [(4, 1.0)],                      # Road        -> Settlement
        5: [(1, 1.0)],                      # Tree        -> Forest
        6: [(0, 1.0)],                      # Water       -> Background
        7: [(3, 1.0)],                      # Agriculture -> Cropland
        8: [(4, 1.0)],                      # Building    -> Settlement
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
NUM_STUDENT_CLASSES = len(STUDENT_CLASSES)   # 6
NUM_OEM_CLASSES = len(OEM_NATIVE_CLASSES)    # 9


def student_name(i: int) -> str:
    return STUDENT_CLASSES[i]


def oem_native_name(i: int) -> str:
    return OEM_NATIVE_CLASSES[i]
