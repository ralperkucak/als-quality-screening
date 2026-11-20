"""Project-wide paths and immutable dataset constants.

All paths are resolved relative to the repository root so that no absolute path
appears anywhere else in the code base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Bulk data lives outside the repository (large, immutable, git-ignored).
# Override with the GEOQUERY_DATA_ROOT environment variable.
DATA_ROOT = Path(os.environ.get("GEOQUERY_DATA_ROOT", Path.home() / "geoquery_store"))

RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
MODEL_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "outputs"
DOC_DIR = REPO_ROOT / "docs"
EXPERIMENT_DIR = REPO_ROOT / "experiments"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MANIFEST_DIR, MODEL_DIR, OUTPUT_DIR,
           DOC_DIR, EXPERIMENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FRACTAL constants (source: dataset card + data paper, verified 2026-08-19)
# ---------------------------------------------------------------------------
CRS_EPSG = 2154                      # RGF93 / Lambert-93, planar metres
PATCH_SIDE_M = 50.0
PATCH_AREA_M2 = PATCH_SIDE_M ** 2
PATCH_AREA_KM2 = PATCH_AREA_M2 / 1e6

# Nomenclature of the 7-class FRACTAL task, in label order.
CLASS_NAMES = ("other", "ground", "vegetation", "building", "water", "bridge",
               "permanent structure")
NUM_CLASSES = len(CLASS_NAMES)

# ASPRS-like codes present in the FRACTAL LAZ files -> contiguous training ids.
# vegetation merges low/medium/high vegetation, matching the published baseline.
ASPRS_TO_TRAIN_ID = {
    1: 0,    # unclassified -> other
    2: 1,    # ground
    3: 2,    # low vegetation
    4: 2,    # medium vegetation
    5: 2,    # high vegetation
    6: 3,    # building
    9: 4,    # water
    17: 5,   # bridge
    64: 6,   # permanent structure
}

# Colour-blind safe qualitative palette, fixed once for every figure.
CLASS_COLORS = {
    "other": "#999999",
    "ground": "#b07b3e",
    "vegetation": "#2e8b57",
    "building": "#d55e00",
    "water": "#0072b2",
    "bridge": "#cc79a7",
    "permanent structure": "#7d3fbf",
}


@dataclass(frozen=True)
class Paths:
    """Convenience container so scripts do not hard-code directory layout."""

    patch_index: Path = PROCESSED_DIR / "patch_index.parquet"
    domain_mapping: Path = PROCESSED_DIR / "domain_mapping.csv"
    split_manifest: Path = MANIFEST_DIR / "lodo_splits.parquet"
    tile_cache: Path = PROCESSED_DIR / "patches"
    registry: Path = EXPERIMENT_DIR / "registry.csv"


PATHS = Paths()
