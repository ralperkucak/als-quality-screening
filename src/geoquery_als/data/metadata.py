"""Read the FRACTAL GeoPackage into a tidy patch-level table.

`fractal.gpkg` ships with the dataset and describes all 100,000 patches
(geometry, split, per-class point counts, acquisition years and landscape
descriptors). Reading it costs 31 MB and gives a complete picture of the
benchmark before a single point cloud is downloaded.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ASPRS_TO_TRAIN_ID, CLASS_NAMES, PATCH_AREA_M2

GPKG_TABLE = "fractal_for_huggingface"
RTREE_TABLE = f"rtree_{GPKG_TABLE}_geom"

# gpkg column -> FRACTAL training class name
_COUNT_COLUMNS = {
    "numpts_unclassified": "other",
    "numpts_ground": "ground",
    "numpts_low_vegetation": "vegetation",
    "numpts_medium_vegetation": "vegetation",
    "numpts_high_vegetation": "vegetation",
    "numpts_buildings": "building",
    "numpts_water": "water",
    "numpts_bridge": "bridge",
    "numpts_permanentstructures": "permanent structure",
}

_DESCRIPTOR_COLUMNS = [
    "BUILD", "BUILD_GREENHOUSE", "BUILD_BIG", "BRIDGE", "WATER", "WATER_SURFACE",
    "PERMSTRUCT", "PERMSTRUCT_PYLON", "PERMSTRUCT_ANTENNA", "OTHER", "OTHER_PARKING",
    "OTHER_HIGHWAY", "FOREST", "HIGHSLOPE1", "HIGHSLOPE2", "MOUTAIN", "WATER_ONLY",
    "SEASHORE", "URBAN",
]


def read_patch_metadata(gpkg_path: Path) -> pd.DataFrame:
    """Return one row per patch with centroid, split, class counts and descriptors.

    Centroids are taken from the GeoPackage R-tree envelope, which for these
    axis-aligned 50 m squares is exact and avoids parsing WKB geometry.
    """
    con = sqlite3.connect(str(gpkg_path))
    try:
        attrs = pd.read_sql_query(f"SELECT * FROM {GPKG_TABLE}", con)  # noqa: S608
        env = pd.read_sql_query(
            f"SELECT id AS fid, minx, maxx, miny, maxy FROM {RTREE_TABLE}", con  # noqa: S608
        )
    finally:
        con.close()

    attrs = attrs.drop(columns=[c for c in ("geom",) if c in attrs.columns])
    df = attrs.merge(env, on="fid", how="left", validate="one_to_one")
    if df[["minx", "miny"]].isna().any().any():
        raise ValueError("patch envelopes missing from the GeoPackage R-tree index")

    df["centroid_x"] = 0.5 * (df["minx"] + df["maxx"])
    df["centroid_y"] = 0.5 * (df["miny"] + df["maxy"])
    df["extent_x_m"] = df["maxx"] - df["minx"]
    df["extent_y_m"] = df["maxy"] - df["miny"]

    # patch_id is "<tile_x>_<tile_y>-<serial>", tile coordinates in km.
    tile = df["patch_id"].str.split("-", n=1, expand=True)
    df["tile_id"] = tile[0]
    df["patch_serial"] = tile[1]
    tile_xy = df["tile_id"].str.split("_", n=1, expand=True)
    df["tile_x_km"] = pd.to_numeric(tile_xy[0], errors="coerce")
    df["tile_y_km"] = pd.to_numeric(tile_xy[1], errors="coerce")

    for name in CLASS_NAMES:
        cols = [c for c, n in _COUNT_COLUMNS.items() if n == name and c in df.columns]
        df[f"n_{_slug(name)}"] = df[cols].fillna(0).sum(axis=1).astype("int64")

    count_cols = [f"n_{_slug(n)}" for n in CLASS_NAMES]
    df["n_points_labelled"] = df[count_cols].sum(axis=1)
    df["point_density"] = df["numpts"] / PATCH_AREA_M2
    for name in CLASS_NAMES:
        c = f"n_{_slug(name)}"
        df[f"frac_{_slug(name)}"] = np.where(
            df["n_points_labelled"] > 0, df[c] / df["n_points_labelled"].clip(lower=1), 0.0
        )

    for c in _DESCRIPTOR_COLUMNS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype("int8")

    df["laz_name"] = df["split"].str.upper() + "-" + df["patch_id"] + ".laz"
    return df.drop(columns=["minx", "maxx", "miny", "maxy"])


def _slug(name: str) -> str:
    return name.replace(" ", "_")


CLASS_COUNT_COLUMNS = [f"n_{_slug(n)}" for n in CLASS_NAMES]
CLASS_FRAC_COLUMNS = [f"frac_{_slug(n)}" for n in CLASS_NAMES]

__all__ = [
    "read_patch_metadata",
    "CLASS_COUNT_COLUMNS",
    "CLASS_FRAC_COLUMNS",
    "ASPRS_TO_TRAIN_ID",
]
