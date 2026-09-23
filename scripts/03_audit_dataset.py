"""Phase 2: build the patch index and audit FRACTAL from its GeoPackage.

Outputs
-------
data/processed/patch_index.parquet      one row per patch, all 100,000 patches
data/processed/domain_mapping.csv       patch -> recovered spatial domain
outputs/audit/dataset_audit.json        machine-readable audit summary
outputs/audit/data_quality_report.json  quality checks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoquery_als.config import (CLASS_NAMES, PATCH_AREA_KM2, PATCH_SIDE_M, PATHS,  # noqa: E402
                                 RAW_DIR, OUTPUT_DIR)
from geoquery_als.data.domains import assign_domain_names, recover_domains  # noqa: E402
from geoquery_als.data.metadata import (CLASS_COUNT_COLUMNS, CLASS_FRAC_COLUMNS,  # noqa: E402
                                        read_patch_metadata)


def main() -> int:
    gpkg = RAW_DIR / "fractal.gpkg"
    if not gpkg.exists():
        raise SystemExit(f"missing {gpkg}; run scripts/01_download_fractal.py first")

    df = read_patch_metadata(gpkg)
    print(f"patches read: {len(df):,}")

    xy = df[["centroid_x", "centroid_y"]].to_numpy()
    rec = recover_domains(xy)
    print(f"connected components at {rec.link_threshold_m/1000:.0f} km: {rec.n_components}")
    print(f"  max intra-domain NN distance : {rec.max_intra_domain_nn_distance_m/1000:.2f} km")
    print(f"  min inter-domain distance    : {rec.min_inter_domain_distance_m/1000:.2f} km")
    df = assign_domain_names(df, rec.labels)

    quality = _quality_checks(df)
    audit = _audit(df, rec)

    PATHS.patch_index.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PATHS.patch_index, index=False)
    df[["patch_id", "split", "domain", "centroid_x", "centroid_y", "tile_id"]].to_csv(
        PATHS.domain_mapping, index=False)

    (OUTPUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "audit" / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "audit" / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8")

    print(json.dumps(audit["per_domain"], indent=2))
    print("quality issues:", {k: v for k, v in quality.items() if v not in (0, [], {})})
    return 0


def _quality_checks(df: pd.DataFrame) -> dict:
    dup_id = df["patch_id"].duplicated().sum()
    dup_xy = df.duplicated(["centroid_x", "centroid_y"]).sum()
    bad_extent = int((~np.isclose(df["extent_x_m"], PATCH_SIDE_M, atol=1.0)).sum()
                     + (~np.isclose(df["extent_y_m"], PATCH_SIDE_M, atol=1.0)).sum())
    zero_pts = int((df["numpts"] <= 0).sum())
    label_mismatch = int((df["n_points_labelled"] != df["numpts"]).sum())
    # split disjointness on identifiers and on geometry
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia = set(map(tuple, df.loc[df.split == a, ["centroid_x", "centroid_y"]].to_numpy()))
        ib = set(map(tuple, df.loc[df.split == b, ["centroid_x", "centroid_y"]].to_numpy()))
        overlaps[f"{a}_vs_{b}_shared_centroids"] = len(ia & ib)
    return {
        "duplicated_patch_ids": int(dup_id),
        "duplicated_centroids": int(dup_xy),
        "patches_with_unexpected_extent": bad_extent,
        "zero_point_patches": zero_pts,
        "patches_where_class_counts_differ_from_numpts": label_mismatch,
        "split_geometry_overlaps": overlaps,
        "missing_values_per_column": {c: int(df[c].isna().sum())
                                      for c in df.columns if df[c].isna().any()},
    }


def _audit(df: pd.DataFrame, rec) -> dict:
    per_domain = {}
    for dom, g in df.groupby("domain"):
        counts = g[CLASS_COUNT_COLUMNS].sum()
        per_domain[dom] = {
            "n_patches": int(len(g)),
            "n_patches_by_split": g["split"].value_counts().to_dict(),
            "area_km2": round(len(g) * PATCH_AREA_KM2, 3),
            "n_points": int(g["numpts"].sum()),
            "mean_density_pts_per_m2": round(float(g["point_density"].mean()), 2),
            "median_density_pts_per_m2": round(float(g["point_density"].median()), 2),
            "elevation_m": {"p05": round(float(g["patch_elevation"].quantile(0.05)), 1),
                            "median": round(float(g["patch_elevation"].median()), 1),
                            "p95": round(float(g["patch_elevation"].quantile(0.95)), 1)},
            "elevation_gain_m_median": round(float(g["patch_elevation_gain"].median()), 2),
            "lidar_years": g["lidar_year_of_acquisition"].value_counts().to_dict(),
            "ortho_years": g["orthos_year_of_acquisition"].value_counts().to_dict(),
            "class_point_fraction": {n: round(float(counts[f"n_{n.replace(' ', '_')}"]
                                                    / counts.sum()), 5)
                                     for n in CLASS_NAMES},
            "patches_containing_class": {
                n: int((g[f"n_{n.replace(' ', '_')}"] > 0).sum()) for n in CLASS_NAMES},
            "bbox": {"xmin": float(g.centroid_x.min()), "xmax": float(g.centroid_x.max()),
                     "ymin": float(g.centroid_y.min()), "ymax": float(g.centroid_y.max())},
            "extent_km": {"x": round(float(g.centroid_x.max() - g.centroid_x.min()) / 1000, 1),
                          "y": round(float(g.centroid_y.max() - g.centroid_y.min()) / 1000, 1)},
        }
    total = df[CLASS_COUNT_COLUMNS].sum()
    return {
        "n_patches": int(len(df)),
        "n_points": int(df["numpts"].sum()),
        "total_area_km2": round(len(df) * PATCH_AREA_KM2, 2),
        "splits": df["split"].value_counts().to_dict(),
        "sampling_type": df["sampling_type"].value_counts().to_dict(),
        "domain_recovery": {
            "n_components": rec.n_components,
            "link_threshold_m": rec.link_threshold_m,
            "min_inter_domain_distance_m": rec.min_inter_domain_distance_m,
            "max_intra_domain_nn_distance_m": rec.max_intra_domain_nn_distance_m,
        },
        "global_class_point_fraction": {
            n: round(float(total[f"n_{n.replace(' ', '_')}"] / total.sum()), 5)
            for n in CLASS_NAMES},
        "per_domain": per_domain,
    }


if __name__ == "__main__":
    raise SystemExit(main())
