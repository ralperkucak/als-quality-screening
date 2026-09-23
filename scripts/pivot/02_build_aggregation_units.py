"""Group patches into spatial aggregation units.

A unit is a square cell of the Lambert-93 grid together with the patches whose
centroid falls inside it. It is not a mapped area: the patches sample the cell
rather than cover it, typically about one percent of it, so the recorded
coverage is carried alongside every unit and the term deployment area is avoided
(decision P13).

Three scales are built. The 2 km scale is primary because it is the one that
holds enough points for class-level metrics while still leaving enough units per
domain to measure anything; the measurement behind that choice is in P04. The
1 km and 4 km scales exist so the sensitivity of every conclusion to the choice
can be reported.

Two families of unit are produced:

  target   pool and test patches of the held-out domain, for the reliability
           evaluation of the source-only model. The pool is included because the
           source-only model never saw its labels, which quadruples the number
           of units at no cost (P05).
  source   validation patches of the four source domains, which is where every
           calibrator is fitted and where no target information may enter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoquery_als.config import (MANIFEST_DIR, OUTPUT_DIR, PATCH_AREA_KM2,  # noqa: E402
                                 PATHS)
from geoquery_als.data.store import PatchStore  # noqa: E402

SCALES_M = (1000, 2000, 4000)
PRIMARY_SCALE_M = 2000
MIN_PATCHES = 8          # about 131,000 points; drops 9 per cent of target units (P17)
TARGET_MEDIAN_PATCHES = 17   # measured on the primary target scale, matched by P17


def assign_units(xy: np.ndarray, scale_m: int) -> np.ndarray:
    """Grid cell index of each patch, as a single integer key per cell."""
    cell = np.floor(xy / scale_m).astype(np.int64)
    # A stable scalar key. The offset keeps negative eastings from colliding.
    return (cell[:, 0] + 2_000_000) * 4_000_000 + (cell[:, 1] + 2_000_000)


def group_by_patch_count(index: pd.DataFrame, patch_ids: list[str], fold: str,
                         seed: int = 20260823) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contiguous groups of patches inside each source domain (decision P17).

    Source validation patches are spread across four whole domains, so a 2 km
    grid produced three to six units per fold and no usable calibration set. The
    grouping instead matches the patch count of the target units, which is what
    calibration transfer actually needs: the same statistical support behind each
    unit-level metric. The groups are far wider geographically than the target
    units and that is recorded as a limitation rather than hidden.
    """
    from sklearn.cluster import KMeans

    sub = index.loc[patch_ids, ["centroid_x", "centroid_y", "domain"]]
    keys = np.empty(len(sub), dtype=np.int64)
    offset = 0
    for domain, block in sub.groupby("domain", sort=True):
        xy = block[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
        k = max(1, int(round(len(block) / TARGET_MEDIAN_PATCHES)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(xy)
        keys[sub.index.get_indexer(block.index)] = labels + offset
        offset += k
    frame = pd.DataFrame({"patch_id": sub.index.to_numpy(), "unit_key": keys,
                          "centroid_x": sub["centroid_x"].to_numpy(),
                          "centroid_y": sub["centroid_y"].to_numpy()})
    grouped = frame.groupby("unit_key")
    units = grouped.agg(n_patches=("patch_id", "size"),
                        unit_x=("centroid_x", "mean"),
                        unit_y=("centroid_y", "mean")).reset_index()
    # How far apart the members of a source unit are, so the limitation is measurable.
    spread = grouped.apply(
        lambda g: float(np.hypot(g.centroid_x - g.centroid_x.mean(),
                                 g.centroid_y - g.centroid_y.mean()).max()) / 1000.0,
        include_groups=False)
    units["spread_km"] = units["unit_key"].map(spread)
    units["fold"], units["family"], units["scale_m"] = fold, "source", PRIMARY_SCALE_M
    units["sampled_km2"] = units["n_patches"] * PATCH_AREA_KM2
    units["cell_km2"] = np.nan
    units["coverage_fraction"] = np.nan
    units["unit_id"] = (fold + "_source_matched_" + units["unit_key"].astype(str))
    return units, frame


def build(index: pd.DataFrame, patch_ids: list[str], scale_m: int, fold: str,
          family: str) -> pd.DataFrame:
    sub = index.loc[patch_ids, ["centroid_x", "centroid_y"]]
    xy = sub.to_numpy(dtype=float)
    keys = assign_units(xy, scale_m)
    frame = pd.DataFrame({"patch_id": sub.index.to_numpy(), "unit_key": keys,
                          "centroid_x": xy[:, 0], "centroid_y": xy[:, 1]})
    grouped = frame.groupby("unit_key")
    units = grouped.agg(n_patches=("patch_id", "size"),
                        unit_x=("centroid_x", "mean"),
                        unit_y=("centroid_y", "mean")).reset_index()
    units["fold"] = fold
    units["family"] = family
    units["scale_m"] = scale_m
    units["sampled_km2"] = units["n_patches"] * PATCH_AREA_KM2
    units["cell_km2"] = (scale_m / 1000.0) ** 2
    units["coverage_fraction"] = units["sampled_km2"] / units["cell_km2"]
    units["unit_id"] = (units["fold"] + "_" + family + "_" + str(scale_m) + "_"
                        + units["unit_key"].astype(str))
    return units, frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-patches", type=int, default=MIN_PATCHES)
    args = ap.parse_args()

    out_dir = OUTPUT_DIR / "pivot" / "units"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(PATHS.patch_index).set_index("patch_id")
    splits = pd.read_parquet(MANIFEST_DIR / "lodo_splits.parquet")
    available = set(PatchStore().available_patches())

    unit_rows, member_rows = [], []
    for fold in sorted(splits["fold"].unique()):
        sel = splits[splits.fold == fold]
        families = {
            "target": sel[sel.role.isin(["target_pool", "target_test"])]["patch_id"],
            "source": sel[sel.role == "source_val"]["patch_id"],
        }
        for family, ids in families.items():
            ids = [p for p in ids if p in available]
            if not ids:
                continue
            if family == "source":
                units, members = group_by_patch_count(index, ids, fold)
                keep = units[units.n_patches >= args.min_patches]
                members = members[members.unit_key.isin(set(keep.unit_key))].copy()
                members["fold"], members["family"] = fold, "source"
                members["scale_m"] = PRIMARY_SCALE_M
                members["unit_id"] = (fold + "_source_matched_"
                                      + members["unit_key"].astype(str))
                unit_rows.append(keep)
                member_rows.append(members)
                continue
            for scale in SCALES_M:
                units, members = build(index, ids, scale, fold, family)
                keep = units[units.n_patches >= args.min_patches]
                members = members[members.unit_key.isin(set(keep.unit_key))].copy()
                members["fold"], members["family"] = fold, family
                members["scale_m"] = scale
                members["unit_id"] = (fold + "_" + family + "_" + str(scale) + "_"
                                      + members["unit_key"].astype(str))
                unit_rows.append(keep)
                member_rows.append(members)

    units = pd.concat(unit_rows, ignore_index=True)
    members = pd.concat(member_rows, ignore_index=True)
    units.to_parquet(out_dir / "units.parquet", index=False)
    members.to_parquet(out_dir / "unit_members.parquet", index=False)

    # The roles a patch may play differ between families, so record which
    # patches need inference at all. A patch can belong to several scales.
    needed = (members.groupby(["fold", "family"])["patch_id"].unique()
              .apply(lambda a: sorted(a.tolist())).to_dict())
    (out_dir / "patches_needing_inference.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}": v for k, v in needed.items()}, indent=1),
        encoding="utf-8")

    if "spread_km" not in units:
        units["spread_km"] = np.nan
    summary = (units.groupby(["family", "scale_m"])
               .agg(units=("unit_id", "size"),
                    median_patches=("n_patches", "median"),
                    min_patches=("n_patches", "min"),
                    median_coverage=("coverage_fraction", "median"),
                    median_spread_km=("spread_km", "median"))
               .reset_index())
    print(summary.to_string(index=False))
    print(f"\nprimary scale {PRIMARY_SCALE_M} m, minimum {args.min_patches} patches")
    for family in ("target", "source"):
        s = units[(units.family == family) & (units.scale_m == PRIMARY_SCALE_M)]
        per_fold = s.groupby("fold").size().to_dict()
        print(f"  {family}: {len(s)} units, per fold {per_fold}")
    print(f"\nwrote {out_dir / 'units.parquet'} and unit_members.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
