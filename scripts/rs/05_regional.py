"""Regional accuracy, the distribution of tile quality and what separates the two.

A region reports one accuracy figure and an inspection programme works on tiles,
and the two disagree. The decomposition here accounts for the whole of that
disagreement with two terms that can be read off a table: the covariance between
how much ground a class occupies in a tile and how well that tile is segmented,
and the reweighting caused by tiles averaging over the classes they happen to
contain rather than over the classes of the region.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs.store import (CLASS_NAMES, FOLDS,  # noqa: E402
                                   load_patch_stats, load_units)

OUT = ROOT / "outputs" / "rs"


def occupancy(confusion: np.ndarray) -> np.ndarray:
    """Points a class occupies on the union of reference and prediction, (U, C)."""
    ref = confusion.sum(axis=2)
    pred = confusion.sum(axis=1)
    tp = np.einsum("ucc->uc", confusion)
    return ref + pred - tp


def decompose(units) -> dict:
    """Split the gap between pooled and tile-mean quality into its two parts."""
    c = units.confusion
    iou = cr.class_iou(c)
    occ = occupancy(c)
    present = occ > 0

    pooled_confusion = c.sum(axis=0, keepdims=True)
    pooled_iou = cr.class_iou(pooled_confusion)[0]
    region_classes = np.flatnonzero(cr.presence(pooled_confusion, "union")[0])

    per_class = []
    for k in region_classes:
        m = present[:, k]
        weights = occ[m, k] / occ[m, k].sum()
        tile_iou = iou[m, k]
        per_class.append({
            "class": CLASS_NAMES[k],
            "tiles_present": int(m.sum()),
            "pooled_iou": float(pooled_iou[k]),
            "mean_tile_iou": float(tile_iou.mean()),
            "covariance_term": float((weights - 1.0 / m.sum()) @ tile_iou),
        })

    frame = pd.DataFrame(per_class)
    n_region = len(region_classes)
    covariance = float(frame["covariance_term"].sum() / n_region)
    same_set = float(frame["mean_tile_iou"].sum() / n_region)
    tile_mean = float(cr.mean_iou(c, "union").mean())
    return {
        "pooled_miou": float(pooled_iou[region_classes].mean()),
        "tile_mean_miou": tile_mean,
        "covariance_term": covariance,
        "class_presence_term": same_set - tile_mean,
        "per_class": frame,
    }


def transfer_by_class() -> pd.DataFrame:
    """Per-class accuracy on the source validation data and on the held-out region."""
    records = []
    for fold in FOLDS:
        for family in ("source", "target"):
            stats = load_patch_stats(fold, family)
            pooled = stats.confusion.sum(axis=0)[None, :, :]
            iou = cr.class_iou(pooled)[0]
            f1 = cr.class_f1(pooled)[0]
            share = pooled.sum(axis=2)[0] / pooled.sum()
            for k, name in enumerate(CLASS_NAMES):
                records.append({"region": fold, "family": family,
                                "class": name, "iou": float(iou[k]),
                                "f1": float(f1[k]), "share": float(share[k])})
    return pd.DataFrame(records)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_class = [], []
    for f in FOLDS:
        units = load_units(f)
        d = decompose(units)
        quality = cr.mean_iou(units.confusion, "union")
        oa = cr.overall_accuracy(units.confusion)
        gap = d["pooled_miou"] - d["tile_mean_miou"]
        rows.append({
            "region": f,
            "tiles": units.n_units,
            "patches": int(units.n_patches.sum()),
            "points_million": round(units.n_points.sum() / 1e6, 1),
            "pooled_miou": d["pooled_miou"],
            "tile_mean_miou": d["tile_mean_miou"],
            "gap": gap,
            "covariance_term": d["covariance_term"],
            "class_presence_term": d["class_presence_term"],
            "residual": gap - d["covariance_term"] - d["class_presence_term"],
            "pooled_oa": float(np.einsum("ucc->", units.confusion)
                               / units.confusion.sum()),
            "tile_miou_p10": float(np.percentile(quality, 10)),
            "tile_miou_median": float(np.median(quality)),
            "tile_miou_p90": float(np.percentile(quality, 90)),
            "tile_oa_median": float(np.median(oa)),
        })
        frame = d["per_class"].assign(region=f)
        per_class.append(frame)

    transfer = transfer_by_class()
    transfer.to_csv(OUT / "transfer_by_class.csv", index=False)
    wide = transfer.pivot_table(index="class", columns="family",
                                values=["iou", "share"], aggfunc="mean")
    wide = wide.reindex(CLASS_NAMES)
    wide["loss"] = wide[("iou", "source")] - wide[("iou", "target")]
    wide.to_csv(OUT / "transfer_summary.csv")
    print()
    print("per-class transfer, averaged over the five folds")
    print(wide.round(4).to_string())

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "regions.csv", index=False)
    pd.concat(per_class).to_csv(OUT / "regions_per_class.csv", index=False)
    print(table.round(4).to_string(index=False))
    print("\nlargest residual of the decomposition: %.2e"
          % table["residual"].abs().max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
