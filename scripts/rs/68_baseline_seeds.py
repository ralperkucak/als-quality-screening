"""Supervised baselines repeated for every training run.

61_baseline_learning_curve.py compares the class-averaged score with ridge
regression and gradient boosting trained on all regrouped source tiles, for the
first training run of each region. This script repeats the full-data
comparison for the two further runs, with the same regroupings of the source
validation patches (they depend on patch positions only), and reports the
paired recall difference with a hierarchical bootstrap over regions and tiles.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rs"))

from geoquery_als.config import PATHS  # noqa: E402
from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import (BASE, FOLDS, group_patches,  # noqa: E402
                                   load_patch_stats, load_units)

lc = importlib.import_module("61_baseline_learning_curve")

OUT = ROOT / "outputs" / "rs"
RUNS = ("randlanet", "randlanet_s2", "randlanet_s3")
BUDGET = 0.20
DRAWS = 4000
SEED = 20261119


def source_tiles(fold: str, backbone: str):
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    members = members[(members["fold"] == fold) & (members["family"] == "source")]
    index = pd.read_parquet(PATHS.patch_index, columns=["patch_id", "domain"])
    members = members.merge(index, on="patch_id", how="left")
    stats = load_patch_stats(fold, "source", backbone)
    parts = [load_units(fold, family="source", backbone=backbone)]
    rng = np.random.default_rng(SEED + 3)
    for g in range(lc.GROUPINGS):
        size = int(rng.integers(9, 26))
        rows = []
        for domain, block in members.groupby("domain", sort=True):
            xy = block[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            k = max(1, int(round(len(block) / size)))
            lab = KMeans(n_clusters=k, n_init=4, random_state=int(rng.integers(1e9))
                         ).fit_predict(xy)
            rows.append(pd.DataFrame({"patch_id": block["patch_id"].to_numpy(),
                                      "unit_id": [f"{fold}_g{g}_{domain}_{v}" for v in lab]}))
        parts.append(group_patches(stats, pd.concat(rows), fold, min_patches=8))
    return parts


def recall_hits(quality, score, k):
    worst = np.zeros(len(quality), bool)
    worst[mt.worst_set(quality, k)] = True
    picked = np.argsort(score, kind="stable")[:k]
    return worst[picked].sum()


def paired_bootstrap(per_region, a, b, rng):
    """Mean recall difference a - b, resampling regions and then tiles."""
    names = list(per_region)
    diffs = np.empty(DRAWS)
    for d in range(DRAWS):
        vals = []
        for r in rng.choice(len(names), size=len(names), replace=True):
            q, s = per_region[names[r]]["quality"], per_region[names[r]]
            idx = rng.integers(0, len(q), len(q))
            k = mt.budget_size(len(q), BUDGET)
            vals.append((recall_hits(q[idx], s[a][idx], k)
                         - recall_hits(q[idx], s[b][idx], k)) / k)
        diffs[d] = np.mean(vals)
    return np.quantile(diffs, [0.025, 0.975])


def main() -> int:
    rows, tests = [], []
    rng = np.random.default_rng(SEED + 11)
    for run in RUNS:
        per_region = {}
        for fold in FOLDS:
            units = load_units(fold, backbone=run)
            quality = cr.mean_iou(units.confusion, "union")
            per_class = sc.per_class_statistics(units, fold, run)["calibrated"]
            entry = {"quality": quality,
                     "score": sc.aggregate(per_class, units.n_pred, 0.0)}
            parts = source_tiles(fold, run)
            x_all, y_all = lc.stack(parts, "calibrated")
            xt = lc.features(units, "calibrated")
            for kind in ("ridge", "gradient boosting"):
                entry[kind] = lc.fit_predict(kind, x_all, y_all, xt)
            per_region[fold] = entry
            for name in ("score", "ridge", "gradient boosting"):
                rows.append({"run": run, "fold": fold, "ordering": name,
                             "training_tiles": len(y_all),
                             "recall": mt.recall_at(quality, entry[name], BUDGET)})
            print(run, fold, "done", flush=True)
        for kind in ("ridge", "gradient boosting"):
            sub = pd.DataFrame([r for r in rows if r["run"] == run])
            piv = sub.pivot(index="fold", columns="ordering", values="recall")
            diff = piv["score"] - piv[kind]
            lo, hi = paired_bootstrap(per_region, "score", kind, rng)
            tests.append({"run": run, "baseline": kind,
                          "recall_score": piv["score"].mean(),
                          "recall_baseline": piv[kind].mean(),
                          "difference": diff.mean(), "lo": lo, "hi": hi,
                          "regions_score_ahead": int((diff > 0).sum()),
                          "regions_tied": int((diff == 0).sum())})
    pd.DataFrame(rows).to_csv(OUT / "baseline_seeds_by_region.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUT / "baseline_seeds.csv", index=False)
    print(tests.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
