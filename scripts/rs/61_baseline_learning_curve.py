"""Supervised baselines trained on a larger set of source tiles, and paired tests.

The source validation patches of a fold give only about seventy source tiles.
The same patches are therefore regrouped sixteen more times with different
seeds and group sizes, which gives about a thousand differently composed
training tiles. Regressors fitted on growing subsets of them give a
learning curve for the supervised baselines. Features are read from raw
confidence as well as calibrated confidence, since the calibration map was
fitted on the same source validation patches.

The proposed score is then compared with the strongest baselines by a paired,
hierarchical bootstrap of the recall difference, resampling regions and then
tiles within each region.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rs"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import (BASE, FOLDS, group_patches,  # noqa: E402
                                   load_patch_stats, load_units)

base = importlib.import_module("54_extra_baselines")

OUT = ROOT / "outputs" / "rs"
BUDGET = 0.20
SEED = 20261119
DRAWS = 4000
SIZES = (25, 50, 100, 200, 400, 800)
REPEATS = 20


def features(units, statistic: str) -> np.ndarray:
    base.STATISTIC = statistic
    return base.features(units)


from geoquery_als.config import PATHS  # noqa: E402

PATCH_INDEX = PATHS.patch_index
GROUPINGS = 16


def source_tiles(fold: str):
    """The original source tiles plus repeated regroupings of the same patches.

    Source tiles are spatially compact k-means groups of source validation
    patches inside each source region. Each regrouping draws a new seed and a
    new target group size between 9 and 25 patches, so the same 1200 patches
    yield many differently composed training tiles.
    """
    from sklearn.cluster import KMeans
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    members = members[(members["fold"] == fold) & (members["family"] == "source")]
    index = pd.read_parquet(PATCH_INDEX, columns=["patch_id", "domain"])
    members = members.merge(index, on="patch_id", how="left")
    stats = load_patch_stats(fold, "source")
    parts = [load_units(fold, family="source")]
    rng = np.random.default_rng(SEED + 3)
    for g in range(GROUPINGS):
        size = int(rng.integers(9, 26))
        rows = []
        for domain, block in members.groupby("domain", sort=True):
            xy = block[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
            k = max(1, int(round(len(block) / size)))
            lab = KMeans(n_clusters=k, n_init=4, random_state=int(rng.integers(1e9))
                         ).fit_predict(xy)
            rows.append(pd.DataFrame({"patch_id": block["patch_id"].to_numpy(),
                                      "unit_id": [f"{fold}_g{g}_{domain}_{v}" for v in lab]}))
        grouped = group_patches(stats, pd.concat(rows), fold, min_patches=8)
        parts.append(grouped)
    return parts


def stack(parts, statistic):
    x = np.vstack([features(p, statistic) for p in parts])
    y = np.concatenate([cr.mean_iou(p.confusion, "union") for p in parts])
    return x, y


def fit_predict(kind, x, y, xt):
    scaler = StandardScaler().fit(x)
    if kind == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=SEED)
    model.fit(scaler.transform(x), y)
    return model.predict(scaler.transform(xt))


def learning_curve() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        quality = cr.mean_iou(units.confusion, "union")
        parts = source_tiles(fold)
        for statistic in ("calibrated", "confidence"):
            x_all, y_all = stack(parts, statistic)
            # one lattice only, as in the original baseline
            x0, y0 = features(parts[0], statistic), cr.mean_iou(parts[0].confusion, "union")
            xt = features(units, statistic)
            for kind in ("ridge", "gradient boosting"):
                s = fit_predict(kind, x0, y0, xt)
                rows.append({"fold": fold, "statistic": statistic, "model": kind,
                             "training_tiles": len(y0), "design": "original grouping",
                             "repeat": 0, "recall": mt.recall_at(quality, s, BUDGET),
                             "spearman": mt.rank_correlation(quality, s)})
                for n in SIZES + (len(y_all),):
                    if n > len(y_all):
                        continue
                    reps = 1 if n == len(y_all) else REPEATS
                    for r in range(reps):
                        take = rng.choice(len(y_all), size=n, replace=False)
                        s = fit_predict(kind, x_all[take], y_all[take], xt)
                        rows.append({"fold": fold, "statistic": statistic,
                                     "model": kind, "training_tiles": n,
                                     "design": "regrouped", "repeat": r,
                                     "recall": mt.recall_at(quality, s, BUDGET),
                                     "spearman": mt.rank_correlation(quality, s)})
        print(fold, "done", flush=True)
    return pd.DataFrame(rows)


def candidate_scores(fold: str) -> dict:
    units = load_units(fold)
    parts = source_tiles(fold)
    x_all, y_all = stack(parts, "calibrated")
    xt = features(units, "calibrated")
    per_class_tail = np.stack(
        [base.hist_quantile(units.conf_hist[:, c, :], 0.10)
         for c in range(units.n_pred.shape[1])], axis=1)
    tail = np.nanmean(np.where(units.n_pred > 0, per_class_tail, np.nan), axis=1)
    x0, y0 = features(parts[0], "calibrated"), cr.mean_iou(parts[0].confusion, "union")
    return units, {
        "class-averaged calibrated confidence": sc.score(units, "calibrated", 0.0),
        "point-weighted calibrated confidence": sc.score(units, "calibrated", 1.0),
        "class-averaged tenth percentile": tail,
        "ridge, original grouping": fit_predict("ridge", x0, y0, xt),
        "ridge, regrouped": fit_predict("ridge", x_all, y_all, xt),
        "gradient boosting, regrouped":
            fit_predict("gradient boosting", x_all, y_all, xt),
    }


def paired_tests() -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 7)
    data = {}
    for fold in FOLDS:
        units, s = candidate_scores(fold)
        data[fold] = (cr.mean_iou(units.confusion, "union"), s, units.n_units)
    ref = "class-averaged calibrated confidence"
    others = [k for k in data[FOLDS[0]][1] if k != ref]

    def diff(name, draw):
        vals = []
        for f, idx in draw:
            q, s, _ = data[f]
            vals.append(mt.recall_at(q[idx], s[ref][idx], BUDGET)
                        - mt.recall_at(q[idx], s[name][idx], BUDGET))
        return float(np.mean(vals))

    draws = []
    for _ in range(DRAWS):
        regions = rng.choice(FOLDS, size=len(FOLDS), replace=True)
        draws.append([(f, rng.integers(0, data[f][2], data[f][2])) for f in regions])
    rows = []
    for name in others:
        full = [(f, np.arange(data[f][2])) for f in FOLDS]
        boot = np.array([diff(name, d) for d in draws])
        per = [diff(name, [(f, np.arange(data[f][2]))]) for f in FOLDS]
        rows.append({"contrast": f"{ref} minus {name}",
                     "recall_reference": np.mean([mt.recall_at(data[f][0], data[f][1][ref], BUDGET) for f in FOLDS]),
                     "recall_other": np.mean([mt.recall_at(data[f][0], data[f][1][name], BUDGET) for f in FOLDS]),
                     "difference": diff(name, full),
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                     "share_at_or_below_zero": float((boot <= 0).mean()),
                     "regions_positive": int(sum(v > 0 for v in per)),
                     "per_region": " ".join(f"{v:+.3f}" for v in per)})
    return pd.DataFrame(rows)


def main() -> int:
    curve = learning_curve()
    curve.to_csv(OUT / "baseline_learning_curve_draws.csv", index=False)
    summary = (curve.groupby(["statistic", "model", "design", "training_tiles", "fold"])
               ["recall"].mean().reset_index()
               .groupby(["statistic", "model", "design", "training_tiles"])["recall"]
               .agg(["mean", "min", "max"]).reset_index())
    summary.to_csv(OUT / "baseline_learning_curve.csv", index=False)
    print(summary.round(4).to_string(index=False))
    tests = paired_tests()
    tests.to_csv(OUT / "baseline_paired_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
