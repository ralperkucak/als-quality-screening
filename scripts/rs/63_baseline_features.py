"""Learning curve summary and a feature ablation of the supervised baseline.

The regressor is refitted with the full set of ~1200 regrouped source tiles
using either the class-wise features (per-class calibrated confidence and class
shares) or only pooled features (point-weighted mean confidence, pooled
confidence quantiles and tile size). If the regressor's skill comes from
learning a class weighting, removing the class-wise features should remove it.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rs"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import FOLDS, load_units  # noqa: E402

lc = importlib.import_module("61_baseline_learning_curve")
base = importlib.import_module("54_extra_baselines")
OUT = ROOT / "outputs" / "rs"


def pooled_features(units):
    n = units.n_pred
    pooled = units.conf_hist.sum(axis=1)
    tail = np.stack([base.hist_quantile(pooled, q) for q in (0.05, 0.10, 0.25)], axis=1)
    mean = sc.score(units, "calibrated", 1.0)[:, None]
    size = np.log1p(n.sum(axis=1, keepdims=True))
    return np.hstack([mean, tail, size])


def main() -> int:
    curve = pd.read_csv(OUT / "baseline_learning_curve_draws.csv")
    full = curve.groupby("fold")["training_tiles"].transform("max")
    curve["size"] = np.where((curve["design"] == "regrouped") & (curve["training_tiles"] == full),
                             "all", curve["training_tiles"].astype(str))
    curve.loc[curve["design"] == "original grouping", "size"] = "original"
    s = (curve.groupby(["statistic", "model", "size", "fold"])["recall"].mean()
         .reset_index().groupby(["statistic", "model", "size"])["recall"]
         .agg(["mean", "min", "max"]).reset_index())
    s.to_csv(OUT / "baseline_learning_curve.csv", index=False)
    print(s.round(3).to_string(index=False))

    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        q = cr.mean_iou(units.confusion, "union")
        parts = lc.source_tiles(fold)
        y = np.concatenate([cr.mean_iou(p.confusion, "union") for p in parts])
        for name, feat in (("class-wise features", lambda u: lc.features(u, "calibrated")),
                           ("pooled features only", pooled_features)):
            x = np.vstack([feat(p) for p in parts])
            for kind in ("ridge", "gradient boosting"):
                pred = lc.fit_predict(kind, x, y, feat(units))
                rows.append({"fold": fold, "features": name, "model": kind,
                             "tiles": len(y), "recall": mt.recall_at(q, pred, 0.20),
                             "spearman": mt.rank_correlation(q, pred)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "baseline_feature_ablation_by_region.csv", index=False)
    summary = d.groupby(["features", "model"]).agg(
        recall=("recall", "mean"), lo=("recall", "min"), hi=("recall", "max"),
        spearman=("spearman", "mean"), tiles=("tiles", "mean")).reset_index()
    summary.to_csv(OUT / "baseline_feature_ablation.csv", index=False)
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
