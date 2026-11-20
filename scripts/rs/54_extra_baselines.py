"""Stronger label-free baselines than a mean of confidences.

This script checks whether the class weighting result survives against
orderings that are harder to beat than a pooled average. Three families are
added here. The first reads the lower tail of the confidence distribution
inside a tile rather than its centre, which is the usual way a pooled statistic
is made sensitive to a small bad area. The second takes the weakest predicted
class, which is the most pessimistic class-conditional reduction available. The
third is lightly supervised: a regressor fitted on the labelled source
validation tiles of the same fold, mapping tile level predicted composition and
confidence onto tile quality, and then applied to the held-out region. It uses
no label from the target region, so it competes on the same footing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import FOLDS, bin_centres, load_units  # noqa: E402
from geoquery_als.rs.surface import BUDGET, lift  # noqa: E402

OUT = ROOT / "outputs" / "rs"
STATISTIC = "calibrated"
SEED = 20260920


def hist_quantile(hist: np.ndarray, q: float) -> np.ndarray:
    """Quantile of a binned distribution, one value per unit.

    `hist` is (U, B) of counts on the bin grid of the confidence histogram. The
    result is the centre of the first bin whose cumulative share reaches q, so
    the resolution is one bin and empty units return zero.
    """
    centres = bin_centres()
    total = hist.sum(axis=1, keepdims=True)
    share = np.divide(hist, np.maximum(total, 1e-12))
    cumulative = np.cumsum(share, axis=1)
    idx = np.argmax(cumulative >= q, axis=1)
    out = centres[idx]
    return np.where(total[:, 0] > 0, out, 0.0)


def features(units) -> np.ndarray:
    """Tile descriptors a producer can compute without any reference label."""
    per_class = sc.per_class_statistics(units, units.fold)[STATISTIC]
    n = units.n_pred
    share = n / np.maximum(n.sum(axis=1, keepdims=True), 1e-12)
    n_classes = (n > 0).sum(axis=1, keepdims=True).astype(float)
    size = np.log1p(n.sum(axis=1, keepdims=True))
    pooled = units.conf_hist.sum(axis=1)
    tail = np.stack([hist_quantile(pooled, q) for q in (0.05, 0.10, 0.25)], axis=1)
    return np.hstack([per_class, share, n_classes, size, tail])


def meta_regressor(fold: str, kind: str):
    """Fit on the labelled source validation tiles of this fold."""
    source = load_units(fold, family="source")
    x = features(source)
    y = cr.mean_iou(source.confusion, "union")
    scaler = StandardScaler().fit(x)
    if kind == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                          learning_rate=0.05,
                                          random_state=SEED)
    model.fit(scaler.transform(x), y)
    return model, scaler, source.n_units


def main() -> int:
    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        per_class = sc.per_class_statistics(units, fold)[STATISTIC]
        quality = cr.mean_iou(units.confusion, "union")
        deficit = mt.deficit_from(quality)
        q = mt.budget_size(units.n_units, BUDGET) / units.n_units
        pooled = units.conf_hist.sum(axis=1)

        weakest = np.where(units.n_pred > 0, per_class, np.inf).min(axis=1)
        per_class_tail = np.stack(
            [hist_quantile(units.conf_hist[:, c, :], 0.10)
             for c in range(units.n_pred.shape[1])], axis=1)
        class_averaged_tail = np.where(units.n_pred > 0, per_class_tail,
                                       np.nan)
        class_averaged_tail = np.nanmean(class_averaged_tail, axis=1)

        candidates = {
            "class-averaged calibrated confidence":
                sc.score(units, STATISTIC, 0.0),
            "point-weighted calibrated confidence":
                sc.score(units, STATISTIC, 1.0),
            "weakest predicted class": weakest,
            "fifth percentile of point confidence":
                hist_quantile(pooled, 0.05),
            "tenth percentile of point confidence":
                hist_quantile(pooled, 0.10),
            "quarter percentile of point confidence":
                hist_quantile(pooled, 0.25),
            "class-averaged tenth percentile": class_averaged_tail,
        }
        for kind in ("ridge", "gradient boosting"):
            model, scaler, n_train = meta_regressor(
                fold, "ridge" if kind == "ridge" else "gbr")
            candidates[f"supervised meta-regressor, {kind}"] = model.predict(
                scaler.transform(features(units)))
            n_source = n_train

        for name, s in candidates.items():
            report = mt.recall_report(quality, s, BUDGET)
            rows.append({
                "fold": fold, "method": name,
                "recall": report["recall"],
                "lift": lift(report["recall"], q),
                "p_exact": report["p_exact"],
                "naurc": mt.normalised_aurc(deficit, s),
                "spearman": mt.rank_correlation(quality, s),
                "n_source_tiles": n_source})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "extra_baselines_by_region.csv", index=False)
    summary = detail.groupby("method", as_index=False).mean(numeric_only=True)
    summary["p_exact"] = detail.groupby("method", as_index=False)[
        "p_exact"].max()["p_exact"]
    summary = summary.sort_values("recall", ascending=False)
    summary.to_csv(OUT / "extra_baselines.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
