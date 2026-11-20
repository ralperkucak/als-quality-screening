"""Minimum class support applied to the quality criterion.

The score-side check (56_support_sensitivity.py) drops classes with few
predicted points before averaging. The criterion has the same exposure: a class
holding a handful of points on either side enters the union mean intersection
over union with the same weight as ground or vegetation. Here classes whose
union mass in a tile is below a minimum are dropped from the criterion, and the
screening is recomputed with the score left unchanged and with the same minimum
applied to the predicted support of the score.

A second summary counts how many tiles of the worst fifth owe their place to a
class below the minimum: tiles that are in the worst fifth under the full
criterion and leave it once those classes are dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import FOLDS, load_units  # noqa: E402
from geoquery_als.rs.surface import BUDGET  # noqa: E402

OUT = ROOT / "outputs" / "rs"
STATISTIC = "calibrated"
MINIMA = (0, 10, 100, 1000)


def restricted_quality(confusion: np.ndarray, minimum: float) -> np.ndarray:
    """Union mean IoU over the classes whose union mass reaches `minimum`."""
    tp = np.einsum("ucc->uc", confusion)
    union = confusion.sum(axis=2) + confusion.sum(axis=1) - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)
    mask = union >= max(minimum, 1e-9)
    empty = mask.sum(axis=1) == 0
    mask[empty] = union[empty] > 0
    return np.where(mask, iou, 0.0).sum(axis=1) / mask.sum(axis=1)


def predicted_restricted_score(per_class, n_pred, minimum):
    kept = np.where(n_pred >= minimum, n_pred, 0.0)
    empty = kept.sum(axis=1) == 0
    kept[empty] = n_pred[empty]
    return sc.aggregate(per_class, kept, 0.0)


def main() -> int:
    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        per_class = sc.per_class_statistics(units, fold)[STATISTIC]
        score = sc.aggregate(per_class, units.n_pred, 0.0)
        full = cr.mean_iou(units.confusion, "union")
        k = mt.budget_size(units.n_units, BUDGET)
        worst_full = set(mt.worst_set(full, k).tolist())
        for minimum in MINIMA:
            quality = restricted_quality(units.confusion, minimum)
            worst = set(mt.worst_set(quality, k).tolist())
            fixed = mt.recall_report(quality, score, BUDGET)
            matched = mt.recall_report(
                quality, predicted_restricted_score(per_class, units.n_pred, minimum),
                BUDGET)
            rows.append({
                "fold": fold, "minimum_union_points": minimum,
                "recall_score_unchanged": fixed["recall"],
                "p_score_unchanged": fixed["p_exact"],
                "recall_score_matched": matched["recall"],
                "p_score_matched": matched["p_exact"],
                "worst_fifth_retained": len(worst & worst_full) / k,
                "spearman_with_full_criterion": mt.rank_correlation(full, quality)})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "criterion_support_by_region.csv", index=False)
    summary = detail.groupby("minimum_union_points", as_index=False).mean(numeric_only=True)
    for col in ("p_score_unchanged", "p_score_matched"):
        summary[col] = detail.groupby("minimum_union_points")[col].max().to_numpy()
    summary.to_csv(OUT / "criterion_support.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(detail.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
