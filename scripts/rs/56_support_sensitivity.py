"""How much a tile score depends on its smallest predicted classes.

At the class averaged operating point a class predicted at ten points carries
the same weight as a class predicted at ten thousand, so a single spurious
prediction could in principle move a tile a long way down the ranking. The
check is to drop classes below a minimum predicted support before averaging and
see how far the screening moves. A ranking that survives the removal of its
smallest classes is not being driven by them.
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
from geoquery_als.rs.surface import BUDGET, lift  # noqa: E402

OUT = ROOT / "outputs" / "rs"
STATISTIC = "calibrated"
MINIMA = (0, 10, 50, 100, 500, 1000)


def main() -> int:
    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        per_class = sc.per_class_statistics(units, fold)[STATISTIC]
        quality = cr.mean_iou(units.confusion, "union")
        deficit = mt.deficit_from(quality)
        q = mt.budget_size(units.n_units, BUDGET) / units.n_units
        reference = sc.aggregate(per_class, units.n_pred, 0.0)

        for minimum in MINIMA:
            kept = np.where(units.n_pred >= minimum, units.n_pred, 0.0)
            # A tile whose classes are all below the threshold keeps its
            # original class set, since dropping every class leaves no score.
            empty = kept.sum(axis=1) == 0
            kept[empty] = units.n_pred[empty]
            s = sc.aggregate(per_class, kept, 0.0)
            report = mt.recall_report(quality, s, BUDGET)
            rows.append({
                "fold": fold, "minimum_predicted_points": minimum,
                "mean_classes_kept": float((kept > 0).sum(axis=1).mean()),
                "share_tiles_changed": float(
                    ((kept > 0) != (units.n_pred > 0)).any(axis=1).mean()),
                "recall": report["recall"],
                "lift": lift(report["recall"], q),
                "p_exact": report["p_exact"],
                "naurc": mt.normalised_aurc(deficit, s),
                "spearman": mt.rank_correlation(quality, s),
                "rank_agreement_with_primary": mt.rank_correlation(reference, s)})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "support_sensitivity_by_region.csv", index=False)
    summary = detail.groupby("minimum_predicted_points",
                             as_index=False).mean(numeric_only=True)
    summary["p_exact"] = detail.groupby("minimum_predicted_points",
                                        as_index=False)["p_exact"].max()["p_exact"]
    summary.to_csv(OUT / "support_sensitivity.csv", index=False)
    print(summary.round(4).to_string(index=False))
    span = summary["recall"].max() - summary["recall"].min()
    print(f"\nrecall spans {span:.4f} over the thresholds examined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
