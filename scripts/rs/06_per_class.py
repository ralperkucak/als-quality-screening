"""Screening for a single class.

A production query is usually narrower than the quality of a tile as a whole.
An operator revising a building layer wants the tiles where the building class
went wrong, and the matching argument predicts that the statistic to read is
that class's own confidence rather than any summary over classes. The comparison
below holds the tiles, the budget and the reference metric fixed and varies only
which confidence the ordering reads.
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
from geoquery_als.rs.store import CLASS_NAMES, FOLDS, load_units  # noqa: E402

OUT = ROOT / "outputs" / "rs"
BUDGET = 0.20
MIN_POINTS = (200, 500, 1000)
PRIMARY_MINIMUM = 500
STATISTIC = "calibrated"
# Which tiles the question is asked of. A producer knows how many points the
# model assigned to a class and can select on that; the number of reference
# points is known only after the check that the screening is meant to target,
# so that rule is reported as a diagnostic and never as the operational result.
ELIGIBILITY = ("predicted", "reference")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in FOLDS:
        units = load_units(f)
        per_class = sc.per_class_statistics(units)[STATISTIC]
        class_balanced = sc.aggregate(per_class, units.n_pred, 0.0)
        point_weighted = sc.aggregate(per_class, units.n_pred, 1.0)
        iou = cr.class_iou(units.confusion)
        support = {"reference": units.n_ref, "predicted": units.n_pred}

        for k, name in enumerate(CLASS_NAMES):
          for rule in ELIGIBILITY:
            for minimum in MIN_POINTS:
                keep = (support[rule][:, k] >= minimum) & np.isfinite(iou[:, k])
                if keep.sum() < 25:
                    continue
                target = iou[keep, k]
                own = per_class[keep, k]
                report = mt.recall_report(target, own, BUDGET)
                rows.append({
                    "region": f, "class": name, "eligibility": rule,
                    "minimum_points": minimum,
                    "eligible_tiles": int(keep.sum()),
                    "mean_iou": float(target.mean()),
                    "spearman_own_class": mt.rank_correlation(target, own),
                    "spearman_class_balanced": mt.rank_correlation(
                        target, class_balanced[keep]),
                    "spearman_point_weighted": mt.rank_correlation(
                        target, point_weighted[keep]),
                    "recall_own_class": report["recall"],
                    "p_exact": report["p_exact"],
                    "recall_class_balanced": mt.recall_at(
                        target, class_balanced[keep], BUDGET),
                    "recall_point_weighted": mt.recall_at(
                        target, point_weighted[keep], BUDGET),
                    "random": report["random_recall"],
                })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "per_class_screening.csv", index=False)

    primary = detail[(detail["minimum_points"] == PRIMARY_MINIMUM)
                     & (detail["eligibility"] == "predicted")]
    summary = (primary.groupby("class", as_index=False)
               .agg(regions=("region", "nunique"),
                    eligible_tiles=("eligible_tiles", "sum"),
                    mean_iou=("mean_iou", "mean"),
                    spearman_own_class=("spearman_own_class", "mean"),
                    spearman_class_balanced=("spearman_class_balanced", "mean"),
                    recall_own_class=("recall_own_class", "mean"),
                    recall_class_balanced=("recall_class_balanced", "mean"),
                    max_p=("p_exact", "max")))
    order = {n: i for i, n in enumerate(CLASS_NAMES)}
    summary = summary.sort_values("class", key=lambda s: s.map(order))
    summary.to_csv(OUT / "per_class_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    wins = int((primary["spearman_own_class"]
                > primary["spearman_class_balanced"]).sum())
    print(f"\nown-class ordering ahead of the class-averaged ordering in "
          f"{wins} of {len(primary)} class and region comparisons")
    for rule in ELIGIBILITY:
        for minimum in MIN_POINTS:
            sub = detail[(detail["minimum_points"] == minimum)
                         & (detail["eligibility"] == rule)]
            w = int((sub["spearman_own_class"]
                     > sub["spearman_class_balanced"]).sum())
            print(f"  {rule} support at least {minimum}: {w} of {len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
