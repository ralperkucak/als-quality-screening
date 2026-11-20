"""Averaging the tile scores of several training runs.

Each region has three models trained from different initializations. Their
tile scores are converted to ranks and averaged, and the averaged ranking is
judged against the tile quality of each of the three runs in turn. Averaging
ranks rather than raw scores keeps one run with a wider score range from
dominating the others. The averaged ranking is compared with the ranking of
each single run judged against the same quality, so the comparison isolates the
effect of averaging the scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import load_units  # noqa: E402
from geoquery_als.rs.surface import BUDGET  # noqa: E402

OUT = ROOT / "outputs" / "rs"
STATISTIC = "calibrated"
VARIANTS = ("randlanet", "randlanet_s2", "randlanet_s3")


def main() -> int:
    rows = []
    for fold in ("D1", "D2", "D3", "D4", "D5"):
        units = {v: load_units(fold, backbone=v) for v in VARIANTS}
        ids = units[VARIANTS[0]].unit_id
        for v in VARIANTS[1:]:
            assert np.array_equal(units[v].unit_id, ids), (fold, v)
        quality = {v: cr.mean_iou(u.confusion, "union") for v, u in units.items()}
        scores = {v: sc.score(u, STATISTIC, 0.0, backbone=v)
                  for v, u in units.items()}
        mean_rank = np.mean([rankdata(s) for s in scores.values()], axis=0)
        for judge in VARIANTS:
            q = quality[judge]
            single = mt.recall_report(q, scores[judge], BUDGET)
            averaged = mt.recall_report(q, mean_rank, BUDGET)
            rows.append({
                "region": fold, "judged_run": judge,
                "single_recall": single["recall"],
                "averaged_recall": averaged["recall"],
                "single_spearman": mt.rank_correlation(q, scores[judge]),
                "averaged_spearman": mt.rank_correlation(q, mean_rank),
                "averaged_p": averaged["p_exact"]})
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "seed_ensemble_by_run.csv", index=False)
    by_region = detail.groupby("region")[
        ["single_recall", "averaged_recall", "single_spearman",
         "averaged_spearman"]].mean()
    by_region.to_csv(OUT / "seed_ensemble_by_region.csv")
    print(detail.round(3).to_string(index=False))
    print(by_region.round(3).to_string())
    print("overall", by_region.mean().round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
