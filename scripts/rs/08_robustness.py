"""Whether the screening result depends on choices made when the tiles were built.

Three choices are arbitrary in the sense that another analyst would have made
them differently: the side length of a tile, where the lattice origin falls, and
whether the pool of target patches is used alongside the held-out test split.
Each is varied here while the score, the criterion and the budget stay fixed.
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

OUT = ROOT / "outputs" / "rs"
BUDGET = 0.20
SEED = 20260920
DRAWS = 2000
OFFSETS = ((0, 0), (500, 500), (1000, 1000), (500, 1500), (1500, 500))
SCALES = (1000, 2000, 4000)


def evaluate(units, alpha: float = 0.0, statistic: str = "calibrated") -> dict:
    quality = cr.mean_iou(units.confusion, "union")
    score = sc.score(units, statistic, alpha)
    report = mt.recall_report(quality, score, BUDGET)
    return {"tiles": units.n_units, "recall": report["recall"],
            "random": report["random_recall"], "p_exact": report["p_exact"],
            "naurc": mt.normalised_aurc(mt.deficit_from(quality), score),
            "spearman": mt.rank_correlation(quality, score)}


def block_bootstrap(units_by_fold, block_km: float, draws: int = DRAWS) -> dict:
    """Paired difference between the two aggregations, resampling square blocks."""
    rng = np.random.default_rng(SEED)
    prepared = {}
    for f, u in units_by_fold.items():
        quality = cr.mean_iou(u.confusion, "union")
        a = sc.score(u, "calibrated", 0.0)
        b = sc.score(u, "calibrated", 1.0)
        key = (np.floor(u.x / (block_km * 1000)).astype(int) * 100000
               + np.floor(u.y / (block_km * 1000)).astype(int))
        blocks = [np.flatnonzero(key == v) for v in np.unique(key)]
        prepared[f] = (quality, a, b, blocks)

    def difference(sample):
        vals = []
        for f, idx in sample:
            quality, a, b, _ = prepared[f]
            vals.append(mt.recall_at(quality[idx], a[idx], BUDGET)
                        - mt.recall_at(quality[idx], b[idx], BUDGET))
        return float(np.mean(vals))

    observed = difference([(f, np.arange(len(prepared[f][0])))
                           for f in units_by_fold])
    boot = np.empty(draws)
    for d in range(draws):
        sample = []
        for f in rng.choice(list(units_by_fold), size=len(units_by_fold),
                            replace=True):
            blocks = prepared[f][3]
            take = rng.integers(0, len(blocks), len(blocks))
            sample.append((f, np.concatenate([blocks[t] for t in take])))
        boot[d] = difference(sample)
    return {"block_km": block_km, "difference": observed,
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in SCALES:
        for f in FOLDS:
            units = load_units(f, scale_m=scale)
            rows.append({"variation": "tile size", "setting": f"{scale // 1000} km",
                         "region": f, **evaluate(units)})
    for offset in OFFSETS:
        for f in FOLDS:
            units = load_units(f, offset=offset)
            rows.append({"variation": "lattice origin",
                         "setting": f"{offset[0]} m east, {offset[1]} m north",
                         "region": f, **evaluate(units)})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "robustness_detail.csv", index=False)
    summary = (detail.groupby(["variation", "setting"], as_index=False)
               .agg(tiles=("tiles", "sum"), recall=("recall", "mean"),
                    random=("random", "mean"), naurc=("naurc", "mean"),
                    spearman=("spearman", "mean"),
                    regions_above_random=("recall", lambda s: int((s > 0.2).sum())),
                    max_p=("p_exact", "max")))
    summary.to_csv(OUT / "robustness_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    units = {f: load_units(f) for f in FOLDS}
    blocks = pd.DataFrame([block_bootstrap(units, km) for km in (6.0, 10.0, 16.0)])
    blocks.to_csv(OUT / "block_bootstrap.csv", index=False)
    print()
    print(blocks.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
