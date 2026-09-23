"""Screening performance over the score exponent and the criterion exponent.

The experiment varies two things and nothing else: how strongly the tile score
weights a predicted class by the number of points it holds, and how strongly the
quality criterion weights a class by the ground it occupies. Both are written as
a normalised power of a point share, so the point weighted convention and the
class averaged convention become two operating points on one axis.

The quantity reported is the set of score exponents that come within one
standard error of the best screening for a given criterion, rather than the
single grid point that happens to be highest. The surface is flat near its
maximum at the point weighted end, and a band states that where an argmax would
hide it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.store import FOLDS, load_units  # noqa: E402
from geoquery_als.rs.surface import (ALPHAS, BETAS, BUDGET,  # noqa: E402
                                     ENDPOINTS, bands, monotone_strength,
                                     surface)

OUT = ROOT / "outputs" / "rs"
DRAWS = 2000
SEED = 20260920
STATISTICS = ("calibrated", "confidence", "atc_classwise")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    units = {f: load_units(f) for f in FOLDS}

    records, ridge_rows, summaries = [], [], []
    for statistic in STATISTICS:
        grids = {f: surface(units[f], statistic) for f in FOLDS}
        for f in FOLDS:
            for i, a in enumerate(ALPHAS):
                for j, b in enumerate(BETAS):
                    records.append({"statistic": statistic, "fold": f,
                                    "alpha": a, "beta": b,
                                    "lift": grids[f][i, j]})

        band = bands(grids).assign(statistic=statistic)
        for f in FOLDS:
            column = grids[f]
            band[f] = [ALPHAS[int(np.argmax(column[:, j]))]
                       for j in range(len(BETAS))]
        ridge_rows.append(band)

        observed = monotone_strength(band["alpha_best"].to_numpy())
        boot = np.empty(DRAWS)
        for d in range(DRAWS):
            drawn = rng.choice(FOLDS, size=len(FOLDS), replace=True)
            drawn_band = bands([grids[f] for f in drawn])
            boot[d] = monotone_strength(drawn_band["alpha_best"].to_numpy())

        summary = {"statistic": statistic,
                   "monotone_rho": observed,
                   "monotone_lo": float(np.percentile(boot, 2.5)),
                   "monotone_hi": float(np.percentile(boot, 97.5))}
        for target, name in ((0.0, "class_averaged"), (1.0, "point_weighted")):
            j = int(np.argmin(np.abs(BETAS - target)))
            row = band.iloc[j]
            summary[f"alpha_{name}"] = row["alpha_best"]
            summary[f"alpha_{name}_lo"] = row["alpha_lo"]
            summary[f"alpha_{name}_hi"] = row["alpha_hi"]
            summary[f"lift_{name}"] = row["lift_best"]
            opposite = 1.0 - target
            i = int(np.argmin(np.abs(ALPHAS - opposite)))
            summary[f"lift_{name}_mismatched"] = float(
                np.mean([grids[f][i, j] for f in FOLDS]))
        summaries.append(summary)

        honest = []
        for held in FOLDS:
            others = [f for f in FOLDS if f != held]
            chosen = bands([grids[f] for f in others])
            for target, name in ((0.0, "class_averaged"), (1.0, "point_weighted")):
                j = int(np.argmin(np.abs(BETAS - target)))
                a = float(chosen.iloc[j]["alpha_best"])
                i = int(np.argmin(np.abs(ALPHAS - a)))
                honest.append({"statistic": statistic, "held_out": held,
                               "criterion": name, "alpha_selected": ALPHAS[i],
                               "lift": grids[held][i, j]})
        pd.DataFrame(honest).to_csv(
            OUT / f"honest_lift_{statistic}.csv", index=False)

    pd.DataFrame(records).to_parquet(OUT / "exponent_surface.parquet", index=False)
    pd.concat(ridge_rows).to_csv(OUT / "exponent_ridge.csv", index=False)
    frame = pd.DataFrame(summaries)
    frame.to_csv(OUT / "exponent_summary.csv", index=False)
    print(frame.round(4).to_string(index=False))
    for statistic in STATISTICS:
        h = pd.read_csv(OUT / f"honest_lift_{statistic}.csv")
        print(f"\nheld-out lift, {statistic}")
        print(h.groupby("criterion")[["alpha_selected", "lift"]].mean()
              .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
