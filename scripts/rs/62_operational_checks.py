"""Three checks that bear on how the screening is used.

1. Choosing the score exponent when the criterion exponent is not fixed in
   advance: the regret of each score exponent against the best one, over a
   range of criterion exponents, and the exponent that minimises the largest
   regret.
2. How many held-out patches enter the tile analysis in each region, and why the
   rest do not.
3. Per-class screening with uncertainty: a tile bootstrap interval for the
   difference in rank correlation between a class's own confidence and the
   class-averaged score, in each region, together with the eligible tile count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import BASE, CLASS_NAMES, FOLDS, load_units  # noqa: E402

DATA = ROOT / "outputs" / "rs"
SEED = 20261119
BUDGET = 0.20
DRAWS = 2000


def exponent_regret() -> pd.DataFrame:
    s = pd.read_parquet(DATA / "exponent_surface.parquet")
    rows = []
    for statistic, block in s.groupby("statistic"):
        g = block.groupby(["alpha", "beta"])["lift"].mean().unstack("beta")
        for lo, hi, label in ((0.0, 1.0, "beta in [0, 1]"),
                              (-0.5, 1.5, "beta in [-0.5, 1.5]")):
            cols = [b for b in g.columns if lo - 1e-9 <= b <= hi + 1e-9]
            sub = g[cols]
            regret = sub.max(axis=0) - sub
            worst = regret.max(axis=1)
            a_star = float(worst.idxmin())
            for a in sorted({0.0, 0.5, 1.0, a_star}):
                i = int(np.argmin(np.abs(g.index.to_numpy() - a)))
                row = g.index[i]
                rows.append({"statistic": statistic, "range": label,
                             "alpha": float(row), "minimax": abs(row - a_star) < 1e-9,
                             "worst_regret": float(worst.loc[row]),
                             "lift_beta0": float(g.loc[row, 0.0]),
                             "lift_beta1": float(g.loc[row, 1.0]),
                             "best_beta0": float(g[0.0].max()),
                             "best_beta1": float(g[1.0].max())})
    return pd.DataFrame(rows)


def exclusions() -> pd.DataFrame:
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    rows = []
    for f in FOLDS:
        m = members[(members.fold == f) & (members.family == "target")
                    & (members.scale_m == 2000)]
        cells = m.groupby("unit_id")["patch_id"].size()
        units = load_units(f)
        kept = int(units.n_patches.sum())
        rows.append({"region": f, "held_out_patches": int(m.patch_id.nunique()),
                     "cells": int(len(cells)), "cells_kept": units.n_units,
                     "cells_dropped": int((cells < 8).sum()),
                     "patches_analysed": kept,
                     "patches_excluded": int(m.patch_id.nunique()) - kept,
                     "share_excluded": 1 - kept / m.patch_id.nunique()})
    return pd.DataFrame(rows)


def per_class_intervals() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for f in FOLDS:
        units = load_units(f)
        per_class = sc.per_class_statistics(units)["calibrated"]
        balanced = sc.aggregate(per_class, units.n_pred, 0.0)
        iou = cr.class_iou(units.confusion)
        for k, name in enumerate(CLASS_NAMES):
            keep = (units.n_pred[:, k] >= 500) & np.isfinite(iou[:, k])
            n = int(keep.sum())
            if n < 25:
                continue
            t, own, bal = iou[keep, k], per_class[keep, k], balanced[keep]
            d0 = spearmanr(t, own).statistic - spearmanr(t, bal).statistic
            boot = np.empty(DRAWS)
            for b in range(DRAWS):
                i = rng.integers(0, n, n)
                boot[b] = spearmanr(t[i], own[i]).statistic - spearmanr(t[i], bal[i]).statistic
            rep = mt.recall_report(t, own, BUDGET)
            rows.append({"region": f, "class": name, "eligible_tiles": n,
                         "rho_own": spearmanr(t, own).statistic,
                         "rho_class_averaged": spearmanr(t, bal).statistic,
                         "difference": d0,
                         "lo": np.nanpercentile(boot, 2.5),
                         "hi": np.nanpercentile(boot, 97.5),
                         "recall_own": rep["recall"], "p_exact": rep["p_exact"]})
    return pd.DataFrame(rows)


def main() -> int:
    r = exponent_regret()
    r.to_csv(DATA / "exponent_regret.csv", index=False)
    print(r.round(3).to_string(index=False))
    e = exclusions()
    e.to_csv(DATA / "patch_exclusions.csv", index=False)
    print(e.to_string(index=False))
    p = per_class_intervals()
    p.to_csv(DATA / "per_class_intervals.csv", index=False)
    summary = (p.groupby("class")
               .agg(regions=("region", "nunique"), tiles=("eligible_tiles", "sum"),
                    mean_difference=("difference", "mean"),
                    regions_ahead=("difference", lambda s: int((s > 0).sum())),
                    regions_interval_above_zero=("lo", lambda s: int((s > 0).sum())),
                    regions_p_below_005=("p_exact", lambda s: int((s < 0.05).sum())))
               .reset_index())
    summary.to_csv(DATA / "per_class_intervals_summary.csv", index=False)
    print(p.round(3).to_string(index=False))
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
