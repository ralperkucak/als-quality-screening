"""Sensitivity of the screening to how much of a tile is observed.

Two kinds of thinning are applied without retraining or re-running inference.

Patch level: a tile is represented by the FRACTAL patches that fall inside its
cell. The score is recomputed from a random subset of those patches, and the
criterion is taken either from the full tile or from the disjoint remainder of
its patches. The split-half form asks whether a score read on one sample of the
cell predicts the quality of another sample of the same cell, which is the
question behind generalising from sampled patches to the cell.

Point level: within each tile the points predicted as each class are thinned at
random, by binomial dropout of each confidence-histogram bin or by a Poisson
bootstrap, and the class-averaged and point-weighted scores are recomputed while
the criterion stays fixed. Classes with few predicted points are the ones whose
class means move most under thinning, so this measures how much the ordering
depends on the exact point counts of low-support classes.
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
from geoquery_als.rs.store import (BASE, FOLDS, NUM_CLASSES, bin_centres,  # noqa: E402
                                   load_calibration, load_patch_stats,
                                   load_units)

OUT = ROOT / "outputs" / "rs"
BUDGET = 0.20
SEED = 20261119
DRAWS = 100


def calibration_map(fold: str):
    f = load_calibration(fold)
    kx, ky = f["cal_knots_x"], f["cal_knots_y"]
    gx, gy = f["cal_global_x"], f["cal_global_y"]
    w = f["cal_shrinkage"]
    centres = bin_centres()
    table = np.empty((NUM_CLASSES, len(centres)))
    base = np.interp(centres, gx, gy)
    for c in range(NUM_CLASSES):
        table[c] = np.clip(w[c] * np.interp(centres, kx[c], ky[c])
                           + (1 - w[c]) * base, 0, 1) if w[c] > 0 else base
    return table


def hist_score(hist: np.ndarray, table: np.ndarray, alpha: float) -> np.ndarray:
    """Calibrated-confidence score of each unit read from its class histograms."""
    n = hist.sum(axis=2)
    total = np.einsum("ucb,cb->uc", hist, table)
    per_class = np.divide(total, n, out=np.zeros_like(total), where=n > 0)
    return sc.aggregate(per_class, n, alpha)


def flagged(score: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(score, kind="stable")[:k]


# ------------------------------------------------------------- point thinning

def point_thinning() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    settings = [("dropout 10%", 0.90), ("dropout 25%", 0.75),
                ("dropout 50%", 0.50), ("bootstrap", None)]
    for fold in FOLDS:
        units = load_units(fold)
        table = calibration_map(fold)
        quality = cr.mean_iou(units.confusion, "union")
        k = mt.budget_size(units.n_units, BUDGET)
        hist = np.rint(units.conf_hist)
        for alpha, name in ((0.0, "class-averaged"), (1.0, "point-weighted")):
            base = hist_score(hist, table, alpha)
            exact = sc.score(units, "calibrated", alpha)
            base_set = set(flagged(base, k))
            rows.append({"fold": fold, "aggregation": name, "setting": "none",
                         "draw": -1,
                         "recall": mt.recall_at(quality, base, BUDGET),
                         "recall_exact_sums": mt.recall_at(quality, exact, BUDGET),
                         "rho_to_full": 1.0, "overlap": 1.0,
                         "lost_classes": 0.0})
            present = hist.sum(axis=2) > 0
            for label, keep in settings:
                for d in range(DRAWS):
                    if keep is None:
                        thin = rng.poisson(hist).astype(float)
                    else:
                        thin = rng.binomial(hist.astype(np.int64), keep).astype(float)
                    s = hist_score(thin, table, alpha)
                    lost = (present & (thin.sum(axis=2) == 0)).sum() / present.sum()
                    rows.append({
                        "fold": fold, "aggregation": name, "setting": label,
                        "draw": d, "recall": mt.recall_at(quality, s, BUDGET),
                        "recall_exact_sums": np.nan,
                        "rho_to_full": spearmanr(s, base).statistic,
                        "overlap": len(base_set & set(flagged(s, k))) / k,
                        "lost_classes": lost})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- patch subsets

def patch_tables(fold: str):
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    members = members[(members["fold"] == fold) & (members["family"] == "target")
                      & (members["scale_m"] == 2000)]
    units = load_units(fold)
    stats = load_patch_stats(fold, "target")
    idx = stats.index
    members = members[members["unit_id"].isin(set(units.unit_id))
                      & members["patch_id"].isin(idx)].copy()
    members["row"] = members["patch_id"].map(idx).astype(np.int64)
    slot = {u: i for i, u in enumerate(units.unit_id)}
    members["slot"] = members["unit_id"].map(slot)
    groups = [g["row"].to_numpy() for _, g in members.sort_values("slot")
              .groupby("slot", sort=True)]
    xy = [g[["centroid_x", "centroid_y"]].to_numpy() for _, g in
          members.sort_values("slot").groupby("slot", sort=True)]
    return units, stats, groups, xy


def summed(stats, rows_per_unit):
    conf = np.stack([stats.confusion[r].sum(axis=0) for r in rows_per_unit])
    hist = np.stack([stats.conf_hist[r].sum(axis=0) for r in rows_per_unit])
    return conf.astype(float), hist


def patch_subsets() -> tuple:
    rng = np.random.default_rng(SEED + 1)
    rows, strata = [], []
    for fold in FOLDS:
        units, stats, groups, xy = patch_tables(fold)
        table = calibration_map(fold)
        full_q = cr.mean_iou(units.confusion, "union")
        k = mt.budget_size(units.n_units, BUDGET)
        for alpha, name in ((0.0, "class-averaged"), (1.0, "point-weighted")):
            for m in (2, 4, 8, 12):
                for d in range(DRAWS // 2):
                    pick = [rng.choice(g, size=min(m, len(g)), replace=False)
                            for g in groups]
                    _, hist = summed(stats, pick)
                    s = hist_score(hist, table, alpha)
                    rows.append({"fold": fold, "aggregation": name,
                                 "design": f"{m} patches scored, full tile judged",
                                 "patches": m, "draw": d,
                                 "recall": mt.recall_at(full_q, s, BUDGET)})
            for d in range(DRAWS // 2):
                a_rows, b_rows = [], []
                for g in groups:
                    perm = rng.permutation(g)
                    half = len(g) // 2
                    a_rows.append(perm[:half])
                    b_rows.append(perm[half:])
                conf_a, hist_a = summed(stats, a_rows)
                conf_b, _ = summed(stats, b_rows)
                s = hist_score(hist_a, table, alpha)
                q_a = cr.mean_iou(conf_a, "union")
                q_b = cr.mean_iou(conf_b, "union")
                rows.append({"fold": fold, "aggregation": name,
                             "design": "half scored, same half judged",
                             "patches": -1, "draw": d,
                             "recall": mt.recall_at(q_a, s, BUDGET)})
                rows.append({"fold": fold, "aggregation": name,
                             "design": "half scored, other half judged",
                             "patches": -1, "draw": d,
                             "recall": mt.recall_at(q_b, s, BUDGET)})
                rows.append({"fold": fold, "aggregation": name,
                             "design": "quality agreement of the two halves",
                             "patches": -1, "draw": d,
                             "recall": mt.recall_at(q_b, q_a, BUDGET)})

        # how the full-tile result is distributed over tiles of different support
        score = sc.score(units, "calibrated", 0.0)
        worst = mt.worst_set(full_q, k)
        picked = set(flagged(score, k))
        n_patches = units.n_patches
        spread = np.array([np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1)).mean()
                           for p in xy]) / 1000.0
        for var, values in (("patches per tile", n_patches),
                            ("patch spread (km)", spread)):
            cuts = np.quantile(values, [1 / 3, 2 / 3])
            band = np.digitize(values, cuts)
            for b in range(3):
                in_band = np.flatnonzero(band == b)
                w = [i for i in worst if band[i] == b]
                strata.append({
                    "fold": fold, "variable": var, "band": ["low", "middle", "high"][b],
                    "range": f"{values[in_band].min():.3g}-{values[in_band].max():.3g}",
                    "tiles": len(in_band), "worst_tiles": len(w),
                    "recovered": sum(i in picked for i in w),
                    "flag_rate": np.mean([i in picked for i in in_band]),
                    "worst_rate": len(w) / max(len(in_band), 1)})
    return pd.DataFrame(rows), pd.DataFrame(strata)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    thin = point_thinning()
    thin.to_csv(OUT / "point_thinning_draws.csv", index=False)
    t = (thin[thin["draw"] >= 0].groupby(["aggregation", "setting", "fold"])
         .agg(recall=("recall", "mean"), rho=("rho_to_full", "mean"),
              overlap=("overlap", "mean"), lost=("lost_classes", "mean"),
              recall_sd=("recall", "std")).reset_index())
    summary = (t.groupby(["aggregation", "setting"])
               .agg(recall=("recall", "mean"), recall_lo=("recall", "min"),
                    recall_hi=("recall", "max"), recall_sd_within=("recall_sd", "mean"),
                    rho=("rho", "mean"), overlap=("overlap", "mean"),
                    lost_classes=("lost", "mean")).reset_index())
    base = thin[thin["draw"] < 0].groupby("aggregation")[
        ["recall", "recall_exact_sums"]].mean()
    summary.to_csv(OUT / "point_thinning_summary.csv", index=False)
    t.to_csv(OUT / "point_thinning_by_region.csv", index=False)
    print(base.round(4))
    print(summary.round(4).to_string(index=False))

    subsets, strata = patch_subsets()
    subsets.to_csv(OUT / "patch_subsets_draws.csv", index=False)
    s = (subsets.groupby(["aggregation", "design", "patches", "fold"])["recall"]
         .mean().reset_index())
    ssum = (s.groupby(["aggregation", "design", "patches"])["recall"]
            .agg(["mean", "min", "max"]).reset_index())
    ssum.to_csv(OUT / "patch_subsets_summary.csv", index=False)
    s.to_csv(OUT / "patch_subsets_by_region.csv", index=False)
    print(ssum.round(4).to_string(index=False))
    strata.to_csv(OUT / "support_strata_by_region.csv", index=False)
    st = (strata.groupby(["variable", "band"])
          .agg(tiles=("tiles", "sum"), worst=("worst_tiles", "sum"),
               recovered=("recovered", "sum")).reset_index())
    st["recall"] = st["recovered"] / st["worst"]
    st.to_csv(OUT / "support_strata.csv", index=False)
    print(st.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
