"""Sensitivity of the screening to the training initialisation.

One model per fold hides how much of a result is the particular draw of initial
weights and batch order. The protocol is repeated from different
initialisations on the two regions where the screening is weakest and
strongest, so the spread is measured at both ends rather than in the middle.
Everything downstream of training is unchanged, so a difference between seeds
is a difference between trained models and nothing else.

The script reports whatever is cached, so it can be run while later seeds are
still training.
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
from geoquery_als.rs.store import load_units, suffix  # noqa: E402
from geoquery_als.rs.surface import (ALPHAS, BETAS, BUDGET,  # noqa: E402
                                     lift, surface)

OUT = ROOT / "outputs" / "rs"
CACHE = ROOT / "outputs" / "pivot" / "unit_statistics"
STATISTIC = "calibrated"
VARIANTS = ("randlanet", "randlanet_s2", "randlanet_s3")
SEED_LABEL = {"randlanet": 1, "randlanet_s2": 2, "randlanet_s3": 3}


def cached(variant: str) -> list:
    """Folds for which this variant has both families cached."""
    tag = suffix(variant)
    return [f for f in ("D1", "D2", "D3", "D4", "D5")
            if (CACHE / f"target_{f}{tag}.npz").exists()
            and (CACHE / f"source_{f}{tag}.npz").exists()]


def best_exponent(grid: np.ndarray, beta: float) -> float:
    j = int(np.argmin(np.abs(BETAS - beta)))
    return float(ALPHAS[int(np.argmax(grid[:, j]))])


def main() -> int:
    available = {v: cached(v) for v in VARIANTS}
    repeated = sorted(set.intersection(
        *[set(v) for v in available.values() if v]) if any(available.values())
        else set())
    extra = [v for v in VARIANTS if v != "randlanet" and available[v]]
    if not extra:
        print("no seed variants cached yet")
        return 1
    folds = sorted(set(available["randlanet"]).intersection(
        *[set(available[v]) for v in extra]))
    if not folds:
        print("no fold is cached for every variant yet")
        return 1
    print("comparing seeds on", folds, "for", ["randlanet"] + extra)

    rows = []
    for variant in ["randlanet"] + extra:
        for fold in folds:
            units = load_units(fold, backbone=variant)
            quality = cr.mean_iou(units.confusion, "union")
            deficit = mt.deficit_from(quality)
            q = mt.budget_size(units.n_units, BUDGET) / units.n_units
            score = sc.score(units, STATISTIC, 0.0, backbone=variant)
            report = mt.recall_report(quality, score, BUDGET)
            grid = surface(units, STATISTIC, backbone=variant)
            pooled = cr.mean_iou(units.confusion.sum(axis=0, keepdims=True),
                                 "union")[0]
            rows.append({
                "seed": SEED_LABEL[variant], "region": fold,
                "pooled_miou": float(pooled),
                "tile_mean_miou": float(quality.mean()),
                "recall": report["recall"],
                "lift": lift(report["recall"], q),
                "p_exact": report["p_exact"],
                "naurc": mt.normalised_aurc(deficit, score),
                "spearman": mt.rank_correlation(quality, score),
                "alpha_at_beta0": best_exponent(grid, 0.0),
                "alpha_at_beta1": best_exponent(grid, 1.0)})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "seed_by_region.csv", index=False)

    summary = (detail.groupby("region")
               .agg(seeds=("seed", "nunique"),
                    pooled_miou_mean=("pooled_miou", "mean"),
                    pooled_miou_sd=("pooled_miou", "std"),
                    recall_mean=("recall", "mean"),
                    recall_sd=("recall", "std"),
                    recall_min=("recall", "min"),
                    recall_max=("recall", "max"),
                    naurc_mean=("naurc", "mean"),
                    naurc_sd=("naurc", "std"),
                    alpha0_min=("alpha_at_beta0", "min"),
                    alpha0_max=("alpha_at_beta0", "max"),
                    alpha1_min=("alpha_at_beta1", "min"),
                    alpha1_max=("alpha_at_beta1", "max"),
                    max_p=("p_exact", "max"))
               .reset_index())
    summary["recall_range"] = summary["recall_max"] - summary["recall_min"]
    summary["pooled_miou_range"] = (
        detail.groupby("region")["pooled_miou"].max().values
        - detail.groupby("region")["pooled_miou"].min().values)
    summary.to_csv(OUT / "seed_summary.csv", index=False)

    print(detail.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
