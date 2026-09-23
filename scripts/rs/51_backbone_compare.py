"""Repeat the screening analysis on a second encoder.

The comparison holds everything fixed except the operator that aggregates local
features: the same patches, the same sampling skeleton, the same input features,
the same training schedule, the same tiles, the same scores and the same
criterion. What it tests is whether the relation between the two class
weightings is a property of the data and the task or of the encoder that
produced the confidences.

The surface, the tie bands and the monotonicity are imported from the module the
main experiment uses, so the two encoders are summarised by one implementation.
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
from geoquery_als.rs.store import FOLDS, load_units, suffix  # noqa: E402
from geoquery_als.rs.surface import (ALPHAS, BETAS, BUDGET,  # noqa: E402
                                     bands, lift, monotone_strength, surface)

OUT = ROOT / "outputs" / "rs"
BACKBONES = ("randlanet", "point_transformer")
LABEL = {"randlanet": "RandLA-Net", "point_transformer": "Point Transformer"}
STATISTIC = "calibrated"
DRAWS = 2000
SEED = 20260920


def available(backbone: str) -> bool:
    tag = suffix(backbone)
    return all((ROOT / "outputs" / "pivot" / "unit_statistics"
                / f"target_{fold}{tag}.npz").exists() for fold in FOLDS)


def endpoint_rows(band: pd.DataFrame, grids: dict, label: str) -> dict:
    """Best exponent and its tie band at each of the two conventional criteria."""
    out = {"backbone": label}
    for target, name in ((0.0, "class_averaged"), (1.0, "point_weighted")):
        j = int(np.argmin(np.abs(BETAS - target)))
        row = band.iloc[j]
        out[f"alpha_{name}"] = float(row["alpha_best"])
        out[f"alpha_{name}_lo"] = float(row["alpha_lo"])
        out[f"alpha_{name}_hi"] = float(row["alpha_hi"])
        out[f"lift_{name}"] = float(row["lift_best"])
        i = int(np.argmin(np.abs(ALPHAS - (1.0 - target))))
        out[f"lift_{name}_mismatched"] = float(
            np.mean([grids[f][i, j] for f in FOLDS]))
    return out


def main() -> int:
    ready = [b for b in BACKBONES if available(b)]
    missing = [b for b in BACKBONES if b not in ready]
    if missing:
        print("not yet cached:", missing)
    if len(ready) < 2:
        print("nothing to compare")
        return 1

    rng = np.random.default_rng(SEED)
    rows, summaries = [], []
    for backbone in ready:
        label = LABEL[backbone]
        grids = {}
        for fold in FOLDS:
            units = load_units(fold, backbone=backbone)
            grids[fold] = surface(units, STATISTIC, backbone=backbone)

            reference = cr.mean_iou(units.confusion, "union")
            deficit = mt.deficit_from(reference)
            pooled = cr.mean_iou(units.confusion.sum(axis=0, keepdims=True),
                                 "union")[0]
            q = mt.budget_size(units.n_units, BUDGET) / units.n_units
            class_averaged = sc.score(units, STATISTIC, 0.0, backbone=backbone)
            point_weighted = sc.score(units, STATISTIC, 1.0, backbone=backbone)
            report = mt.recall_report(reference, class_averaged, BUDGET)
            rows.append({
                "backbone": label, "region": fold,
                "n_units": int(units.n_units),
                "pooled_miou": float(pooled),
                "tile_mean_miou": float(reference.mean()),
                "recall_class_averaged": report["recall"],
                "lift_at_alpha0": lift(report["recall"], q),
                "p_exact": report["p_exact"],
                "recall_point_weighted": mt.recall_at(reference, point_weighted,
                                                      BUDGET),
                "naurc_class_averaged": mt.normalised_aurc(deficit,
                                                           class_averaged),
                "naurc_point_weighted": mt.normalised_aurc(deficit,
                                                           point_weighted),
                "spearman": mt.rank_correlation(reference, class_averaged)})

        band = bands(grids)
        observed = monotone_strength(band["alpha_best"].to_numpy())
        boot = np.empty(DRAWS)
        for d in range(DRAWS):
            drawn = rng.choice(FOLDS, size=len(FOLDS), replace=True)
            boot[d] = monotone_strength(
                bands([grids[f] for f in drawn])["alpha_best"].to_numpy())
        summary = {"monotone_rho": observed,
                   "monotone_lo": float(np.percentile(boot, 2.5)),
                   "monotone_hi": float(np.percentile(boot, 97.5))}
        summary.update(endpoint_rows(band, grids, label))
        summaries.append(summary)
        band.assign(backbone=label).to_csv(
            OUT / f"backbone_ridge_{backbone}.csv", index=False)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "backbone_by_region.csv", index=False)
    means = detail.groupby("backbone", as_index=False).mean(numeric_only=True)
    ends = pd.DataFrame(summaries)
    clash = (set(means.columns) & set(ends.columns)) - {"backbone"}
    if clash:
        raise SystemExit(f"column names collide across the two summaries: {clash}")
    summary = means.merge(ends, on="backbone")
    summary.to_csv(OUT / "backbone_summary.csv", index=False)

    print(detail.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
