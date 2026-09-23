"""Quantities quoted in the prose that are summaries of the analysis outputs.

A figure such as the spread of tile quality between two percentiles, or the mean
over regions of a held-out result, belongs to no single row of any table. Writing
them here keeps every number in the manuscript traceable to a file rather than to
an arithmetic step taken while writing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "outputs" / "rs"


def doc_agreement() -> tuple:
    """Rank agreement between the class-conditional DoC and mean confidence.

    The pooled difference of confidences is an affine function of mean target
    confidence and therefore orders tiles identically to it. The class
    conditional form carries a separate constant per class, so the equivalence
    has to be measured rather than asserted.
    """
    from geoquery_als.rs import metrics as mt
    from geoquery_als.rs import scores as sc
    from geoquery_als.rs.store import FOLDS, load_units
    values = []
    for fold in FOLDS:
        units = load_units(fold)
        values.append(mt.rank_correlation(
            sc.score(units, "doc_classwise", 0.0),
            sc.score(units, "calibrated", 0.0)))
    return float(min(values)), float(max(values))


def main() -> int:
    regions = pd.read_csv(DATA / "regions.csv")
    omission = pd.read_csv(DATA / "class_omission.csv")
    operational = pd.read_csv(DATA / "operational.csv")
    per_class = pd.read_csv(DATA / "per_class_screening.csv")
    per_class = per_class[(per_class["minimum_points"] == 500)
                          & (per_class["eligibility"] == "predicted")]

    transfer = pd.read_csv(DATA / "transfer_summary.csv", header=[0, 1],
                           index_col=0)
    transfer.columns = [a if b.startswith("Unnamed") else f"{a}_{b}"
                        for a, b in transfer.columns]
    abundant = ["ground", "vegetation", "building"]
    rare = ["water", "bridge", "permanent_structure"]

    spread = regions["tile_miou_p90"] - regions["tile_miou_p10"]
    derived = {
        "abundant_share_target": float(
            100 * transfer.loc[abundant, "share_target"].sum()),
        "rare_share_target": float(
            100 * transfer.loc[rare, "share_target"].sum()),
        "abundant_loss_max": float(transfer.loc[abundant, "loss"].max()),
        "rare_loss_min": float(transfer.loc[rare, "loss"].min()),
        "rare_loss_max": float(transfer.loc[rare, "loss"].max()),
        "tile_quality_spread_min": float(spread.min()),
        "tile_quality_spread_max": float(spread.max()),
        "pooled_oa_range": float(regions["pooled_oa"].max()
                                 - regions["pooled_oa"].min()),
        "pooled_miou_range": float(regions["pooled_miou"].max()
                                   - regions["pooled_miou"].min()),
        "omission_share_of_tiles_min":
            float(omission["tiles_with_an_unpredicted_reference_class"].min() * 100),
        "omission_share_of_tiles_max":
            float(omission["tiles_with_an_unpredicted_reference_class"].max() * 100),
        "omission_share_of_deficit_min":
            float(omission["share_of_deficit_in_those_tiles"].min() * 100),
        "omission_share_of_deficit_max":
            float(omission["share_of_deficit_in_those_tiles"].max() * 100),
        "omission_ratio_min": float(
            (omission["frequency_in_worst_quintile"]
             / omission["frequency_elsewhere"]).min()),
        "omission_ratio_max": float(
            (omission["frequency_in_worst_quintile"]
             / omission["frequency_elsewhere"]).max()),
        "flagged_patch_share_min": float(operational["patch_share_of_flagged"].min()),
        "flagged_patch_share_max": float(operational["patch_share_of_flagged"].max()),
        "flagged_point_share_min": float(operational["point_share_of_flagged"].min()),
        "flagged_point_share_max": float(operational["point_share_of_flagged"].max()),
        "isolated_share_min": float(operational["join_isolated_share"].min()),
        "isolated_share_max": float(operational["join_isolated_share"].max()),
        "join_z_min": float(operational["join_z"].min()),
        "join_z_max": float(operational["join_z"].max()),
        "morans_i_min": float(operational["morans_i_quality"].min()),
        "morans_i_max": float(operational["morans_i_quality"].max()),
        "composition_distance_min":
            float(operational["composition_distance_median"].min()),
        "composition_distance_max":
            float(operational["composition_distance_median"].max()),
        "per_class_own_recall_min": float(per_class["recall_own_class"].min()),
        "per_class_own_recall_max": float(per_class["recall_own_class"].max()),
        "per_class_own_spearman_min": float(per_class["spearman_own_class"].min()),
        "per_class_balanced_spearman_min":
            float(per_class["spearman_class_balanced"].min()),
        "per_class_balanced_spearman_max":
            float(per_class["spearman_class_balanced"].max()),
        "per_class_comparisons": int(len(per_class)),
        "per_class_own_wins": int((per_class["spearman_own_class"]
                                   > per_class["spearman_class_balanced"]).sum()),
    }

    for statistic in ("calibrated", "confidence", "atc_classwise"):
        honest = pd.read_csv(DATA / f"honest_lift_{statistic}.csv")
        grouped = honest.groupby("criterion")[["alpha_selected", "lift"]].mean()
        for criterion, row in grouped.iterrows():
            derived[f"honest_alpha_{statistic}_{criterion}"] = float(
                row["alpha_selected"])
            derived[f"honest_lift_{statistic}_{criterion}"] = float(row["lift"])

    ladder = pd.read_csv(DATA / "control_ladder.csv").set_index(
        ["criterion", "control"])
    derived["composition_gap_point_weighted"] = float(
        ladder.loc[("point weighted", "matched exponents"), "lift"]
        - ladder.loc[("point weighted", "composition only"), "lift"])

    lo, hi = doc_agreement()
    derived["doc_classwise_agreement_min"] = lo
    derived["doc_classwise_agreement_max"] = hi

    path = DATA / "derived_values.json"
    path.write_text(json.dumps(derived, indent=2), encoding="utf-8")
    for key, value in derived.items():
        print(f"{key:<48} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
