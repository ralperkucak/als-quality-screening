"""What the review queue costs, where it lies, and what it cannot see.

Three questions an operator would ask before adopting a screening order. Whether
the tiles it flags are cheap or expensive to open. Whether they sit together on
the ground or are scattered across the block. And what kind of failure the
statistics are structurally unable to report.
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
PERMUTATIONS = 2000
SCALE_M = 2000


def flagged(score: np.ndarray, q: float = BUDGET) -> np.ndarray:
    k = mt.budget_size(len(score), q)
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(score, kind="stable")[:k]] = True
    return mask


def join_count(mask: np.ndarray, x: np.ndarray, y: np.ndarray,
               scale: int = SCALE_M) -> dict:
    """Adjacency between flagged tiles on the regular lattice, against a permutation null."""
    key = {(int(round(a / scale)), int(round(b / scale))): i
           for i, (a, b) in enumerate(zip(x, y))}
    edges = []
    for (cx, cy), i in key.items():
        for dx, dy in ((1, 0), (0, 1)):
            j = key.get((cx + dx, cy + dy))
            if j is not None:
                edges.append((i, j))
    if not edges:
        return {"edges": 0}
    edges = np.asarray(edges)

    def count(m):
        return int((m[edges[:, 0]] & m[edges[:, 1]]).sum())

    observed = count(mask)
    rng = np.random.default_rng(SEED)
    null = np.empty(PERMUTATIONS)
    for r in range(PERMUTATIONS):
        null[r] = count(rng.permutation(mask))
    isolated = float(np.mean([
        not (mask[edges[edges[:, 0] == i, 1]].any()
             or mask[edges[edges[:, 1] == i, 0]].any())
        for i in np.flatnonzero(mask)]))
    return {"edges": len(edges), "adjacent_pairs": observed,
            "null_mean": float(null.mean()),
            "z": float((observed - null.mean()) / max(null.std(), 1e-9)),
            "isolated_share": isolated}


def morans_i(values: np.ndarray, x: np.ndarray, y: np.ndarray,
             scale: int = SCALE_M) -> float:
    key = {(int(round(a / scale)), int(round(b / scale))): i
           for i, (a, b) in enumerate(zip(x, y))}
    pairs = []
    for (cx, cy), i in key.items():
        for dx, dy in ((1, 0), (0, 1)):
            j = key.get((cx + dx, cy + dy))
            if j is not None:
                pairs.append((i, j))
    if not pairs:
        return float("nan")
    pairs = np.asarray(pairs)
    z = values - values.mean()
    num = 2.0 * float((z[pairs[:, 0]] * z[pairs[:, 1]]).sum())
    return float(len(values) * num / (2 * len(pairs) * float((z ** 2).sum())))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, omission = [], []
    for f in FOLDS:
        units = load_units(f)
        quality = cr.mean_iou(units.confusion, "union")
        deficit = mt.deficit_from(quality)
        score = sc.score(units, "calibrated", 0.0)
        mask = flagged(score)

        cost_patches = units.n_patches.astype(float)
        cost_points = units.n_points.astype(float)
        row = {
            "region": f,
            "recall_20": mt.recall_at(quality, score, BUDGET),
            "captured_20": mt.captured_deficit(deficit, score, BUDGET),
            "patch_share_of_flagged": float(cost_patches[mask].sum()
                                            / cost_patches.sum()),
            "point_share_of_flagged": float(cost_points[mask].sum()
                                            / cost_points.sum()),
            "captured_at_20pc_patch_cost": mt.cost_weighted_capture(
                deficit, score, cost_patches, BUDGET),
            "captured_at_20pc_point_cost": mt.cost_weighted_capture(
                deficit, score, cost_points, BUDGET),
            "morans_i_quality": morans_i(quality, units.x, units.y),
            "morans_i_score": morans_i(score, units.x, units.y),
        }
        predicted = units.n_pred / np.maximum(
            units.n_pred.sum(axis=1, keepdims=True), 1e-12)
        reference = units.n_ref / np.maximum(
            units.n_ref.sum(axis=1, keepdims=True), 1e-12)
        variation = 0.5 * np.abs(predicted - reference).sum(axis=1)
        row["composition_distance_median"] = float(np.median(variation))
        row["composition_distance_p90"] = float(np.percentile(variation, 90))
        row.update({f"join_{k}": v for k, v in
                    join_count(mask, units.x, units.y).items()})
        rows.append(row)

        ref_present = units.n_ref > 0
        predicted = units.n_pred > 0
        missed = ref_present & ~predicted
        any_missed = missed.any(axis=1)
        worst = np.zeros(units.n_units, dtype=bool)
        worst[np.argsort(quality, kind="stable")[:mt.budget_size(
            units.n_units, BUDGET)]] = True
        omission.append({
            "region": f,
            "tiles_with_an_unpredicted_reference_class": float(any_missed.mean()),
            "share_of_deficit_in_those_tiles": float(
                deficit[any_missed].sum() / deficit.sum()),
            "frequency_in_worst_quintile": float(any_missed[worst].mean()),
            "frequency_elsewhere": float(any_missed[~worst].mean()),
            "mean_points_missed": float(
                units.n_ref[missed].mean() if missed.any() else 0.0),
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "operational.csv", index=False)
    print(table.round(4).to_string(index=False))
    miss = pd.DataFrame(omission)
    miss.to_csv(OUT / "class_omission.csv", index=False)
    print()
    print(miss.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
