"""Controls that separate the weighting rule from everything it could be confused with.

Each control removes one thing the result might be resting on and leaves the rest
of the procedure untouched. Removing the confidence value leaves the class
composition of a tile. Ordering by the number of predicted classes leaves only
the quantity a class-averaged mean reads when the values it averages carry
nothing. Replacing the predicted composition with the reference one removes the
model's view of the scene. The ladder reports what survives each removal at the
matched setting of the two exponents.
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
SHUFFLES = 200
DRAWS = 2000
RARE = (4, 5, 6)


def lift(recall: float, q: float) -> float:
    """Share of the improvement over random review that an ordering realises."""
    return (recall - q) / (1.0 - q)


def base_rate(n: int) -> float:
    """What a review order chosen at random reaches, given integer budgets."""
    return mt.budget_size(n, BUDGET) / n


def evaluate(scores, quality) -> float:
    return float(np.mean([
        lift(mt.recall_at(quality[f], scores[f], BUDGET), base_rate(len(quality[f])))
        for f in FOLDS]))


def interval(scores, quality, rng, draws: int = DRAWS) -> tuple:
    prepared = {f: (quality[f], scores[f]) for f in FOLDS}
    boot = np.empty(draws)
    for d in range(draws):
        vals = []
        for f in rng.choice(FOLDS, size=len(FOLDS), replace=True):
            q, s = prepared[f]
            idx = rng.integers(0, len(q), len(q))
            vals.append(lift(mt.recall_at(q[idx], s[idx], BUDGET),
                             base_rate(len(idx))))
        boot[d] = float(np.mean(vals))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    units = {f: load_units(f) for f in FOLDS}
    stats = {f: sc.per_class_statistics(units[f]) for f in FOLDS}
    quality = {b: {f: cr.quality(units[f].confusion, b) for f in FOLDS}
               for b in (0.0, 1.0)}

    def weighted(alpha, key="calibrated", weights=None):
        out = {}
        for f in FOLDS:
            w = (sc.class_weights(units[f].n_pred, alpha) if weights is None
                 else weights(f, alpha))
            out[f] = (w * stats[f][key]).sum(axis=1)
        return out

    def reference_weights(f, alpha):
        occ = units[f].n_ref.astype(float)
        share = occ / np.maximum(occ.sum(axis=1, keepdims=True), 1e-12)
        w = np.where(occ > 0, np.power(np.maximum(share, 1e-12), alpha), 0.0)
        return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    rows = []
    for beta, alpha, label in ((0.0, 0.0, "class averaged"),
                               (1.0, 1.0, "point weighted")):
        target = quality[beta]

        primary = weighted(alpha)
        lo, hi = interval(primary, target, rng)
        rows.append({"criterion": label, "control": "matched exponents",
                     "lift": evaluate(primary, target), "lo": lo, "hi": hi})

        mismatched = weighted(1.0 - alpha)
        lo, hi = interval(mismatched, target, rng)
        rows.append({"criterion": label, "control": "mismatched exponents",
                     "lift": evaluate(mismatched, target), "lo": lo, "hi": hi})

        blind = weighted(alpha, key="source_accuracy_prior")
        lo, hi = interval(blind, target, rng)
        rows.append({"criterion": label, "control": "composition only",
                     "lift": evaluate(blind, target), "lo": lo, "hi": hi})

        counted = np.empty(SHUFFLES)
        for s in range(SHUFFLES):
            draw = {f: (units[f].n_pred > 0).sum(axis=1).astype(float)
                       + rng.uniform(0, 1e-6, units[f].n_units) for f in FOLDS}
            counted[s] = evaluate(draw, target)
        rows.append({"criterion": label, "control": "class count only",
                     "lift": float(counted.mean()),
                     "lo": float(np.percentile(counted, 2.5)),
                     "hi": float(np.percentile(counted, 97.5))})

        oracle = weighted(alpha, weights=reference_weights)
        lo, hi = interval(oracle, target, rng)
        rows.append({"criterion": label, "control": "reference class weights",
                     "lift": evaluate(oracle, target), "lo": lo, "hi": hi})

    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / "control_ladder.csv", index=False)
    print(ladder.round(4).to_string(index=False))

    alphas = np.round(np.arange(-1.0, 1.01, 0.125), 3)
    screening = []
    for alpha in alphas:
        vals = []
        for f in FOLDS:
            target = cr.rare_class_quality(units[f].confusion, RARE)
            keep = np.isfinite(target)
            s = (sc.class_weights(units[f].n_pred, alpha)
                 * stats[f]["calibrated"]).sum(axis=1)[keep]
            vals.append(mt.recall_at(target[keep], s, BUDGET))
        screening.append({"alpha": alpha, "mean_recall": float(np.mean(vals)),
                          **{f: v for f, v in zip(FOLDS, vals)}})
    rare = pd.DataFrame(screening)
    rare.to_csv(OUT / "controls_rare_class.csv", index=False)
    print()
    print(rare.round(3).to_string(index=False))
    print("\nbest score exponent for the rare class criterion: %g"
          % rare.loc[rare["mean_recall"].idxmax(), "alpha"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
