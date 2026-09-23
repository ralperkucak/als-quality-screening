"""What recalibrating the confidence contributes to the ordering.

Every score in this study reads a confidence value, and confidence from a deep
network is known to be optimistic. The question here is whether correcting that
optimism on labelled source data changes which units of an unseen region are
ranked lowest. Six recalibration maps are fitted on the source validation points
of each fold, scored out of sample so that the fit cannot flatter itself, and
then carried to the target where the ordering they produce is compared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import (FOLDS, NUM_CLASSES, bin_centres,  # noqa: E402
                                   load_source_points, load_units)

OUT = ROOT / "outputs" / "rs"
# Points per patch in the calibration subsample, set by
# scripts/pivot/03_fit_source_calibration.py; the sample is stored patch by patch.
POINTS_PER_PATCH = 200
SEED = 20260920
N_ECE_BINS = 20
SHRINK_AT = 1000.0
CLIP = 1e-6


def _clip(p):
    return np.clip(p, CLIP, 1.0 - CLIP)


def fit_identity(conf, correct, weight):
    return lambda x: x


def fit_isotonic(conf, correct, weight):
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(conf, correct, sample_weight=weight)
    return lambda x: np.clip(model.predict(np.clip(x, 0.0, 1.0)), 0.0, 1.0)


def fit_platt(conf, correct, weight):
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit(_clip(conf)).reshape(-1, 1), correct, sample_weight=weight)
    return lambda x: model.predict_proba(
        logit(_clip(np.asarray(x, dtype=float))).reshape(-1, 1))[:, 1]


def fit_temperature(conf, correct, weight):
    """One parameter rescaling of the confidence in the logit domain."""
    z = logit(_clip(conf))

    def loss(t):
        p = _clip(expit(z / max(t, 1e-3)))
        return -float(np.sum(weight * (correct * np.log(p)
                                       + (1 - correct) * np.log(1 - p))))

    best = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded").x
    return lambda x: expit(logit(_clip(np.asarray(x, dtype=float))) / best)


def fit_beta(conf, correct, weight):
    """Beta calibration, a two parameter generalisation of Platt scaling."""
    p = _clip(conf)
    design = np.column_stack([np.log(p), -np.log(1.0 - p)])
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(design, correct, sample_weight=weight)

    def apply(x):
        q = _clip(np.asarray(x, dtype=float))
        return model.predict_proba(
            np.column_stack([np.log(q), -np.log(1.0 - q)]))[:, 1]

    return apply


def fit_histogram(conf, correct, weight, bins=N_ECE_BINS):
    edges = np.quantile(conf, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, len(edges) - 2)
    num = np.bincount(idx, weights=weight * correct, minlength=len(edges) - 1)
    den = np.bincount(idx, weights=weight, minlength=len(edges) - 1)
    value = np.divide(num, den, out=np.full_like(num, 0.5), where=den > 0)

    def apply(x):
        j = np.clip(np.digitize(np.asarray(x, dtype=float), edges[1:-1]),
                    0, len(value) - 1)
        return value[j]

    return apply


FAMILIES = {
    "raw confidence": fit_identity,
    "temperature scaling": fit_temperature,
    "Platt scaling": fit_platt,
    "beta calibration": fit_beta,
    "histogram binning": fit_histogram,
    "isotonic regression": fit_isotonic,
}


def expected_calibration_error(conf, correct, weight, bins=N_ECE_BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = weight.sum()
    err = 0.0
    for b in range(bins):
        m = idx == b
        w = weight[m].sum()
        if w <= 0:
            continue
        err += (w / total) * abs(np.average(correct[m], weights=weight[m])
                                 - np.average(conf[m], weights=weight[m]))
    return float(err)


def brier(conf, correct, weight):
    return float(np.average((conf - correct) ** 2, weights=weight))


def classwise(fitter, conf, correct, weight, pred, shrink=SHRINK_AT):
    """Per predicted class map, mixed towards the pooled map where support is thin."""
    pooled = fitter(conf, correct, weight)
    maps = {}
    for c in range(NUM_CLASSES):
        m = pred == c
        n = float(weight[m].sum())
        if n <= 0:
            maps[c] = (pooled, 0.0)
            continue
        try:
            local = fitter(conf[m], correct[m], weight[m])
        except Exception:
            local = pooled
        maps[c] = (local, min(1.0, n / shrink))

    def apply(x, c):
        local, w = maps[c]
        return w * np.asarray(local(x), dtype=float) + (1 - w) * np.asarray(
            pooled(x), dtype=float)

    return apply


def apply_classwise(mapping, conf, pred):
    """Evaluate a per class map on points carrying their predicted class."""
    out = np.empty(len(conf))
    for c in range(NUM_CLASSES):
        m = pred == c
        if m.any():
            out[m] = mapping(conf[m], c)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    units = {f: load_units(f) for f in FOLDS}
    quality = {f: cr.mean_iou(units[f].confusion, "union") for f in FOLDS}
    centres = bin_centres()

    rows = []
    for f in FOLDS:
        src = load_source_points(f)
        conf = src["sample_conf"].astype(np.float64)
        correct = src["sample_correct"].astype(np.float64)
        weight = src["sample_weight"].astype(np.float64)
        pred = src["sample_pred"].astype(np.int64)
        # Cross fitting is grouped by patch. Two points of the same 50 m patch
        # are far from independent, so splitting them between the fitting and
        # the evaluation side would let a calibrator see the neighbourhood it
        # is scored on and report an error that is too small.
        if len(conf) % POINTS_PER_PATCH:
            raise SystemExit("calibration sample is not a whole number of patches")
        n_patches = len(conf) // POINTS_PER_PATCH
        block = np.arange(len(conf)) // POINTS_PER_PATCH
        groups = np.array_split(rng.permutation(n_patches), 5)
        parts = [np.flatnonzero(np.isin(block, g)) for g in groups]
        u = units[f]

        for name, fitter in FAMILIES.items():
            held = np.empty(len(conf))
            for k in range(len(parts)):
                test = parts[k]
                train = np.concatenate([parts[j] for j in range(len(parts))
                                        if j != k])
                mapping = classwise(fitter, conf[train], correct[train],
                                    weight[train], pred[train])
                held[test] = apply_classwise(mapping, conf[test], pred[test])

            full = classwise(fitter, conf, correct, weight, pred)
            in_sample = apply_classwise(full, conf, pred)
            target_map = np.stack([full(centres, c) for c in range(NUM_CLASSES)])
            per_class = np.einsum("ucb,cb->uc", u.conf_hist, target_map)
            per_class = np.divide(per_class, u.n_pred,
                                  out=np.zeros_like(per_class),
                                  where=u.n_pred > 0)
            rows.append({
                "fold": f, "family": name,
                "ece_in_sample": expected_calibration_error(
                    in_sample, correct, weight),
                "ece_out_of_sample": expected_calibration_error(
                    held, correct, weight),
                "brier_out_of_sample": brier(held, correct, weight),
                "recall_20_class_averaged": mt.recall_at(
                    quality[f], sc.aggregate(per_class, u.n_pred, 0.0), 0.20),
                "recall_20_point_weighted": mt.recall_at(
                    quality[f], sc.aggregate(per_class, u.n_pred, 1.0), 0.20),
            })

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "calibration_by_fold.csv", index=False)
    summary = detail.groupby("family", as_index=False).mean(numeric_only=True)
    summary = summary.sort_values("recall_20_class_averaged", ascending=False)
    summary.to_csv(OUT / "calibration_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
