"""Source-fitted calibrators and the empirical support distribution.

Everything here is estimated on labelled source validation data and then applied
unchanged to the target. That is the assumption the whole Setting A design rests
on, and it is an assumption rather than a guarantee: if the shift is large
enough, a mapping learned on the source will be biased on the target. The bias
is measurable per fold and is reported rather than assumed away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import NUM_CLASSES

MIN_CLASS_POINTS = 500        # below this a class-wise fit is shrunk to the global one


@dataclass
class ClasswiseCalibrator:
    """Maps a raw confidence to the probability that the prediction is correct.

    A separate isotonic fit per predicted class, because class imbalance biases a
    single global calibration: the dominant classes decide the mapping and the
    rare ones inherit a curve that does not describe them. Classes with too few
    source points to support their own fit are shrunk towards the global curve
    rather than left to overfit, and the shrinkage weight is recorded so the
    reader can see which classes are effectively global.
    """

    knots_x: np.ndarray          # (C, K) confidence grid per class
    knots_y: np.ndarray          # (C, K) calibrated value per class
    global_x: np.ndarray
    global_y: np.ndarray
    shrinkage: np.ndarray        # (C,) 1.0 means fully class-wise
    n_points: np.ndarray

    def __call__(self, conf: np.ndarray, pred: np.ndarray) -> np.ndarray:
        out = np.interp(conf, self.global_x, self.global_y)
        for c in range(NUM_CLASSES):
            m = pred == c
            if not m.any() or self.shrinkage[c] <= 0:
                continue
            cls = np.interp(conf[m], self.knots_x[c], self.knots_y[c])
            w = self.shrinkage[c]
            out[m] = w * cls + (1.0 - w) * out[m]
        return np.clip(out, 0.0, 1.0)

    def as_dict(self) -> dict:
        return {"knots_x": self.knots_x, "knots_y": self.knots_y,
                "global_x": self.global_x, "global_y": self.global_y,
                "shrinkage": self.shrinkage, "n_points": self.n_points}

    @classmethod
    def from_npz(cls, data, prefix: str = "cal_") -> "ClasswiseCalibrator":
        return cls(*(data[prefix + k] for k in
                     ("knots_x", "knots_y", "global_x", "global_y",
                      "shrinkage", "n_points")))


def _isotonic(conf: np.ndarray, correct: np.ndarray, weight: np.ndarray,
              grid: np.ndarray) -> np.ndarray:
    """Weighted isotonic regression of correctness on confidence, read on a grid."""
    from sklearn.isotonic import IsotonicRegression
    if len(conf) < 2 or correct.max() == correct.min():
        return np.full_like(grid, float(np.average(correct, weights=weight)))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(conf.astype(np.float64), correct.astype(np.float64),
            sample_weight=weight.astype(np.float64))
    return iso.predict(grid)


def fit_calibrator(conf: np.ndarray, correct: np.ndarray, pred: np.ndarray,
                   weight: np.ndarray, n_knots: int = 64) -> ClasswiseCalibrator:
    grid = np.linspace(1.0 / NUM_CLASSES, 1.0, n_knots)
    global_y = _isotonic(conf, correct, weight, grid)
    kx = np.tile(grid, (NUM_CLASSES, 1))
    ky = np.tile(global_y, (NUM_CLASSES, 1))
    shrink = np.zeros(NUM_CLASSES)
    counts = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        m = pred == c
        counts[c] = m.sum()
        if counts[c] == 0:
            continue
        ky[c] = _isotonic(conf[m], correct[m], weight[m], grid)
        # Linear ramp to full class-wise weight at twice the minimum support.
        shrink[c] = float(np.clip(counts[c] / (2.0 * MIN_CLASS_POINTS), 0.0, 1.0))
    return ClasswiseCalibrator(kx, ky, grid, global_y, shrink, counts)


@dataclass
class SupportDistribution:
    """Empirical distribution of source cosine distance, one per predicted class.

    The support score of a target point is one minus the fraction of source
    points of the same predicted class that lie closer to the prototype. A point
    sitting where the source model has seen many correct examples scores near
    one; a point beyond anything the source contains scores near zero.
    """

    quantiles: np.ndarray        # (C, Q) sorted source distances
    n_points: np.ndarray

    def __call__(self, distance: np.ndarray, pred: np.ndarray) -> np.ndarray:
        out = np.zeros_like(distance, dtype=np.float64)
        for c in range(NUM_CLASSES):
            m = pred == c
            if not m.any():
                continue
            q = self.quantiles[c]
            if not np.isfinite(q).any():
                continue
            out[m] = 1.0 - np.searchsorted(q, distance[m], side="left") / len(q)
        return np.clip(out, 0.0, 1.0)

    def as_dict(self) -> dict:
        return {"quantiles": self.quantiles, "n_points": self.n_points}

    @classmethod
    def from_npz(cls, data, prefix: str = "sup_") -> "SupportDistribution":
        return cls(data[prefix + "quantiles"], data[prefix + "n_points"])


def fit_support(distance: np.ndarray, pred: np.ndarray,
                n_quantiles: int = 1024) -> SupportDistribution:
    q = np.full((NUM_CLASSES, n_quantiles), np.nan)
    counts = np.zeros(NUM_CLASSES)
    probs = np.linspace(0.0, 1.0, n_quantiles)
    for c in range(NUM_CLASSES):
        m = pred == c
        counts[c] = m.sum()
        if counts[c] < 2:
            continue
        q[c] = np.quantile(distance[m].astype(np.float64), probs)
    return SupportDistribution(q, counts)


def average_thresholded_confidence(source_conf: np.ndarray,
                                   source_correct: np.ndarray,
                                   source_weight: np.ndarray) -> float:
    """The ATC threshold of Garg et al.: source error rate matched on confidence.

    The threshold t is chosen so that the weighted fraction of source points
    below t equals the weighted source error rate. The estimate of target
    accuracy is then the fraction of target points above t.
    """
    order = np.argsort(source_conf)
    conf, w = source_conf[order], source_weight[order]
    err = np.average(1.0 - source_correct, weights=source_weight)
    cum = np.cumsum(w) / w.sum()
    idx = int(np.searchsorted(cum, err, side="left"))
    return float(conf[min(idx, len(conf) - 1)])


def classwise_atc_thresholds(source_conf: np.ndarray, source_correct: np.ndarray,
                             source_pred: np.ndarray,
                             source_weight: np.ndarray) -> np.ndarray:
    """One ATC threshold per predicted class, following Li et al.

    A single threshold is set by whichever classes dominate the source, so on an
    imbalanced problem the rare classes are thresholded by a rule fitted to the
    common ones. This is the baseline the proposed score has to beat.
    """
    out = np.full(NUM_CLASSES, np.nan)
    for c in range(NUM_CLASSES):
        m = source_pred == c
        if m.sum() < MIN_CLASS_POINTS:
            continue
        out[c] = average_thresholded_confidence(source_conf[m], source_correct[m],
                                                source_weight[m])
    globally = average_thresholded_confidence(source_conf, source_correct, source_weight)
    return np.where(np.isfinite(out), out, globally)
