"""Label free unit scores and the class weighting family that connects them.

A point level confidence statistic has to be reduced to one number per spatial
unit before it can order units for review. The reduction is written here as a
one parameter family, so that the point weighted average and the class averaged
alternative are two settings of the same estimator rather than two competing
methods.
"""
from __future__ import annotations

import numpy as np

from .store import HIST_BINS, UnitStats, bin_centres, load_calibration

EPS = 1e-12


def class_weights(n_pred: np.ndarray, alpha: float) -> np.ndarray:
    """Class weights of the score family, (U, C).

    The weight of class c in unit B is its share of the points predicted as that
    class raised to the power alpha, normalised over the classes the model
    predicts in the unit. Setting alpha to one recovers the point weighted
    average, since the weights are then the point shares themselves. Setting
    alpha to zero gives every predicted class the same weight.
    """
    share = n_pred / np.maximum(n_pred.sum(axis=1, keepdims=True), EPS)
    present = n_pred > 0
    w = np.where(present, np.power(np.maximum(share, EPS), alpha), 0.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), EPS)


def class_means(total: np.ndarray, n_pred: np.ndarray) -> np.ndarray:
    """Per class mean of a summed point statistic, (U, C)."""
    return np.divide(total, n_pred, out=np.zeros_like(total, dtype=float),
                     where=n_pred > 0)


def aggregate(per_class: np.ndarray, n_pred: np.ndarray, alpha: float) -> np.ndarray:
    """Reduce a per class statistic to one value per unit at exponent alpha."""
    return (class_weights(n_pred, alpha) * per_class).sum(axis=1)


def thresholded_fraction(conf_hist: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per class share of points whose confidence reaches the class threshold.

    This is the class conditional form of average thresholded confidence. The
    histogram carries two hundred equal width bins on the unit interval, so a
    threshold is applied at bin resolution and the discretisation error is
    bounded by the width of one bin.
    """
    centres = bin_centres()
    u, c, _ = conf_hist.shape
    above = np.empty((u, c))
    for k in range(c):
        mask = centres >= thresholds[k]
        above[:, k] = conf_hist[:, k, mask].sum(axis=1)
    total = conf_hist.sum(axis=2)
    return np.divide(above, total, out=np.zeros_like(above), where=total > 0)


def apply_mapping(conf_hist: np.ndarray, mapping) -> np.ndarray:
    """Per class mean of a recalibrated confidence, (U, C).

    `mapping` takes the bin centres and a class index and returns the mapped
    value of each bin, so an arbitrary monotone recalibration can be applied to
    the target without revisiting the points.
    """
    centres = bin_centres()
    u, c, _ = conf_hist.shape
    out = np.zeros((u, c))
    total = conf_hist.sum(axis=2)
    for k in range(c):
        mapped = mapping(centres, k)
        out[:, k] = conf_hist[:, k, :] @ mapped
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)


def per_class_statistics(units: UnitStats, fold: str = None,
                         backbone: str = "randlanet") -> dict:
    """Every per class point statistic the study uses, each (U, C)."""
    fitted = load_calibration(fold or units.fold, backbone)
    n = units.n_pred
    stats = {
        "confidence": class_means(units.sum_conf, n),
        "calibrated": class_means(units.sum_calibrated, n),
        "negative_entropy": -class_means(units.sum_entropy, n),
        "support": class_means(units.sum_support, n),
        "reliability": class_means(units.sum_reliability, n),
    }
    thr_global = float(fitted["atc_threshold"])
    thr_class = np.asarray(fitted["atc_threshold_classwise"], dtype=float)
    stats["atc_global"] = thresholded_fraction(
        units.conf_hist, np.full(n.shape[1], thr_global))
    stats["atc_classwise"] = thresholded_fraction(units.conf_hist, thr_class)

    acc = np.asarray(fitted["source_class_accuracy"], dtype=float)
    conf = np.asarray(fitted["source_class_confidence"], dtype=float)
    stats["doc_classwise"] = acc[None, :] - (conf[None, :] - stats["confidence"])
    stats["source_accuracy_prior"] = np.broadcast_to(
        acc[None, :], stats["confidence"].shape).copy()
    return stats


def score(units: UnitStats, statistic: str, alpha: float,
          fold: str = None, backbone: str = "randlanet") -> np.ndarray:
    """One label free score per unit, oriented so that larger means better."""
    per_class = per_class_statistics(units, fold, backbone)[statistic]
    return aggregate(per_class, units.n_pred, alpha)


STATISTICS = ("confidence", "calibrated", "negative_entropy", "atc_classwise",
              "doc_classwise", "reliability", "support", "atc_global",
              "source_accuracy_prior")
