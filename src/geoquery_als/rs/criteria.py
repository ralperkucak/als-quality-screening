"""Unit-level quality criteria, all computed from the unit confusion matrix.

A quality control programme states what it means by a good delivery. The choice
is not neutral: a criterion that averages over classes and a criterion that
averages over points disagree about which parts of a map product are worst
whenever the class distribution is long tailed. The criteria here are written as
one family with a single exponent so that the choice becomes an explicit
parameter of the analysis rather than an implicit convention.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def class_iou(confusion: np.ndarray) -> np.ndarray:
    """Per-class intersection over union, (U, C).

    `confusion` is (U, C, C) with reference on the first axis and prediction on
    the second. Classes absent from both sides give not a number.
    """
    tp = np.einsum("ucc->uc", confusion)
    ref = confusion.sum(axis=2)
    pred = confusion.sum(axis=1)
    union = ref + pred - tp
    out = np.divide(tp, union, out=np.full_like(tp, np.nan, dtype=float),
                    where=union > 0)
    return out


def class_f1(confusion: np.ndarray) -> np.ndarray:
    """Per-class F1 score, (U, C)."""
    tp = np.einsum("ucc->uc", confusion)
    ref = confusion.sum(axis=2)
    pred = confusion.sum(axis=1)
    denom = ref + pred
    return np.divide(2.0 * tp, denom, out=np.full_like(tp, np.nan, dtype=float),
                     where=denom > 0)


def overall_accuracy(confusion: np.ndarray) -> np.ndarray:
    """Proportion of correctly predicted points, (U,)."""
    tp = np.einsum("ucc->uc", confusion).sum(axis=1)
    total = confusion.sum(axis=(1, 2))
    return np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)


def presence(confusion: np.ndarray, rule: str = "union") -> np.ndarray:
    """Boolean mask of the classes entering the average, (U, C)."""
    ref = confusion.sum(axis=2) > 0
    pred = confusion.sum(axis=1) > 0
    if rule == "union":
        return ref | pred
    if rule == "reference":
        return ref
    if rule == "fixed":
        return np.ones_like(ref, dtype=bool)
    raise ValueError(f"unknown presence rule: {rule}")


def mean_iou(confusion: np.ndarray, rule: str = "union") -> np.ndarray:
    """Mean intersection over union over the class set fixed by `rule`, (U,)."""
    iou = class_iou(confusion)
    mask = presence(confusion, rule)
    filled = np.where(np.isnan(iou), 0.0, iou)
    count = mask.sum(axis=1)
    total = np.where(mask, filled, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def macro_f1(confusion: np.ndarray, rule: str = "union") -> np.ndarray:
    f1 = class_f1(confusion)
    mask = presence(confusion, rule)
    filled = np.where(np.isnan(f1), 0.0, f1)
    count = mask.sum(axis=1)
    total = np.where(mask, filled, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def criterion_weights(confusion: np.ndarray, beta: float,
                      rule: str = "union") -> np.ndarray:
    """Class weights of the criterion family, (U, C).

    The weight of class c in unit B is proportional to its share of the points
    that the class occupies on the union of the reference and the prediction,
    raised to the power beta. At beta equal to zero every class present carries
    the same weight, which recovers the mean intersection over union. At beta
    equal to one the weight is the point share, which reproduces the behaviour of
    a point weighted criterion such as overall accuracy.
    """
    mask = presence(confusion, rule)
    ref = confusion.sum(axis=2)
    pred = confusion.sum(axis=1)
    tp = np.einsum("ucc->uc", confusion)
    occupancy = ref + pred - tp
    share = occupancy / np.maximum(occupancy.sum(axis=1, keepdims=True), EPS)
    w = np.where(mask, np.power(np.maximum(share, EPS), beta), 0.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), EPS)


def quality(confusion: np.ndarray, beta: float = 0.0,
            rule: str = "union") -> np.ndarray:
    """Class weighted intersection over union with exponent beta, (U,)."""
    iou = np.where(np.isnan(class_iou(confusion)), 0.0, class_iou(confusion))
    w = criterion_weights(confusion, beta, rule)
    return (w * iou).sum(axis=1)


def rare_class_quality(confusion: np.ndarray, classes) -> np.ndarray:
    """Mean intersection over union restricted to a nominated set of classes.

    Units in which none of the nominated classes appears on either side return
    not a number and are excluded by the caller, since the operational question
    does not arise there.
    """
    iou = class_iou(confusion)
    mask = presence(confusion, "union")
    sel = np.zeros(confusion.shape[1], dtype=bool)
    sel[list(classes)] = True
    use = mask & sel[None, :]
    filled = np.where(np.isnan(iou), 0.0, iou)
    count = use.sum(axis=1)
    total = np.where(use, filled, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


CRITERIA = {
    "miou_union": lambda c: mean_iou(c, "union"),
    "miou_reference": lambda c: mean_iou(c, "reference"),
    "miou_fixed": lambda c: mean_iou(c, "fixed"),
    "overall_accuracy": overall_accuracy,
    "macro_f1": lambda c: macro_f1(c, "union"),
}
