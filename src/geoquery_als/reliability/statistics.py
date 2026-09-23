"""Per-patch sufficient statistics for label-free reliability assessment.

The network runs on a subsample of each patch and its prediction is transferred
to every stored point by nearest-neighbour assignment, which is the evaluation
protocol the rest of the study uses. Every quantity here is therefore expressed
over the full stored point set: each subsampled point carries the weight of the
full points that map to it, so weighted sums and weighted histograms over the
subsample equal the corresponding full-point quantities exactly.

What is stored per patch is a set of sufficient statistics rather than per-point
arrays (decision P16). Unit-level confusion matrices are sums of patch matrices;
unit-level class IoU follows from those; threshold rules such as ATC are read off
the histograms. Nothing the analysis needs is lost, and a fold costs about ten
megabytes instead of a hundred.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import NUM_CLASSES

N_BINS = 200          # quantisation of a threshold is then about 0.005, negligible


@dataclass
class PatchStatistics:
    """Everything the reliability analysis needs from one patch."""

    patch_id: str
    n_full: int
    confusion: np.ndarray                     # (C, C) reference by prediction
    n_pred: np.ndarray                        # (C,) full points per predicted class
    sum_conf: np.ndarray                      # (C,)
    sum_conf_sq: np.ndarray
    sum_entropy: np.ndarray
    sum_calibrated: np.ndarray
    sum_support: np.ndarray
    sum_reliability: np.ndarray
    conf_hist: np.ndarray                     # (C, N_BINS)
    support_hist: np.ndarray                  # (C, N_BINS)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("n_full", "confusion", "n_pred", "sum_conf", "sum_conf_sq",
                 "sum_entropy", "sum_calibrated", "sum_support", "sum_reliability",
                 "conf_hist", "support_hist")}


@dataclass
class PrototypeAccumulator:
    """Exact class embedding means over correctly predicted full points.

    Accumulating sums rather than storing embeddings keeps the prototypes exact
    while writing nothing per point. Only correctly predicted points contribute,
    so the prototype describes where the source model is right about a class, not
    merely where it guesses it.
    """

    dim: int
    total: np.ndarray = field(default=None)
    count: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        if self.total is None:
            self.total = np.zeros((NUM_CLASSES, self.dim), dtype=np.float64)
            self.count = np.zeros(NUM_CLASSES, dtype=np.float64)

    def update(self, emb: np.ndarray, pred: np.ndarray, label: np.ndarray,
               weight: np.ndarray) -> None:
        """emb (N, D) on the subsample; pred, label, weight aligned with it."""
        correct = (pred == label) & (label >= 0)
        if not correct.any():
            return
        w = weight[correct].astype(np.float64)
        np.add.at(self.total, pred[correct], emb[correct].astype(np.float64) * w[:, None])
        np.add.at(self.count, pred[correct], w)

    def prototypes(self) -> np.ndarray:
        """Unit-norm class prototypes; classes never predicted correctly give NaN."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = self.total / self.count[:, None]
        norm = np.linalg.norm(mu, axis=1, keepdims=True)
        return np.where(norm > 0, mu / np.maximum(norm, 1e-12), np.nan)


def point_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per point, normalised to [0, 1]. probs is (C, N)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=0) / np.log(probs.shape[0])


def cosine_distance_to_prototype(emb: np.ndarray, prototypes: np.ndarray,
                                 pred: np.ndarray) -> np.ndarray:
    """Cosine distance from each point to the prototype of its predicted class.

    Cosine is the primary distance because a 32-dimensional class covariance
    estimated from a handful of rare-class points is not stable enough to invert,
    and an unstable Mahalanobis distance would put noise into the score the study
    is trying to evaluate.
    """
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    proto = prototypes[pred]                                   # (N, D)
    d = 1.0 - np.einsum("nd,nd->n", unit, proto)
    return np.where(np.isfinite(d), d, 1.0)                    # unseen class: no support


def weighted_histogram(values: np.ndarray, pred: np.ndarray, weight: np.ndarray,
                       lo: float, hi: float) -> np.ndarray:
    """(C, N_BINS) weighted histogram of `values`, one row per predicted class."""
    out = np.zeros((NUM_CLASSES, N_BINS), dtype=np.float64)
    scaled = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    bins = np.minimum((scaled * N_BINS).astype(np.int64), N_BINS - 1)
    np.add.at(out, (pred, bins), weight)
    return out


def reference_confusion(pred_full: np.ndarray, label_full: np.ndarray) -> np.ndarray:
    """Reference confusion matrix of one patch. Evaluation only.

    This is the only function in the module that reads a reference label, and
    nothing it returns is ever passed to the feature side. The separation is
    asserted by `tests/pivot/test_no_target_label_in_features.py`.
    """
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    valid = label_full >= 0
    np.add.at(confusion, (label_full[valid].astype(np.int64), pred_full[valid]), 1)
    return confusion


def reliability_features(probs: np.ndarray, emb: np.ndarray, weight: np.ndarray,
                         prototypes: np.ndarray | None, calibrator=None) -> dict:
    """Label-free statistics of one patch.

    Takes model outputs, the source-fitted prototypes and the source-fitted
    calibrator, and nothing else. There is deliberately no parameter capable of
    carrying a target reference label, so target truth cannot reach a feature
    even by mistake.
    """
    n_sub = probs.shape[1]
    pred = probs.argmax(axis=0)
    conf = probs.max(axis=0)
    ent = point_entropy(probs)

    if prototypes is not None:
        support = 1.0 - cosine_distance_to_prototype(emb.T, prototypes, pred)
        support = np.clip(support, 0.0, 1.0)
    else:
        support = np.zeros(n_sub)
    calibrated = calibrator(conf, pred) if calibrator is not None else conf
    reliability = np.sqrt(np.clip(calibrated, 0.0, 1.0) * support)

    def sums(v):
        out = np.zeros(NUM_CLASSES)
        np.add.at(out, pred, v * weight)
        return out

    n_pred = np.zeros(NUM_CLASSES)
    np.add.at(n_pred, pred, weight)
    return {
        "pred": pred, "n_pred": n_pred,
        "sum_conf": sums(conf), "sum_conf_sq": sums(conf ** 2),
        "sum_entropy": sums(ent), "sum_calibrated": sums(calibrated),
        "sum_support": sums(support), "sum_reliability": sums(reliability),
        "conf_hist": weighted_histogram(conf, pred, weight, 1.0 / NUM_CLASSES, 1.0),
        "support_hist": weighted_histogram(support, pred, weight, 0.0, 1.0)}


def accumulate_patch(patch_id: str, probs: np.ndarray, emb: np.ndarray,
                     nn: np.ndarray, label_full: np.ndarray,
                     prototypes: np.ndarray | None,
                     calibrator=None) -> PatchStatistics:
    """Features and reference metrics of one patch, kept in separate calls.

    probs is (C, N_sub), emb is (D, N_sub), nn maps every full point to the index
    of its nearest subsampled point, and label_full holds the reference label of
    every full point. The reference labels reach only `reference_confusion`.
    """
    weight = np.bincount(nn, minlength=probs.shape[1]).astype(np.float64)
    feats = reliability_features(probs, emb, weight, prototypes, calibrator)
    confusion = reference_confusion(feats["pred"][nn], label_full)
    return PatchStatistics(
        patch_id=patch_id, n_full=int(weight.sum()), confusion=confusion,
        **{k: v for k, v in feats.items() if k != "pred"})


def iou_from_confusion(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class IoU and a mask of the classes the metric is defined on.

    A class counts when it appears in the reference or in the prediction
    (decision P11). Restricting to reference-present classes would drop a class
    the model invents where none exists, which is one of the failures a
    reliability estimator has to be able to see.
    """
    tp = np.diag(confusion).astype(np.float64)
    ref = confusion.sum(axis=1).astype(np.float64)
    prd = confusion.sum(axis=0).astype(np.float64)
    union = ref + prd - tp
    present = union > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(present, tp / np.maximum(union, 1e-12), np.nan)
    return iou, present


def unit_metrics(confusion: np.ndarray) -> dict:
    """Reference metrics of one aggregation unit, both mIoU conventions."""
    iou, present = iou_from_confusion(confusion)
    ref_present = confusion.sum(axis=1) > 0
    total = confusion.sum()
    return {
        "mIoU_union": float(np.nanmean(iou[present])) if present.any() else np.nan,
        "n_classes_union": int(present.sum()),
        "mIoU_reference_present": (float(np.nanmean(iou[ref_present]))
                                   if ref_present.any() else np.nan),
        "n_classes_reference_present": int(ref_present.sum()),
        "OA": float(np.diag(confusion).sum() / total) if total else np.nan,
        "n_points": int(total),
    }
