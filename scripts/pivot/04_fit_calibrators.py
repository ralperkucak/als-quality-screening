"""Fit every source-side mapping the estimator and its baselines need.

Nothing here touches a target patch. The inputs are the prototypes and the
calibration subsample written by the previous script, both from labelled source
validation data; the outputs are the class-wise confidence calibrator, the
empirical support distribution, and the confidence thresholds the ATC baselines
are defined by.

Fitting the baselines here rather than beside the proposed score keeps them on
exactly the same footing: same source points, same weights, same fold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoquery_als.config import NUM_CLASSES, OUTPUT_DIR  # noqa: E402
from geoquery_als.reliability.calibration import (  # noqa: E402
    average_thresholded_confidence, classwise_atc_thresholds, fit_calibrator,
    fit_support)
from geoquery_als.reliability.statistics import cosine_distance_to_prototype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--backbone", default="randlanet")
    args = ap.parse_args()

    cal_dir = OUTPUT_DIR / "pivot" / "calibration"
    sfx = "" if args.backbone == "randlanet" else f"_{args.backbone}"
    folds = args.folds or sorted({p.stem.split("_")[1]
                                  for p in cal_dir.glob(f"source_D?{sfx}.npz")})

    print(f"{'fold':5s}{'acc':>7s}{'ATC t':>8s}{'ECE ham':>9s}{'ECE kalibre':>13s}"
          f"{'medyan uzaklık':>16s}")
    for fold in folds:
        d = np.load(cal_dir / f"source_{fold}{sfx}.npz")
        conf = d["sample_conf"].astype(np.float64)
        correct = d["sample_correct"].astype(np.float64)
        pred = d["sample_pred"].astype(np.int64)
        weight = d["sample_weight"].astype(np.float64)
        emb = d["sample_emb"].astype(np.float32)
        prototypes = d["prototypes"]

        calibrator = fit_calibrator(conf, correct, pred, weight)
        distance = cosine_distance_to_prototype(emb, prototypes, pred)
        support = fit_support(distance, pred)
        atc_t = average_thresholded_confidence(conf, correct, weight)
        atc_class = classwise_atc_thresholds(conf, correct, pred, weight)

        # Source-side reference quantities the target estimates are compared against.
        source_accuracy = float(np.average(correct, weights=weight))
        source_mean_conf = float(np.average(conf, weights=weight))
        class_acc = np.full(NUM_CLASSES, np.nan)
        class_conf = np.full(NUM_CLASSES, np.nan)
        class_prevalence = np.zeros(NUM_CLASSES)
        for c in range(NUM_CLASSES):
            m = pred == c
            class_prevalence[c] = weight[m].sum() / weight.sum()
            if weight[m].sum() > 0:
                class_acc[c] = float(np.average(correct[m], weights=weight[m]))
                class_conf[c] = float(np.average(conf[m], weights=weight[m]))

        calibrated = calibrator(conf, pred)
        ece_raw = _ece(conf, correct, weight)
        ece_cal = _ece(calibrated, correct, weight)

        np.savez_compressed(
            cal_dir / f"fitted_{fold}{sfx}.npz",
            prototypes=prototypes,
            atc_threshold=atc_t, atc_threshold_classwise=atc_class,
            source_accuracy=source_accuracy, source_mean_confidence=source_mean_conf,
            source_class_accuracy=class_acc, source_class_confidence=class_conf,
            source_class_prevalence=class_prevalence,
            ece_raw=ece_raw, ece_calibrated=ece_cal,
            **{f"cal_{k}": v for k, v in calibrator.as_dict().items()},
            **{f"sup_{k}": v for k, v in support.as_dict().items()})
        print(f"{fold:5s}{source_accuracy:7.3f}{atc_t:8.3f}{ece_raw:9.4f}"
              f"{ece_cal:13.4f}{np.median(distance):16.4f}")

    print(f"\nwrote fitted_*.npz to {cal_dir}")
    print("every quantity above is estimated on labelled source validation only")
    return 0


def _ece(conf: np.ndarray, correct: np.ndarray, weight: np.ndarray,
         bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = weight.sum()
    out = 0.0
    for b in range(bins):
        m = idx == b
        w = weight[m].sum()
        # A bin can be non-empty and still carry no weight, in which case its
        # contribution is zero and the weighted mean inside it is undefined.
        if w <= 0:
            continue
        out += w / total * abs(np.average(conf[m], weights=weight[m])
                               - np.average(correct[m], weights=weight[m]))
    return float(out)


if __name__ == "__main__":
    raise SystemExit(main())
