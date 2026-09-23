"""Does a recalibration fitted on the source hold in the held-out region?

The recalibration maps are fitted on labelled source validation points and then
carried unchanged to a region the model has never seen. Reporting their
calibration error only where they were fitted leaves the question that matters
unanswered, so the same comparison is made on the target side, using the target
reference labels for evaluation alone and never for producing a score.

The cached statistics hold, for each predicted class, the number of points, the
summed confidence before and after recalibration, and the confusion matrix. The
quantity those support is the class-conditional reliability gap: the difference
between the mean confidence of the points assigned to a class and the share of
them that are correct, averaged over classes in proportion to how many points
each holds. That is the expected calibration error with the predicted class as
the binning variable, which is also the granularity at which the tile score
consumes the confidences.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.store import FOLDS, load_units  # noqa: E402

OUT = ROOT / "outputs" / "rs"
CLASSES = ("other", "ground", "vegetation", "building", "water", "bridge",
           "permanent structure")


def gap(units) -> dict:
    """Class-conditional reliability of one region, pooled over its tiles."""
    n = units.n_pred.sum(axis=0)                         # (C,)
    correct = np.einsum("ucc->c", units.confusion).astype(float)
    raw = units.sum_conf.sum(axis=0)
    calibrated = units.sum_calibrated.sum(axis=0)
    total = n.sum()

    present = n > 0
    precision = np.divide(correct, n, out=np.zeros_like(n, dtype=float),
                          where=present)
    mean_raw = np.divide(raw, n, out=np.zeros_like(raw), where=present)
    mean_cal = np.divide(calibrated, n, out=np.zeros_like(calibrated),
                         where=present)
    weight = n / max(total, 1)
    return {
        "accuracy": float(correct.sum() / max(total, 1)),
        "mean_confidence_raw": float(raw.sum() / max(total, 1)),
        "mean_confidence_calibrated": float(calibrated.sum() / max(total, 1)),
        "gap_raw": float((weight * np.abs(mean_raw - precision)).sum()),
        "gap_calibrated": float(
            (weight * np.abs(mean_cal - precision)).sum()),
        "per_class": (precision, mean_raw, mean_cal, n),
    }


def main() -> int:
    rows, per_class_rows = [], []
    for fold in FOLDS:
        for family in ("source", "target"):
            units = load_units(fold, family=family)
            g = gap(units)
            precision, mean_raw, mean_cal, n = g.pop("per_class")
            rows.append({"fold": fold, "family": family, **g,
                         "overconfidence_raw":
                             g["mean_confidence_raw"] - g["accuracy"],
                         "overconfidence_calibrated":
                             g["mean_confidence_calibrated"] - g["accuracy"]})
            for c, name in enumerate(CLASSES):
                if n[c] == 0:
                    continue
                per_class_rows.append({
                    "fold": fold, "family": family, "class": name,
                    "n_points": int(n[c]), "precision": float(precision[c]),
                    "confidence_raw": float(mean_raw[c]),
                    "confidence_calibrated": float(mean_cal[c]),
                    "gap_raw": float(mean_raw[c] - precision[c]),
                    "gap_calibrated": float(mean_cal[c] - precision[c])})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "target_calibration_by_region.csv", index=False)
    summary = detail.groupby("family", as_index=False).mean(numeric_only=True)
    summary.to_csv(OUT / "target_calibration.csv", index=False)

    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(OUT / "target_calibration_by_class.csv", index=False)
    class_summary = (per_class.groupby(["family", "class"], as_index=False)
                     .mean(numeric_only=True))
    class_summary.to_csv(OUT / "target_calibration_class_summary.csv",
                         index=False)

    print(detail.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))
    print("\nclass-conditional gap after recalibration, target side")
    print(per_class[per_class["family"] == "target"]
          .groupby("class")[["n_points", "precision", "confidence_calibrated",
                             "gap_calibrated"]]
          .mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
