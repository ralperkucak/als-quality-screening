"""Alternative class universes on the criterion side and on the score side.

Two questions are settled here. The first is which classes a tile's quality
criterion should average over: the classes present on either side, the classes
present in the reference, or all seven regardless of presence. The second is
what happens when the score is given the same fixed class universe as the
criterion, so that a class the model predicts nowhere in a tile enters the score
as a zero rather than being skipped. The second variant is the only way the
score can react to a class the model has missed entirely, and whether it helps
is an empirical question rather than a design choice.
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
from geoquery_als.rs.surface import BUDGET, lift  # noqa: E402

OUT = ROOT / "outputs" / "rs"
STATISTIC = "calibrated"
RULES = ("union", "reference", "fixed")
RULE_LABEL = {"union": "Union of prediction and reference",
              "reference": "Reference only",
              "fixed": "All seven classes"}


def fixed_universe_score(per_class: np.ndarray, n_pred: np.ndarray,
                         absent: float = 0.0) -> np.ndarray:
    """Class average over every class, with unpredicted classes set to `absent`.

    The predicted class set of a tile is replaced by the full class universe, so
    a class the model does not propose anywhere in the tile lowers the score
    instead of leaving it unchanged. At the class averaged operating point every
    class carries the same weight, which is what makes the substitution
    well defined: a share raised to a positive power would return a weight of
    zero to exactly the classes the variant exists to notice.
    """
    filled = np.where(n_pred > 0, per_class, absent)
    return filled.mean(axis=1)


def main() -> int:
    rows, variant_rows = [], []
    for fold in FOLDS:
        units = load_units(fold)
        per_class = sc.per_class_statistics(units, fold)[STATISTIC]
        q = mt.budget_size(units.n_units, BUDGET) / units.n_units
        primary = sc.score(units, STATISTIC, 0.0)

        for rule in RULES:
            quality = cr.quality(units.confusion, 0.0, rule)
            deficit = mt.deficit_from(quality)
            report = mt.recall_report(quality, primary, BUDGET)
            rows.append({
                "fold": fold, "rule": rule, "criterion": RULE_LABEL[rule],
                "mean_quality": float(quality.mean()),
                "recall": report["recall"],
                "lift": lift(report["recall"], q),
                "p_exact": report["p_exact"],
                "naurc": mt.normalised_aurc(deficit, primary),
                "spearman": mt.rank_correlation(quality, primary)})

        # The blind spot and the variant that is meant to see it.
        ref_present = units.confusion.sum(axis=2) > 0
        pred_present = units.n_pred > 0
        missed = (ref_present & ~pred_present).any(axis=1)
        variant = fixed_universe_score(per_class, units.n_pred)
        # The control the comparison needs. Averaging over a fixed universe
        # multiplies the score by the number of predicted classes, and
        # averaging the criterion over a fixed universe multiplies it by the
        # number of classes present, so the two share a factor that carries no
        # confidence at all. Ordering on that factor alone measures how much of
        # the agreement is the shared count.
        count = (units.n_pred > 0).sum(axis=1).astype(float)
        k = mt.budget_size(units.n_units, BUDGET)
        for rule in RULES:
            quality = cr.quality(units.confusion, 0.0, rule)
            deficit = mt.deficit_from(quality)
            for name, s in (("predicted class set", primary),
                            ("fixed class universe", variant),
                            ("predicted class count", count)):
                report = mt.recall_report(quality, s, BUDGET)
                variant_rows.append({
                    "fold": fold, "rule": rule, "criterion": RULE_LABEL[rule],
                    "score": name,
                    "recall": report["recall"],
                    "lift": lift(report["recall"], q),
                    "p_exact": report["p_exact"],
                    "naurc": mt.normalised_aurc(deficit, s),
                    "spearman": mt.rank_correlation(quality, s),
                    "share_tiles_with_missed_class": float(missed.mean()),
                    "recall_of_missed_tiles": float(
                        missed[np.argsort(s, kind="stable")[:k]].sum()
                        / max(missed.sum(), 1))})

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "criterion_conventions_by_region.csv", index=False)
    summary = detail.groupby(["rule", "criterion"], as_index=False).mean(
        numeric_only=True)
    # A mean of p values says nothing; the worst region is the claim.
    summary["p_exact"] = detail.groupby(["rule", "criterion"], as_index=False)[
        "p_exact"].max()["p_exact"]
    summary.to_csv(OUT / "criterion_conventions.csv", index=False)

    variants = pd.DataFrame(variant_rows)
    variants.to_csv(OUT / "missing_class_variant_by_region.csv", index=False)
    vsummary = variants.groupby(["rule", "score"], as_index=False).mean(
        numeric_only=True)
    vsummary["p_exact"] = variants.groupby(["rule", "score"], as_index=False)[
        "p_exact"].max()["p_exact"]
    vsummary.to_csv(OUT / "missing_class_variant.csv", index=False)

    print("criterion conventions, same score, averaged over regions")
    print(summary.round(4).to_string(index=False))
    print("\nper region recall")
    print(detail.pivot(index="fold", columns="rule", values="recall")
          .round(4).to_string())
    print("\nscore class universe")
    print(vsummary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
