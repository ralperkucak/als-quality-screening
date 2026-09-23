"""Inspection prioritisation under a fixed review budget.

Each held-out region is ordered by every candidate score and the lowest ranked
fraction is treated as the set an operator would examine. What the table records
is how much of the genuinely poor output that set contains, how much of the
region's quality deficit it removes, and how far that is from what reviewing the
same number of units at random would give. The null is hypergeometric and exact,
so the p values carry no simulation floor.
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
BUDGETS = (0.10, 0.20, 0.30)
SEED = 20260920
DRAWS = 4000

# name -> (point statistic, class weighting exponent)
METHODS = {
    "class-specific ATC": ("atc_classwise", 0.0),
    "class-specific DoC": ("doc_classwise", 0.0),
    "class-averaged calibrated confidence": ("calibrated", 0.0),
    "class-averaged raw confidence": ("confidence", 0.0),
    "class-averaged negative entropy": ("negative_entropy", 0.0),
    "average thresholded confidence": ("atc_global", 1.0),
    "mean calibrated confidence": ("calibrated", 1.0),
    "mean raw confidence": ("confidence", 1.0),
    "mean negative entropy": ("negative_entropy", 1.0),
    "feature support reliability": ("reliability", 1.0),
}


def evaluate(criterion: str = "miou_union", reference: float = None) -> pd.DataFrame:
    units = {f: load_units(f) for f in FOLDS}
    quality = {f: cr.CRITERIA[criterion](units[f].confusion) for f in FOLDS}
    deficit = {f: mt.deficit_from(quality[f], reference) for f in FOLDS}

    rows = []
    for name, (statistic, alpha) in METHODS.items():
        for f in FOLDS:
            s = sc.score(units[f], statistic, alpha)
            row = {"method": name, "statistic": statistic, "alpha": alpha,
                   "fold": f, "n_units": units[f].n_units}
            for q in BUDGETS:
                rep = mt.recall_report(quality[f], s, q)
                row[f"recall_{int(q * 100)}"] = rep["recall"]
                row[f"p_{int(q * 100)}"] = rep["p_exact"]
                row[f"random_{int(q * 100)}"] = rep["random_recall"]
            row["captured_20"] = mt.captured_deficit(deficit[f], s, 0.20)
            row["aurc"] = mt.aurc(deficit[f], s)
            row["aurc_normalised"] = mt.normalised_aurc(deficit[f], s)
            row["spearman"] = mt.rank_correlation(quality[f], s)
            row["ndcg"] = mt.ndcg(quality[f], s, deficit[f])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_difference(a: str, b: str, criterion: str = "miou_union",
                      budget: float = 0.20, draws: int = DRAWS) -> dict:
    """Paired contrast between two methods, resampling regions then units."""
    rng = np.random.default_rng(SEED)
    units = {f: load_units(f) for f in FOLDS}
    quality = {f: cr.CRITERIA[criterion](units[f].confusion) for f in FOLDS}
    sa = {f: sc.score(units[f], *METHODS[a]) for f in FOLDS}
    sb = {f: sc.score(units[f], *METHODS[b]) for f in FOLDS}

    def statistic(draw_units):
        vals = []
        for f, idx in draw_units:
            vals.append(mt.recall_at(quality[f][idx], sa[f][idx], budget)
                        - mt.recall_at(quality[f][idx], sb[f][idx], budget))
        return float(np.mean(vals))

    observed = statistic([(f, np.arange(units[f].n_units)) for f in FOLDS])
    boot = np.empty(draws)
    for i in range(draws):
        drawn = rng.choice(FOLDS, size=len(FOLDS), replace=True)
        pick = [(f, rng.integers(0, units[f].n_units, units[f].n_units))
                for f in drawn]
        boot[i] = statistic(pick)
    per_fold = [mt.recall_at(quality[f], sa[f], budget)
                - mt.recall_at(quality[f], sb[f], budget) for f in FOLDS]
    return {"contrast": f"{a} minus {b}", "difference": observed,
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "draws_at_or_below_zero": int((boot <= 0).sum()), "draws": draws,
            "folds_positive": int(sum(v > 0 for v in per_fold)),
            "per_fold": [round(v, 4) for v in per_fold]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    detail = evaluate()
    detail.to_csv(OUT / "retrieval_by_fold.csv", index=False)

    summary = (detail.groupby(["method", "statistic", "alpha"], as_index=False)
               .agg({c: "mean" for c in detail.columns
                     if c.startswith(("recall_", "random_", "captured_"))
                     or c in ("aurc", "aurc_normalised", "spearman", "ndcg")}))
    worst_p = (detail.groupby("method")[["p_10", "p_20", "p_30"]].max()
               .rename(columns=lambda c: f"max_{c}"))
    summary = summary.merge(worst_p, on="method").sort_values("recall_20",
                                                              ascending=False)
    summary.to_csv(OUT / "retrieval_summary.csv", index=False)
    print(summary[["method", "recall_10", "recall_20", "recall_30", "captured_20",
                   "aurc_normalised", "spearman", "max_p_20"]]
          .round(4).to_string(index=False))

    contrasts = [
        ("class-averaged calibrated confidence", "mean calibrated confidence"),
        ("class-specific ATC", "average thresholded confidence"),
        ("class-averaged raw confidence", "mean raw confidence"),
        ("class-averaged negative entropy", "mean negative entropy"),
    ]
    out = [paired_difference(a, b) for a, b in contrasts]
    frame = pd.DataFrame(out)
    frame.to_csv(OUT / "retrieval_contrasts.csv", index=False)
    print()
    print(frame[["contrast", "difference", "lo", "hi", "folds_positive"]]
          .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
