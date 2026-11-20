"""Full metric rows for the regrouped and pooled-feature supervised baselines."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "rs"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs.store import FOLDS, load_units  # noqa: E402
from geoquery_als.rs.surface import lift  # noqa: E402

lc = importlib.import_module("61_baseline_learning_curve")
fa = importlib.import_module("63_baseline_features")
OUT = ROOT / "outputs" / "rs"


def main() -> int:
    rows = []
    for fold in FOLDS:
        units = load_units(fold)
        q = cr.mean_iou(units.confusion, "union")
        d = mt.deficit_from(q)
        k = mt.budget_size(units.n_units, 0.20) / units.n_units
        parts = lc.source_tiles(fold)
        y = np.concatenate([cr.mean_iou(p.confusion, "union") for p in parts])
        for fname, feat in (("class-wise", lambda u: lc.features(u, "calibrated")),
                            ("pooled", fa.pooled_features)):
            x = np.vstack([feat(p) for p in parts])
            for kind in ("ridge", "gradient boosting"):
                s = lc.fit_predict(kind, x, y, feat(units))
                rep = mt.recall_report(q, s, 0.20)
                rows.append({"fold": fold, "method": f"{kind}, regrouped, {fname}",
                             "recall": rep["recall"], "lift": lift(rep["recall"], k),
                             "naurc": mt.normalised_aurc(d, s),
                             "spearman": mt.rank_correlation(q, s),
                             "p": rep["p_exact"]})
    df = pd.DataFrame(rows)
    s = df.groupby("method").agg(recall=("recall", "mean"), lift=("lift", "mean"),
                                 naurc=("naurc", "mean"), spearman=("spearman", "mean"),
                                 max_p=("p", "max")).reset_index()
    s.to_csv(OUT / "baseline_table_rows.csv", index=False)
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
