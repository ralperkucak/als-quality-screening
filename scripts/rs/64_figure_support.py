"""Figure: point thinning, patch subsampling and supervised learning curves."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.style import (COL_WIDE, FAMILY_COLOUR, apply_style,  # noqa: E402
                                   despine, panel, save)

DATA = ROOT / "outputs" / "rs"
FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))
CA, PW, RND = FAMILY_COLOUR["class-averaged"], FAMILY_COLOUR["point-weighted"], "#8a8f94"


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_WIDE * 0.98, COL_WIDE * 0.34),
                             gridspec_kw={"wspace": 0.55})

    # (a) point thinning
    ax = axes[0]
    t = pd.read_csv(DATA / "point_thinning_by_region.csv")
    draws = pd.read_csv(DATA / "point_thinning_draws.csv")
    base = draws[draws["draw"] < 0].groupby(["aggregation", "fold"])["recall"].mean()
    full = base.groupby("aggregation").mean()
    order = ["dropout 10%", "dropout 25%", "dropout 50%", "bootstrap"]
    xs = np.arange(len(order) + 1)
    for agg, colour in (("class-averaged", CA), ("point-weighted", PW)):
        g = t[t["aggregation"] == agg].groupby("setting")["recall"]
        mean = [full[agg]] + [g.mean()[o] for o in order]
        lo = [base[agg].min()] + [g.min()[o] for o in order]
        hi = [base[agg].max()] + [g.max()[o] for o in order]
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(xs, mean, "o-", color=colour, ms=3.5, lw=1.2, label=agg)
    ax.axhline(0.2, color=RND, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels(["none", "10%", "25%", "50%", "Poisson"], fontsize=6.8)
    ax.set_xlabel("Point dropout or resampling")
    ax.set_ylabel("Recall of the worst 20%")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=6.4, loc="lower left", handlelength=1.2)
    despine(ax)
    panel(ax, "a", dx=-0.30, dy=1.04)

    # (b) patch subsampling and split halves
    ax = axes[1]
    s = pd.read_csv(DATA / "patch_subsets_by_region.csv")
    for agg, colour in (("class-averaged", CA), ("point-weighted", PW)):
        g = s[(s["aggregation"] == agg) & (s["patches"] > 0)].groupby("patches")["recall"]
        m, lo, hi = g.mean(), g.min(), g.max()
        m.loc[17] = full[agg]
        lo.loc[17], hi.loc[17] = base[agg].min(), base[agg].max()
        m, lo, hi = m.sort_index(), lo.sort_index(), hi.sort_index()
        ax.plot(m.index, m.values, "o-", color=colour, ms=3.5, lw=1.2)
        ax.fill_between(lo.index, lo.values, hi.values, color=colour, alpha=0.15, lw=0)
    half = s[(s["aggregation"] == "class-averaged") & (s["patches"] < 0)].groupby("design")["recall"].mean()
    ax.errorbar([20.5], [half["half scored, other half judged"]], fmt="D", ms=4,
                color=CA, mfc="white")
    ax.errorbar([20.5], [half["quality agreement of the two halves"]], fmt="s", ms=4,
                color="#20242a", mfc="white")
    ax.annotate("other half,\nscore", (20.5, half["half scored, other half judged"]),
                xytext=(6, -10), textcoords="offset points", fontsize=5.8)
    ax.annotate("other half,\nquality of\nfirst half", (20.5, half["quality agreement of the two halves"]),
                xytext=(6, 2), textcoords="offset points", fontsize=5.8)
    ax.axhline(0.2, color=RND, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks([2, 4, 8, 12, 17])
    ax.set_xticklabels(["2", "4", "8", "12", "all"], fontsize=6.8)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Patches scored per tile")
    despine(ax)
    panel(ax, "b", dx=-0.22, dy=1.04)

    # (c) supervised learning curve
    ax = axes[2]
    c = pd.read_csv(DATA / "baseline_learning_curve.csv")
    c = c[(c["statistic"] == "calibrated") & (~c["size"].isin(["original"]))]
    n_all = pd.read_csv(DATA / "baseline_feature_ablation.csv")["tiles"].mean()
    c["n"] = c["size"].replace({"all": n_all}).astype(float)
    for model, mk in (("ridge", "o"), ("gradient boosting", "s")):
        g = c[c["model"] == model].sort_values("n")
        ax.plot(g["n"], g["mean"], mk + "-", color="#3a4047", ms=3.2, lw=1.0,
                mfc="white" if model == "ridge" else "#3a4047", label=model)
    ab = pd.read_csv(DATA / "baseline_feature_ablation.csv")
    pooled = ab[ab["features"] == "pooled features only"].set_index("model")["recall"]
    ax.axhline(full["class-averaged"], color=CA, lw=1.2, label="class-averaged score")
    ax.axhline(pooled["ridge"], color=PW, lw=1.0, ls=(0, (4, 2)),
               label="ridge, pooled features")
    ax.axhline(pooled["gradient boosting"], color=PW, lw=1.0, ls=(0, (1, 1.5)),
               label="boosting, pooled features")
    ax.axhline(0.2, color=RND, lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("Source tiles used for training")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=5.2, loc="upper left", handlelength=1.8, ncol=2,
              columnspacing=0.8, framealpha=0.95, borderaxespad=0.2)
    despine(ax)
    panel(ax, "c", dx=-0.22, dy=1.04)

    save(fig, "fig_support", FIG)
    print("written fig_support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
