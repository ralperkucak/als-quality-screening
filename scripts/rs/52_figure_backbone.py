"""The exponent relation measured on both encoders.

Panel (a) shows, for each criterion exponent, the score exponent that screens
best and the interval that ties with it, drawn once per encoder. Panel (b) shows
what the two conventional criteria recover when the score is weighted to match
them and when it is weighted the other way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.style import (COL_WIDE, apply_style,  # noqa: E402
                                   despine, panel, save)

DATA = ROOT / "outputs" / "rs"
FIG = ROOT / "manuscript_rs" / "figures"

ENCODER = [("randlanet", "RandLA-Net", "#1f6fb4"),
           ("point_transformer", "Point Transformer", "#cf5a31")]


def main() -> int:
    summary_path = DATA / "backbone_summary.csv"
    if not summary_path.exists():
        print("no backbone summary yet, nothing to draw")
        return 1
    summary = pd.read_csv(summary_path).set_index("backbone")

    apply_style()
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(COL_WIDE, COL_WIDE * 0.42),
        gridspec_kw={"width_ratios": [1.28, 1.0], "wspace": 0.34})

    ax.plot([-0.5, 1.5], [-0.5, 1.5], color="#9aa3ad", lw=0.9,
            ls=(0, (3, 2)), zorder=1)
    ax.text(1.42, 1.34, r"$\alpha=\beta$", color="#6b747e", fontsize=7.5,
            ha="right", va="top", rotation=38)
    for key, label, colour in ENCODER:
        ridge = pd.read_csv(DATA / f"backbone_ridge_{key}.csv")
        ax.fill_between(ridge["beta"], ridge["alpha_lo"], ridge["alpha_hi"],
                        color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(ridge["beta"], ridge["alpha_best"], color=colour, lw=1.7,
                label=label, zorder=3)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlabel(r"criterion exponent $\beta$")
    ax.set_ylabel(r"best score exponent $\alpha^{\ast}$")
    ax.axvspan(-0.5, 0.0, color="#f2f4f6", lw=0, zorder=0)
    ax.axvspan(1.0, 1.5, color="#f2f4f6", lw=0, zorder=0)
    ax.legend(loc="upper left", frameon=False, handlelength=1.4,
              borderaxespad=0.2)
    despine(ax)
    panel(ax, "a")

    # Matched against mismatched weighting, as a slope between two positions on
    # one axis rather than as four separated bars.
    criteria = [("class_averaged", "mean IoU\n" r"($\beta=0$)"),
                ("point_weighted", "overall accuracy\n" r"($\beta=1$)")]
    offset = {"randlanet": -0.115, "point_transformer": 0.115}
    for key, label, colour in ENCODER:
        row = summary.loc[label]
        for j, (name, _) in enumerate(criteria):
            x = j + offset[key]
            hi = float(row[f"lift_{name}"])
            lo = float(row[f"lift_{name}_mismatched"])
            bx.plot([x, x], [lo, hi], color=colour, lw=1.1, alpha=0.55,
                    zorder=2, solid_capstyle="round")
            bx.scatter([x], [hi], s=30, color=colour, zorder=4)
            bx.scatter([x], [lo], s=30, facecolor="white", edgecolor=colour,
                       linewidth=1.0, zorder=4)
            bx.annotate(f"{hi:.2f}", (x, hi), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=colour)
            bx.annotate(f"{lo:.2f}", (x, lo), textcoords="offset points",
                        xytext=(0, -11), ha="center", fontsize=7, color=colour)
    bx.axhline(0.0, color="#9aa3ad", lw=0.8, zorder=1)
    bx.set_xticks(range(len(criteria)))
    bx.set_xticklabels([c[1] for c in criteria])
    bx.set_xlim(-0.55, len(criteria) - 0.45)
    bx.set_ylabel("screening lift")
    bx.set_ylim(-0.14, 0.80)
    despine(bx)
    panel(bx, "b")

    save(fig, "fig_backbone", FIG)
    print("wrote fig_backbone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
