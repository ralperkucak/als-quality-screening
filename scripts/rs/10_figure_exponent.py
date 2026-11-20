"""Centrepiece figure: the response surface, the exponents that tie with the best,
and what each control leaves of the screening."""
from __future__ import annotations

import os

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patheffects import withStroke

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.style import (COL_WIDE, DOMAIN_COLOUR,  # noqa: E402
                                   FAMILY_COLOUR, apply_style, despine, panel,
                                   save)

DATA = ROOT / "outputs" / "rs"
FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))
STATISTIC = "calibrated"
FOLDS = ("D1", "D2", "D3", "D4", "D5")


def main() -> int:
    apply_style()
    surface = pd.read_parquet(DATA / "exponent_surface.parquet")
    surface = surface[surface["statistic"] == STATISTIC]
    ridge = pd.read_csv(DATA / "exponent_ridge.csv")
    ridge = ridge[ridge["statistic"] == STATISTIC]
    summary = pd.read_csv(DATA / "exponent_summary.csv")
    summary = summary[summary["statistic"] == STATISTIC].iloc[0]
    ladder = pd.read_csv(DATA / "control_ladder.csv")

    grid = (surface.groupby(["alpha", "beta"])["lift"].mean()
            .unstack("beta").sort_index())
    alphas, betas = grid.index.to_numpy(), grid.columns.to_numpy()

    fig = plt.figure(figsize=(COL_WIDE, COL_WIDE * 0.60))
    gs = GridSpec(2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.0, 0.92],
                  wspace=0.58, hspace=0.55, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.grid(False)
    mesh = ax.pcolormesh(betas, alphas, grid.to_numpy(), cmap="viridis",
                         shading="nearest", vmin=0.0, vmax=0.70, rasterized=True)
    ax.contour(betas, alphas, grid.to_numpy(), levels=[0.2, 0.4, 0.6],
               colors="white", linewidths=0.45, alpha=0.7)
    ax.fill_between(ridge["beta"], ridge["alpha_lo"], ridge["alpha_hi"],
                    color="white", alpha=0.30, lw=0)
    ax.plot(ridge["beta"], ridge["alpha_best"], color="white", lw=2.0)
    ax.plot(ridge["beta"], ridge["alpha_best"], color="#15181c", lw=1.0,
            label="best score exponent")
    ax.plot([-0.5, 1.5], [-0.5, 1.5], color="white", lw=1.5, ls=(0, (3, 2)))
    ax.plot([-0.5, 1.5], [-0.5, 1.5], color="#15181c", lw=0.8, ls=(0, (3, 2)),
            label=r"$\alpha=\beta$")
    for (b, a, matched) in ((0.0, 0.0, True), (1.0, 1.0, True),
                            (0.0, 1.0, False), (1.0, 0.0, False)):
        i = int(np.argmin(np.abs(alphas - a)))
        j = int(np.argmin(np.abs(betas - b)))
        value = grid.to_numpy()[i, j]
        if matched:
            ax.plot(b, a, "*", ms=13.0, mfc="white", mec="#15181c", mew=0.9,
                    zorder=6)
        else:
            ax.plot(b, a, "X", ms=7.0, mfc="#f2f2f2", mec="#15181c", mew=0.7,
                    zorder=6)
        ax.annotate(f"{value:.2f}", xy=(b, a), xytext=(7, 5),
                    textcoords="offset points", fontsize=7.6, color="white",
                    fontweight="bold",
                    path_effects=[withStroke(linewidth=1.8,
                                             foreground="#15181c")])
    halo = [withStroke(linewidth=1.6, foreground="#15181c")]
    ax.annotate(r"class-averaged, $\alpha=\beta=0$", xy=(0.0, 0.0),
                xytext=(-0.45, -0.80), fontsize=6.8, color="white",
                path_effects=halo,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7))
    ax.annotate(r"point-weighted, $\alpha=\beta=1$", xy=(1.0, 1.0),
                xytext=(-0.42, 1.55), fontsize=6.8, color="white",
                path_effects=halo,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.7))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlabel(r"Criterion class weighting $\beta$")
    ax.set_ylabel(r"Score class weighting $\alpha$")
    ax.legend(loc="upper left", fontsize=6.6, labelcolor="white")
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.042, aspect=22)
    bar.set_label("Screening lift at a 20% budget", fontsize=6.8)
    bar.ax.tick_params(labelsize=6.5)
    panel(ax, "a", dx=-0.21, dy=1.03)

    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(ridge["beta"], ridge["alpha_lo"], ridge["alpha_hi"],
                    color="#c7d3e0", alpha=0.8, lw=0,
                    label="ties with the best")
    for f in FOLDS:
        ax.plot(ridge["beta"], ridge[f], color=DOMAIN_COLOUR[f], lw=0.6,
                alpha=0.7)
    ax.plot(ridge["beta"], ridge["alpha_best"], color="#15181c", lw=1.7,
            label="best exponent")
    ax.plot([-0.5, 1.5], [-0.5, 1.5], color="#8a8f94", lw=0.8, ls=(0, (3, 2)))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlabel(r"Criterion class weighting $\beta$")
    ax.set_ylabel(r"Score exponent $\alpha^{\ast}$")
    despine(ax)
    ax.text(0.03, 0.96,
            rf"$\rho={summary['monotone_rho']:.2f}$ "
            rf"[{summary['monotone_lo']:.2f}, {summary['monotone_hi']:.2f}]",
            transform=ax.transAxes, fontsize=7.4, va="top")
    ax.legend(fontsize=6.0, loc="lower right", handlelength=1.2,
              handletextpad=0.4)
    panel(ax, "b", dx=-0.20, dy=1.02)

    ax = fig.add_subplot(gs[1, 1])
    order = ["matched exponents", "mismatched exponents",
             "reference class weights", "composition only",
             "class count only"]
    label = {"matched exponents": "matched",
             "mismatched exponents": "mismatched",
             "reference class weights": "reference weights",
             "composition only": "composition only",
             "class count only": "class count only"}
    offset = {"class averaged": -0.16, "point weighted": 0.16}
    colour = {"class averaged": FAMILY_COLOUR["class-averaged"],
              "point weighted": FAMILY_COLOUR["point-weighted"]}
    for criterion, group in ladder.groupby("criterion"):
        g = group.set_index("control").reindex(order)
        y = np.arange(len(order)) + offset[criterion]
        ax.errorbar(g["lift"], y,
                    xerr=[g["lift"] - g["lo"], g["hi"] - g["lift"]],
                    fmt="o", ms=3.2, lw=0.9, capsize=1.6,
                    color=colour[criterion], label=criterion)
    ax.axvline(0.0, color="#8a8f94", lw=0.7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([label[o] for o in order], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("Screening lift")
    ax.set_xlim(-0.25, 1.05)
    ax.grid(axis="y", visible=False)
    despine(ax)
    ax.legend(fontsize=6.2, loc="lower right", handletextpad=0.3,
              bbox_to_anchor=(1.0, -0.04))
    panel(ax, "c", dx=-0.26, dy=1.08)

    save(fig, "fig_exponent_surface", FIG)
    print("monotone rho %.3f [%.2f, %.2f]"
          % (summary["monotone_rho"], summary["monotone_lo"],
             summary["monotone_hi"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
