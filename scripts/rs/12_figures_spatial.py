"""Study area, the screening workflow and the spatial layout of the review set."""
from __future__ import annotations

import os

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.patheffects import withStroke

try:
    import contextily as cx
except Exception:
    cx = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import BASE, CLASS_NAMES, FOLDS, load_units  # noqa: E402
from geoquery_als.rs.style import (CLASS_COLOUR, CLASS_LABEL,  # noqa: E402
                                   COL_TEXT, COL_WIDE, DOMAIN_COLOUR, INK,
                                   apply_style, despine, panel, save)

FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))
BUDGET = 0.20
OUTCOME = {"hit": "#1f6fb4", "miss": "#cf5a31", "false": "#e0b93c",
           "rest": "#e4e8eb"}


def scale_bar(ax, length_m, label, x=0.06, y=0.06):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = x0 + x * (x1 - x0)
    sy = y0 + y * (y1 - y0)
    ax.plot([sx, sx + length_m], [sy, sy], color=INK, lw=1.8,
            solid_capstyle="butt")
    ax.text(sx + length_m / 2, sy + 0.018 * (y1 - y0), label, ha="center",
            va="bottom", fontsize=6.0)


def north_arrow(ax, x=0.94, y=0.86):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.11),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=7.0, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK,
                                shrinkA=0, shrinkB=0))


def basemap(ax, source, zoom=None, alpha=1.0) -> bool:
    """Draw a web basemap under the axes, in the projected coordinate system."""
    if cx is None:
        return False
    try:
        cx.add_basemap(ax, crs="EPSG:2154", source=source, zoom=zoom,
                       alpha=alpha, attribution=False)
        return True
    except Exception:
        return False


def figure_study_area():
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    members = members[(members["family"] == "target") &
                      (members["scale_m"] == 2000)]
    units = {f: load_units(f) for f in FOLDS}

    fig = plt.figure(figsize=(COL_WIDE, COL_WIDE * 0.60))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.74], hspace=0.34,
                            wspace=0.18)

    ax = fig.add_subplot(grid[0, :3])
    pad = 70_000
    xs, ys = members["centroid_x"], members["centroid_y"]
    ax.set_xlim(xs.min() - pad * 3, xs.max() + pad)
    ax.set_ylim(ys.min() - pad * 2.4, ys.max() + pad * 2.6)
    ax.set_aspect("equal")
    basemap(ax, cx.providers.Esri.WorldGrayCanvas if cx else None, zoom=7)
    for f in FOLDS:
        m = members[members["fold"] == f]
        ax.plot(m["centroid_x"], m["centroid_y"], ".", ms=1.4,
                color=DOMAIN_COLOUR[f], mew=0)
        ax.annotate(f, xy=(m["centroid_x"].mean(), m["centroid_y"].max()),
                    xytext=(0, 9), textcoords="offset points", fontsize=8.0,
                    ha="center", color=DOMAIN_COLOUR[f], fontweight="bold",
                    path_effects=[withStroke(linewidth=2.0, foreground="white")])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    scale_bar(ax, 150_000, "150 km")
    north_arrow(ax)
    ax.text(0.02, 0.975, "EPSG:2154 (RGF93 / Lambert-93)",
            transform=ax.transAxes, fontsize=6.2, va="top",
            path_effects=[withStroke(linewidth=2.0, foreground="white")])
    panel(ax, "a", dx=-0.02, dy=1.02)

    ax = fig.add_subplot(grid[0, 3:])
    bottom = np.zeros(len(FOLDS))
    y = np.arange(len(FOLDS))[::-1]
    for c, name in enumerate(CLASS_NAMES):
        share = np.array([units[f].n_ref[:, c].sum() / units[f].n_ref.sum()
                          for f in FOLDS])
        ax.barh(y, share, left=bottom, height=0.58, color=CLASS_COLOUR[name],
                edgecolor="white", linewidth=0.4, label=CLASS_LABEL[name])
        bottom += share
    ax.set_yticks(y)
    ax.set_yticklabels(FOLDS, fontsize=7)
    for tick, f in zip(ax.get_yticklabels(), FOLDS):
        tick.set_color(DOMAIN_COLOUR[f])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of reference points", fontsize=7, labelpad=1.5)
    ax.grid(axis="y", visible=False)
    despine(ax)
    handles, labels = ax.get_legend_handles_labels()
    panel(ax, "b", dx=-0.14, dy=1.02)

    fig.legend(handles, labels, fontsize=6.2, ncol=7, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), handlelength=0.9,
               handletextpad=0.35, columnspacing=1.2, frameon=False)

    span = max(max(np.ptp(u.x), np.ptp(u.y)) for u in units.values()) + 12_000
    for k, f in enumerate(FOLDS):
        ax = fig.add_subplot(grid[1, k])
        u = units[f]
        mx, my = 0.5 * (u.x.min() + u.x.max()), 0.5 * (u.y.min() + u.y.max())
        ax.set_xlim(mx - span / 2, mx + span / 2)
        ax.set_ylim(my - span / 2, my + span / 2)
        ax.set_aspect("equal")
        basemap(ax, cx.providers.Esri.WorldShadedRelief if cx else None,
                zoom=9, alpha=0.85)
        ax.plot(u.x, u.y, "s", ms=1.3, color=DOMAIN_COLOUR[f], mew=0,
                alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{f}, {u.n_units} tiles", fontsize=6.8,
                     color=DOMAIN_COLOUR[f])
        scale_bar(ax, 20_000, "20 km", x=0.07, y=0.05)
        if k == 0:
            panel(ax, "c", dx=-0.06, dy=1.16)
    save(fig, "fig_study_area", FIG)


def figure_workflow():
    fig, ax = plt.subplots(figsize=(COL_WIDE, COL_WIDE * 0.30))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)

    labelled, unlabelled = "#dbe4ee", "#f3e3d7"
    boxes = [
        (1.5, "Source regions\nwith labels", labelled),
        (21.0, "Trained\nsegmentation model", labelled),
        (40.5, "New region,\nno labels", unlabelled),
        (60.0, "Per-class confidence\ninside each tile", unlabelled),
        (79.5, "Tiles ordered\nfor review", unlabelled),
    ]
    width, height, base = 17.5, 9.5, 12.0
    for x, text, colour in boxes:
        ax.add_patch(Rectangle((x, base), width, height, facecolor=colour,
                               edgecolor=INK, lw=0.6, zorder=2))
        ax.text(x + width / 2, base + height / 2, text, ha="center",
                va="center", fontsize=6.8, zorder=3)
    for x in boxes[:-1]:
        ax.add_patch(FancyArrowPatch((x[0] + width, base + height / 2),
                                     (x[0] + width + 2.0, base + height / 2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=0.9, color=INK, zorder=3))

    ax.add_patch(Rectangle((21.0, 1.5), width, 6.5, facecolor="white",
                           edgecolor=INK, lw=0.6, ls=(0, (2, 2)), zorder=2))
    ax.text(21.0 + width / 2, 4.75,
            "Recalibration map\nfitted on source labels",
            ha="center", va="center", fontsize=6.2, zorder=3)
    ax.add_patch(FancyArrowPatch((21.0 + width, 4.75), (60.0 + 1.5, base),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                 color="#6b7278", ls=(0, (2, 2)),
                                 connectionstyle="arc3,rad=-0.06", zorder=4))

    ax.add_patch(Rectangle((60.0, 24.5), width, 6.5, facecolor="white",
                           edgecolor=INK, lw=0.6, zorder=2))
    ax.text(60.0 + width / 2, 27.75,
            "Quality criterion\nsets the class weighting", ha="center",
            va="center", fontsize=6.2, zorder=3)
    ax.add_patch(FancyArrowPatch((60.0 + width / 2, 24.5),
                                 (60.0 + width / 2, base + height),
                                 arrowstyle="-|>", mutation_scale=7, lw=0.9,
                                 color=INK, zorder=3))
    ax.text(60.0 + width / 2 + 1.2, (24.5 + base + height) / 2,
            r"$\alpha=\beta$", fontsize=7.0, va="center", ha="left")

    ax.text(1.5, 32.5, "No reference label of the new region enters any score.",
            ha="left", va="top", fontsize=6.4, color="#5b6168")
    save(fig, "fig_workflow", FIG)


def figure_maps():
    units = {f: load_units(f) for f in FOLDS}
    span = max(max(np.ptp(u.x), np.ptp(u.y))
               for u in units.values()) / 1000 * 1.06
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDE, COL_WIDE * 0.30))
    for ax, f in zip(axes, FOLDS):
        u = units[f]
        mx = 0.5 * (u.x.min() + u.x.max()) / 1000
        my = 0.5 * (u.y.min() + u.y.max()) / 1000
        ax.set_xlim(mx - span / 2, mx + span / 2)
        ax.set_ylim(my - span / 2, my + span / 2)
        quality = cr.mean_iou(u.confusion, "union")
        score = sc.score(u, "calibrated", 0.0)
        k = mt.budget_size(u.n_units, BUDGET)
        worst = np.zeros(u.n_units, dtype=bool)
        worst[np.argsort(quality, kind="stable")[:k]] = True
        picked = np.zeros(u.n_units, dtype=bool)
        picked[np.argsort(score, kind="stable")[:k]] = True

        kinds = np.where(picked & worst, "hit",
                         np.where(picked, "false",
                                  np.where(worst, "miss", "rest")))
        for kind in ("rest", "false", "miss", "hit"):
            m = kinds == kind
            ax.plot(u.x[m] / 1000, u.y[m] / 1000, "s", ms=2.0,
                    color=OUTCOME[kind], mew=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f, fontsize=7.5, color=DOMAIN_COLOUR[f])
        ax.text(0.5, -0.05, f"recall {mt.recall_at(quality, score, BUDGET):.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=6.4)
        scale_bar(ax, 10, "10 km", x=0.06, y=0.03)
    north_arrow(axes[-1])
    handles = [plt.Line2D([], [], marker="s", ls="", ms=3.4,
                          color=OUTCOME[k]) for k in
               ("hit", "false", "miss", "rest")]
    fig.legend(handles,
               ["flagged and truly worst", "flagged, not worst",
                "worst, not flagged", "neither"], fontsize=6.2,
               loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4,
               handletextpad=0.35, columnspacing=1.4, frameon=False)
    save(fig, "fig_quality_maps", FIG)


def main() -> int:
    apply_style()
    figure_study_area()
    figure_workflow()
    figure_maps()
    print("spatial figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
