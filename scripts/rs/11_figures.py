"""Figures for the regional distributions, the review budget and the per-class result."""
from __future__ import annotations

import os

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import CLASS_NAMES, FOLDS, load_units  # noqa: E402
from geoquery_als.rs.style import (CLASS_COLOUR, CLASS_LABEL,  # noqa: E402
                                   COL_TEXT, COL_WIDE, DOMAIN_COLOUR,
                                   FAMILY_COLOUR, apply_style, despine, panel,
                                   save)

DATA = ROOT / "outputs" / "rs"
FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))
BUDGET = 0.20
COVERAGE = np.round(np.arange(0.05, 0.501, 0.0125), 4)
FAMILIES = {"class averaged": ("calibrated", 0.0),
            "point weighted": ("calibrated", 1.0)}


def figure_distribution(units):
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDE, COL_WIDE * 0.34),
                             sharey=True)
    for ax, f in zip(axes, FOLDS):
        q = cr.mean_iou(units[f].confusion, "union")
        parts = ax.violinplot([q], positions=[0], widths=0.85,
                              showextrema=False, showmedians=False,
                              bw_method=0.25)
        for body in parts["bodies"]:
            vertices = body.get_paths()[0].vertices
            vertices[:, 1] = np.clip(vertices[:, 1], q.min(), q.max())
        for body in parts["bodies"]:
            body.set_facecolor(DOMAIN_COLOUR[f])
            body.set_alpha(0.30)
            body.set_edgecolor(DOMAIN_COLOUR[f])
            body.set_linewidth(0.7)
        jitter = np.random.default_rng(3).normal(0, 0.055, len(q))
        ax.plot(jitter, q, ".", ms=1.6, color=DOMAIN_COLOUR[f], alpha=0.55,
                mew=0)
        pooled = cr.mean_iou(units[f].confusion.sum(axis=0, keepdims=True),
                             "union")[0]
        ax.plot([-0.42, 0.42], [q.mean()] * 2, color="#15181c", lw=1.2)
        ax.plot([-0.42, 0.42], [pooled] * 2, color="#15181c", lw=1.2,
                ls=(0, (3, 2)))
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_title(f, fontsize=7.5, color=DOMAIN_COLOUR[f])
        ax.grid(axis="x", visible=False)
        despine(ax, ("top", "right", "bottom"))
    axes[0].set_ylabel("Tile mean IoU")
    axes[0].set_ylim(0, 1)
    handles = [plt.Line2D([], [], color="#15181c", lw=1.2),
               plt.Line2D([], [], color="#15181c", lw=1.2, ls=(0, (3, 2)))]
    axes[-1].legend(handles, ["mean over tiles", "pooled over the region"],
                    fontsize=6.2, loc="lower right")
    save(fig, "fig_tile_quality", FIG)


def figure_coverage(units):
    quality = {f: cr.mean_iou(units[f].confusion, "union") for f in FOLDS}
    deficit = {f: mt.deficit_from(quality[f]) for f in FOLDS}
    fig, axes = plt.subplots(1, 5, figsize=(COL_WIDE, COL_WIDE * 0.34),
                             sharey=True)
    for ax, f in zip(axes, FOLDS):
        ax.axhline(1.0, color="#c6ccd2", lw=0.8)
        for name, (statistic, alpha) in FAMILIES.items():
            s = sc.score(units[f], statistic, alpha)
            curve = np.array([(mt.recall_at(quality[f], s, c) - c) / (1 - c)
                              for c in COVERAGE])
            key = "class-averaged" if name == "class averaged" else "point-weighted"
            ax.plot(COVERAGE, curve, color=FAMILY_COLOUR[key], lw=1.4,
                    label=name)
            ax.text(0.97, 0.20 if alpha == 0.0 else 0.08,
                    f"{mt.normalised_aurc(deficit[f], s):.2f}",
                    transform=ax.transAxes, fontsize=6.4,
                    color=FAMILY_COLOUR[key], ha="right")
        ax.axhline(0.0, color=FAMILY_COLOUR["random"], lw=0.8, ls=(0, (1, 2)))
        ax.set_title(f, fontsize=7.5, color=DOMAIN_COLOUR[f])
        ax.set_xlabel("Review budget")
        ax.set_xlim(0.05, 0.5)
        ax.set_ylim(-0.15, 1.08)
        despine(ax)
    axes[0].set_ylabel("Screening lift")
    axes[0].text(0.03, 0.20, "NAURC", transform=axes[0].transAxes, fontsize=6.4,
                 color="#15181c")
    axes[-1].text(0.97, 0.955, "perfect ordering", transform=axes[-1].transAxes,
                  fontsize=6.2, color="#6b7278", ha="right", va="top")
    axes[0].legend(fontsize=6.2, loc="upper left")
    save(fig, "fig_review_budget", FIG)


def figure_per_class():
    d = pd.read_csv(DATA / "per_class_screening.csv")
    d = d[(d["minimum_points"] == 500)
          & (d["eligibility"] == "predicted")]
    own = d.pivot(index="class", columns="region", values="spearman_own_class")
    other = d.pivot(index="class", columns="region",
                    values="spearman_class_balanced")
    order = [c for c in CLASS_NAMES if c in own.index]
    own, other = own.reindex(order), other.reindex(order)

    fig, ax = plt.subplots(figsize=(COL_TEXT, COL_TEXT * 0.62))
    ax.grid(False)
    mesh = ax.pcolormesh(np.arange(len(own.columns) + 1),
                         np.arange(len(order) + 1),
                         np.ma.masked_invalid(own.to_numpy()),
                         cmap="viridis", vmin=0.5, vmax=1.0, shading="flat")
    for i in range(len(order)):
        for j in range(len(own.columns)):
            a, b = own.to_numpy()[i, j], other.to_numpy()[i, j]
            if not np.isfinite(a):
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center",
                        fontsize=7, color="#8a8f94")
                continue
            ax.text(j + 0.5, i + 0.36, f"{a:.2f}", ha="center", va="center",
                    fontsize=7.6, color="white" if a < 0.82 else "#15181c")
            ax.text(j + 0.5, i + 0.68, f"({b:.2f})", ha="center", va="center",
                    fontsize=6.6, color="#dfe4e8" if a < 0.82 else "#3a4047")
    ax.set_xticks(np.arange(len(own.columns)) + 0.5)
    ax.set_xticklabels(own.columns, fontsize=7.8)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([CLASS_LABEL[c] for c in order], fontsize=7.8)
    ax.invert_yaxis()
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.045, aspect=20)
    bar.set_label("Rank correlation, own-class confidence", fontsize=6.8)
    bar.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Held-out region")
    ax.tick_params(length=0)
    save(fig, "fig_per_class", FIG)


def figure_decomposition(units):
    d = pd.read_csv(DATA / "regions_per_class.csv")
    # The region shown is the one where the effect being illustrated is
    # largest, so the choice is a stated rule rather than an inspection.
    region = d.groupby("region")["covariance_term"].sum().idxmax()
    g = d[d["region"] == region].set_index("class")
    order = [c for c in CLASS_NAMES if c in g.index]

    fig, ax = plt.subplots(figsize=(COL_TEXT, COL_TEXT * 0.55))
    for i, c in enumerate(order):
        r = g.loc[c]
        ax.plot([r["mean_tile_iou"], r["pooled_iou"]], [i, i],
                color=CLASS_COLOUR[c],
                lw=1.0 + 6.0 * abs(r["covariance_term"]), alpha=0.65,
                solid_capstyle="round")
        ax.plot(r["mean_tile_iou"], i, "o", ms=4.0, color=CLASS_COLOUR[c],
                mec="white", mew=0.6)
        ax.plot(r["pooled_iou"], i, "D", ms=3.6, color=CLASS_COLOUR[c],
                mec="white", mew=0.6)
        ax.text(1.02, i, f"{int(r['tiles_present'])}", fontsize=6.2,
                va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([CLASS_LABEL[c] for c in order], fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Intersection over union in {region}")
    ax.text(1.02, -0.9, "Tiles", fontsize=6.2, va="center",
            transform=ax.get_yaxis_transform())
    ax.grid(axis="y", visible=False)
    despine(ax)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.0, color="#5b6168"),
               plt.Line2D([], [], marker="D", ls="", ms=3.6, color="#5b6168")]
    ax.legend(handles, ["mean over tiles", "pooled over the region"],
              fontsize=6.4, loc="lower left")
    save(fig, "fig_decomposition", FIG)


def main() -> int:
    apply_style()
    units = {f: load_units(f) for f in FOLDS}
    figure_distribution(units)
    figure_coverage(units)
    figure_per_class()
    figure_decomposition(units)
    print("figures written to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
