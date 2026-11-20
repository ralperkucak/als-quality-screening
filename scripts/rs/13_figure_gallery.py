"""What a flagged tile looks like to the operator who opens it.

Four tiles of one region, one for each outcome of the screening, each shown as
the reference labelling of its largest patch and as the points where the
prediction disagrees with it. The tiles are selected from the cached scenes by
the outcome they receive under the score reported in the paper, so the panels
show the procedure working and failing on the same terms.
"""
from __future__ import annotations

import os

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs import criteria as cr  # noqa: E402
from geoquery_als.rs import metrics as mt  # noqa: E402
from geoquery_als.rs import scores as sc  # noqa: E402
from geoquery_als.rs.store import BASE, CLASS_NAMES, FOLDS, load_units  # noqa: E402
from geoquery_als.rs.style import (CLASS_COLOUR, CLASS_LABEL,  # noqa: E402
                                   COL_WIDE, INK, apply_style, panel, save)

FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))
CLOUDS = BASE / "pointclouds"
REGION = "D4"
BUDGET = 0.20
CELL = 0.40
ORDER = ("recovered", "selected only", "missed", "clean")
TITLE = {"recovered": "flagged, truly worst",
         "selected only": "flagged, not worst",
         "missed": "worst, not flagged",
         "clean": "neither"}


def raster(x, y, value, cell=CELL, reducer="max"):
    """Highest return per cell of a regular grid, returned as an image."""
    xi = np.floor((x - x.min()) / cell).astype(np.int64)
    yi = np.floor((y - y.min()) / cell).astype(np.int64)
    w, h = xi.max() + 1, yi.max() + 1
    flat = yi * w + xi
    out = np.full(w * h, np.nan)
    order = np.argsort(value if reducer == "max" else -value)
    out[flat[order]] = value[order]
    return out.reshape(h, w)


def paint(ax, cloud, mode):
    x, y, z = cloud["xy"][:, 0], cloud["xy"][:, 1], cloud["z"]
    keep = cloud["patch_of"] == np.bincount(
        cloud["patch_of"].astype(np.int64)).argmax()
    x, y, z = x[keep], y[keep], z[keep]
    reference = cloud["reference"][keep].astype(np.int64)
    prediction = cloud["prediction"][keep].astype(np.int64)

    top = np.argsort(z)
    xi = np.floor((x - x.min()) / CELL).astype(np.int64)
    yi = np.floor((y - y.min()) / CELL).astype(np.int64)
    w, h = xi.max() + 1, yi.max() + 1
    flat = (yi * w + xi)[top]

    height = np.full(w * h, np.nan)
    height[flat] = z[top]
    shade = hillshade(height.reshape(h, w))

    if mode == "reference":
        label = np.full(w * h, -1, dtype=np.int64)
        label[flat] = reference[top]
        image = np.zeros((h * w, 3))
        for c, name in enumerate(CLASS_NAMES):
            image[label == c] = plt.matplotlib.colors.to_rgb(CLASS_COLOUR[name])
        image[label < 0] = 0.99
        image = image.reshape(h, w, 3) * shade[..., None]
    else:
        wrong = np.full(w * h, np.nan)
        wrong[flat] = (reference[top] != prediction[top]).astype(float)
        base = np.repeat(shade[..., None], 3, axis=2) * 0.34 + 0.63
        mask = np.nan_to_num(wrong.reshape(h, w)) > 0.5
        image = base.copy()
        image[mask] = np.array([0.76, 0.09, 0.36])
        image[np.isnan(wrong.reshape(h, w))] = 1.0
    ax.imshow(np.clip(image, 0, 1), origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_linewidth(0.5)


def hillshade(height, azimuth=315.0, altitude=45.0):
    filled = np.where(np.isnan(height), np.nanmin(height), height)
    dy, dx = np.gradient(filled, CELL)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip(0.55 + 0.45 * shaded, 0, 1)


def classify(region: str):
    units = load_units(region)
    quality = cr.mean_iou(units.confusion, "union")
    score = sc.score(units, "calibrated", 0.0)
    k = mt.budget_size(units.n_units, BUDGET)
    worst = set(np.argsort(quality, kind="stable")[:k].tolist())
    picked = set(np.argsort(score, kind="stable")[:k].tolist())
    labels = {}
    for i, uid in enumerate(units.unit_id):
        if i in picked and i in worst:
            labels[uid] = "recovered"
        elif i in picked:
            labels[uid] = "selected only"
        elif i in worst:
            labels[uid] = "missed"
        else:
            labels[uid] = "clean"
    return labels, dict(zip(units.unit_id, quality))


def main() -> int:
    apply_style()
    labels, quality = classify(REGION)
    available = {p.stem: p for p in CLOUDS.glob(f"{REGION}_*.npz")}
    chosen = {}
    # Sorted so the choice is reproducible rather than filesystem order, and
    # taken without looking at how well each candidate illustrates the point.
    for uid, path in sorted(available.items()):
        kind = labels.get(uid)
        if kind in ORDER and kind not in chosen:
            chosen[kind] = (uid, path)
    missing = [k for k in ORDER if k not in chosen]
    if missing:
        print("no cached scene for:", missing)

    present = [k for k in ORDER if k in chosen]
    fig, axes = plt.subplots(2, len(present),
                             figsize=(COL_WIDE, COL_WIDE * 0.56))
    axes = np.atleast_2d(axes)
    for column, kind in enumerate(present):
        uid, path = chosen[kind]
        cloud = np.load(path, allow_pickle=True)
        paint(axes[0, column], cloud, "reference")
        paint(axes[1, column], cloud, "disagreement")
        axes[0, column].set_title(
            f"{TITLE[kind]}\nunit mIoU {quality[uid]:.2f}", fontsize=6.8)
    axes[0, 0].set_ylabel("Reference", fontsize=7.2)
    axes[1, 0].set_ylabel("Disagreement", fontsize=7.2)
    for row, letter in zip(axes[:, 0], "ab"):
        panel(row, letter, dx=-0.13, dy=1.02)

    drawn = sorted({int(c) for _, path in chosen.values()
                    for c in np.unique(np.load(path, allow_pickle=True)
                                       ["reference"])})
    shown = [CLASS_NAMES[c] for c in drawn if 0 <= c < len(CLASS_NAMES)]
    handles = [plt.Line2D([], [], marker="s", ls="", ms=3.6,
                          color=CLASS_COLOUR[c]) for c in shown]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=3.6,
                              color="#c2185b"))
    fig.legend(handles, [CLASS_LABEL[c] for c in shown] + ["Disagreement"],
               fontsize=6.2, ncol=len(shown) + 1, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.35,
               columnspacing=1.2, frameon=False)
    save(fig, "fig_gallery", FIG)
    print("gallery written for", REGION, list(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
