"""Architecture of the segmentation network and of its local aggregation unit.

Drawn from the configuration of `geoquery_als.models.randlanet` so the channel
widths and point counts in the figure are the ones the experiments used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from geoquery_als.rs.style import COL_WIDE, INK, apply_style, save  # noqa: E402

FIG = Path(os.environ.get("RS_FIGDIR", ROOT / "manuscript_rs" / "figures"))

ENC = "#dbe7f3"
DEC = "#e4f1e1"
HEAD = "#f6e2d6"
NEUTRAL = "#eef1f4"
EDGE = "#4a525a"


def box(ax, x, y, w, h, text, colour, fs=8.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=colour, edgecolor=EDGE, lw=0.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)


def arrow(ax, a, b, colour=INK, ls="-", rad=0.0, lw=0.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6.5,
                                 lw=lw, color=colour, ls=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))


def network(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.set_axis_off()
    points = [12288, 3072, 768, 192, 48]
    enc_c = [32, 128, 256, 512]
    dec_c = [256, 128, 32, 8]
    w, h = 10.0, 6.8
    top, bottom = 31.5, 6.0

    box(ax, 0.5, top, 9.5, h, "Input\n$N\\times 11$\nxyz + 8 features", NEUTRAL, fs=7.6)
    box(ax, 12.0, top, 7.5, h, "Shared\nMLP\n8", ENC, fs=7.6)
    arrow(ax, (10.0, top + h / 2), (12.0, top + h / 2))

    xs = [22.0 + i * 13.0 for i in range(4)]
    prev = (19.5, top + h / 2)
    for i, x in enumerate(xs):
        y = top - i * 5.2
        box(ax, x, y, w, h, f"LFA $\\times$2\n{enc_c[i]} ch\nsample $\\downarrow$4", ENC, fs=7.6)
        arrow(ax, prev, (x, y + h / 2))
        ax.text(x + w / 2, y + h + 0.8, f"{points[i]:,} $\\to$ {points[i + 1]:,} pts",
                ha="center", fontsize=6.8, color="#5b6168")
        prev = (x + w, y + h / 2)

    bx = 74.0
    box(ax, bx + 2.0, top - 3 * 5.2 - 9.5, 10.5, 5.6, "MLP\n512", ENC, fs=7.6)
    arrow(ax, (xs[-1] + w / 2, top - 3 * 5.2), (bx + 7.25, top - 3 * 5.2 - 3.9))

    dxs = [60.0, 47.0, 34.0, 21.0]
    prev = (bx + 2.0, top - 3 * 5.2 - 6.7)
    for i, x in enumerate(dxs):
        y = bottom + i * 0.0
        box(ax, x, y, w, h, f"NN up $\\uparrow$4\n+ skip\nMLP {dec_c[i]}", DEC, fs=7.6)
        arrow(ax, prev, (x + w, y + h / 2))
        prev = (x, y + h / 2)
    for i in range(3):
        sx = xs[2 - i] + w / 2
        sy = top - (2 - i) * 5.2
        arrow(ax, (sx, sy), (dxs[i] + w / 2, bottom + h), colour="#8a929a",
              ls=(0, (2, 1.5)), lw=0.6)
    arrow(ax, (12.0 + 3.75, top), (dxs[3] + w / 2, bottom + h), colour="#8a929a",
          ls=(0, (2, 1.5)), lw=0.6)

    box(ax, 6.0, bottom, 12.0, h, "MLP 64, 32\ndropout 0.5\nlinear 7", HEAD, fs=7.6)
    arrow(ax, (dxs[3], bottom + h / 2), (18.0, bottom + h / 2))
    ax.text(12.0, bottom - 2.2, "softmax $\\to p_{ic}$,  $q_i=\\max_c p_{ic}$",
            ha="center", fontsize=7.4)
    ax.text(4.0, 44.0, "Encoder", fontsize=8.4, color="#1f4f7a", fontweight="bold")
    ax.text(4.0, 1.0, "Decoder and classifier", fontsize=8.4, color="#2f6b2a",
            fontweight="bold")
    ax.text(99.5, 1.0, "dashed: skip connections", ha="right", fontsize=6.8,
            color="#5b6168")


def lfa(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.set_axis_off()
    box(ax, 1.0, 30.0, 18.0, 8.0, "point $p_i$, feature $f_i$\nneighbors $\\mathcal{N}_i$\n(16 nearest)",
        NEUTRAL, fs=7.4)
    box(ax, 24.0, 33.5, 24.0, 8.0, "local spatial encoding\n$\\eta(p_i\\oplus p_j\\oplus(p_i-p_j)\\oplus\\|p_i-p_j\\|)$",
        ENC, fs=7.0)
    box(ax, 24.0, 22.5, 24.0, 7.0, "concatenate with $f_j$\n$\\hat f_{ij}$", ENC, fs=7.4)
    box(ax, 53.0, 26.0, 22.0, 10.0, "attentive pooling\n$\\sum_j \\mathrm{softmax}_j(W\\hat f_{ij})\\odot \\hat f_{ij}$",
        ENC, fs=7.0)
    box(ax, 80.0, 27.5, 18.0, 7.0, "MLP $\\xi$ $\\to y_i$", ENC, fs=7.6)
    arrow(ax, (19.0, 34.0), (24.0, 37.5))
    arrow(ax, (36.0, 33.5), (36.0, 29.5))
    arrow(ax, (48.0, 26.0), (53.0, 31.0))
    arrow(ax, (75.0, 31.0), (80.0, 31.0))
    ax.text(50.0, 43.5, "One unit, used twice per residual block (Equation (1))",
            ha="center", fontsize=7.2, color="#5b6168")

    box(ax, 24.0, 4.0, 51.0, 11.0,
        "second encoder: vector self-attention replaces attentive pooling\n"
        "$y_i=\\sum_j \\mathrm{softmax}_j(\\gamma(W_q f_i - W_k f_j+\\delta_{ij}))\\odot(W_v f_j+\\delta_{ij})$",
        "#efe6f6", fs=7.0)
    ax.text(50.0, 17.5, "Equation (2); all other layers unchanged", ha="center",
            fontsize=7.0, color="#5b6168")


def main() -> int:
    apply_style()
    fig = plt.figure(figsize=(COL_WIDE * 1.12, COL_WIDE * 0.78))
    ax1 = fig.add_axes([0.0, 0.40, 1.0, 0.60])
    ax2 = fig.add_axes([0.0, 0.0, 1.0, 0.40])
    network(ax1)
    lfa(ax2)
    fig.text(0.005, 0.985, "(a)", fontsize=8, fontweight="bold", va="top")
    fig.text(0.005, 0.39, "(b)", fontsize=8, fontweight="bold", va="top")
    save(fig, "fig_architecture", FIG)
    print("architecture figure written to", FIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
