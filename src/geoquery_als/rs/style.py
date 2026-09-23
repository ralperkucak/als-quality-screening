"""Figure style for the Remote Sensing layout.

Figure widths follow the text block of the journal rather than a generic page,
so a figure is drawn at the size it will be printed and nothing is rescaled by
the typesetter. Colour is carried by a palette that keeps its ordering when
converted to greyscale, and the seven semantic classes keep one colour and one
order everywhere they appear.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Text block of the Remote Sensing submission layout, in inches.
COL_TEXT = 13.86 / 2.54      # 5.457, a figure inside the text block
COL_WIDE = 18.46 / 2.54      # 7.268, a figure spanning the margin as well
COL_HALF = COL_TEXT / 2

CLASS_ORDER = ("other", "ground", "vegetation", "building", "water", "bridge",
               "permanent_structure")
CLASS_LABEL = {
    "other": "Other", "ground": "Ground", "vegetation": "Vegetation",
    "building": "Building", "water": "Water", "bridge": "Bridge",
    "permanent_structure": "Permanent structure",
}
# Hues follow the usual reading of the classes and the luminances are spread
# from 0.24 to 0.84 with no pair closer than 0.09, so the seven classes stay
# separable in greyscale print.
CLASS_COLOUR = {
    "other": "#d3d8dd",
    "ground": "#c9975a",
    "vegetation": "#57993f",
    "building": "#b83b2c",
    "water": "#1a4466",
    "bridge": "#8a5cb4",
    "permanent_structure": "#e0b93c",
}
DOMAIN_COLOUR = {"D1": "#1f6fb4", "D2": "#1a9464", "D3": "#d99114",
                 "D4": "#cf5a31", "D5": "#7a4fa3"}

# Two families of aggregation, kept apart by hue and by lightness.
FAMILY_COLOUR = {"class-averaged": "#1f6fb4", "point-weighted": "#cf5a31",
                 "random": "#8a8f94", "oracle": "#20242a"}

SEQUENTIAL = "viridis"
DIVERGING = "RdBu_r"

INK = "#1b1f24"
GRID = "#d7dbe0"
FAINT = "#eef1f4"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "text.color": INK,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel(ax, letter: str, dx: float = -0.10, dy: float = 1.04) -> None:
    """Panel letter in the bold lower case form the journal uses."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


def despine(ax, sides=("top", "right")) -> None:
    for s in sides:
        ax.spines[s].set_visible(False)


def save(fig, name: str, directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}")
    plt.close(fig)


def greyscale_safe(colours) -> np.ndarray:
    """Relative luminance of a colour list, to check greyscale separation."""
    rgb = np.array([mpl.colors.to_rgb(c) for c in colours])
    return rgb @ np.array([0.2126, 0.7152, 0.0722])
