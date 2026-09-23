"""Screening lift over the score exponent and the criterion exponent.

The grid this module builds is the object the main experiment reports and the
object the architecture check repeats on a second encoder. Both call the same
functions, so a difference between the two encoders cannot come from a
difference between two implementations of the same summary.

The lift is measured against the budget the ranking can actually spend. A review
budget of a fifth of the tiles is realised as the ceiling of a fifth of however
many tiles a region holds, which is slightly more than a fifth, and the share
that a random ordering would recover has to be the realised share rather than
the nominal one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import criteria as cr
from . import metrics as mt
from . import scores as sc

__all__ = ["ALPHAS", "BETAS", "BUDGET", "ENDPOINTS",
           "lift", "surface", "bands", "monotone_strength"]

ALPHAS = np.round(np.arange(-1.0, 2.001, 0.05), 3)
BETAS = np.round(np.arange(-0.5, 1.501, 0.05), 3)
BUDGET = 0.20
ENDPOINTS = (0.0, 1.0)


def lift(recall: float, q: float) -> float:
    """Share of the improvement over random review that the ordering realises."""
    return (recall - q) / (1.0 - q)


def surface(units, statistic: str, backbone: str = "randlanet") -> np.ndarray:
    """Lift at every pair of exponents, for one region and one statistic."""
    quality = {b: cr.quality(units.confusion, b) for b in BETAS}
    score = {a: sc.score(units, statistic, a, backbone=backbone) for a in ALPHAS}
    q = mt.budget_size(units.n_units, BUDGET) / units.n_units
    out = np.empty((len(ALPHAS), len(BETAS)))
    for i, a in enumerate(ALPHAS):
        for j, b in enumerate(BETAS):
            out[i, j] = lift(mt.recall_at(quality[b], score[a], BUDGET), q)
    return out


def bands(grids) -> pd.DataFrame:
    """Best score exponent for each criterion, with the interval that ties with it.

    The tolerance at a criterion is the standard error over the five regions of
    the lift at the best exponent, so the band widens exactly where the regions
    disagree or the surface is flat.
    """
    stack = np.stack(list(grids.values()) if isinstance(grids, dict)
                     else list(grids))               # (R, A, B)
    pooled = stack.mean(axis=0)
    error = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    rows = []
    for j, b in enumerate(BETAS):
        column = pooled[:, j]
        i = int(np.argmax(column))
        tolerance = float(error[i, j])
        inside = np.flatnonzero(column >= column[i] - tolerance)
        rows.append({"beta": b, "alpha_best": float(ALPHAS[i]),
                     "alpha_lo": float(ALPHAS[inside.min()]),
                     "alpha_hi": float(ALPHAS[inside.max()]),
                     "lift_best": float(column[i]),
                     "tolerance": tolerance})
    return pd.DataFrame(rows)


def monotone_strength(alpha_best: np.ndarray, betas=BETAS,
                      window=ENDPOINTS) -> float:
    """Rank correlation between the criterion exponent and the best score exponent."""
    from scipy.stats import spearmanr
    m = (betas >= window[0]) & (betas <= window[1])
    return float(spearmanr(betas[m], alpha_best[m]).statistic)
