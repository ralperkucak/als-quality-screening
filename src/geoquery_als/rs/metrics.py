"""Retrieval, selective risk and the exact null for a review budget.

An inspection programme orders the units of a delivery and examines the first
fraction of that order. The quantities here describe what such an order buys:
how many of the genuinely worst units it reaches, how much of the total quality
deficit it removes, and how those compare with the distribution produced by
reviewing the same number of units at random. The null is available in closed
form, so no resampling is needed to test a single ordering.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom, rankdata, spearmanr

EPS = 1e-12


def worst_set(quality: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k units with the lowest quality, ties broken by position."""
    return np.argsort(quality, kind="stable")[:k]


def budget_size(n: int, q: float) -> int:
    return max(1, int(np.ceil(q * n)))


def recall_at(quality: np.ndarray, score: np.ndarray, q: float) -> float:
    """Share of the truly worst q of units that the lowest scoring q contains."""
    k = budget_size(len(quality), q)
    worst = set(worst_set(quality, k).tolist())
    picked = np.argsort(score, kind="stable")[:k]
    return len(worst.intersection(picked.tolist())) / k


def exact_p_recall(n: int, k: int, hits: int) -> float:
    """Probability that random selection reaches at least `hits` of the worst k.

    Selecting k of n units without replacement and counting how many fall in a
    fixed set of size k is a hypergeometric experiment, so the null needs no
    simulation and the p value has no resolution floor.
    """
    return float(hypergeom.sf(hits - 1, n, k, k))


def recall_report(quality: np.ndarray, score: np.ndarray, q: float) -> dict:
    n = len(quality)
    k = budget_size(n, q)
    worst = set(worst_set(quality, k).tolist())
    picked = np.argsort(score, kind="stable")[:k]
    hits = len(worst.intersection(picked.tolist()))
    rv = hypergeom(n, k, k)
    return {
        "n_units": n, "k": k, "hits": hits,
        "recall": hits / k,
        "random_recall": k / n,
        "p_exact": exact_p_recall(n, k, hits),
        "random_lo": float(rv.ppf(0.025)) / k,
        "random_hi": float(rv.ppf(0.975)) / k,
    }


def captured_deficit(deficit: np.ndarray, score: np.ndarray, q: float) -> float:
    """Share of the total quality deficit held by the selected units."""
    k = budget_size(len(deficit), q)
    picked = np.argsort(score, kind="stable")[:k]
    return float(deficit[picked].sum() / max(deficit.sum(), EPS))


def risk_coverage(deficit: np.ndarray, score: np.ndarray) -> tuple:
    """Selective risk against coverage for an ordering that reviews low scores first.

    Coverage is the share of units left unreviewed and selective risk is the mean
    deficit among them, which is the quantity an operator carries after the
    review stops. Both curves run from reviewing nothing to reviewing everything.
    """
    order = np.argsort(score, kind="stable")
    d = deficit[order]
    n = len(d)
    remaining = d[::-1].cumsum()[::-1]
    counts = np.arange(n, 0, -1)
    coverage = counts / n
    risk = remaining / counts
    return coverage[::-1], risk[::-1]


def aurc(deficit: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk coverage curve."""
    coverage, risk = risk_coverage(deficit, score)
    return float(np.trapezoid(risk, coverage))


def normalised_aurc(deficit: np.ndarray, score: np.ndarray) -> float:
    """Excess area over the oracle, scaled by the excess a random order carries.

    Zero marks an ordering that cannot be improved and one marks an ordering no
    better than reviewing in arbitrary order, so values are comparable between
    regions of different difficulty.
    """
    best = aurc(deficit, -deficit)
    random = float(np.trapezoid(np.full(len(deficit), deficit.mean()),
                                np.arange(1, len(deficit) + 1) / len(deficit)))
    got = aurc(deficit, score)
    return float((got - best) / max(random - best, EPS))


def cost_weighted_capture(deficit: np.ndarray, score: np.ndarray,
                          cost: np.ndarray, budget: float) -> float:
    """Deficit recovered when the review budget is expressed as cost, not count."""
    order = np.argsort(score, kind="stable")
    spent = np.cumsum(cost[order]) / max(cost.sum(), EPS)
    taken = order[spent <= budget]
    if len(taken) == 0:
        taken = order[:1]
    return float(deficit[taken].sum() / max(deficit.sum(), EPS))


def rank_correlation(quality: np.ndarray, score: np.ndarray) -> float:
    return float(spearmanr(quality, score).statistic)


def ndcg(quality: np.ndarray, score: np.ndarray, deficit: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    gain = deficit[order]
    discount = 1.0 / np.log2(np.arange(2, len(gain) + 2))
    ideal = np.sort(deficit)[::-1]
    return float((gain * discount).sum() / max((ideal * discount).sum(), EPS))


def combine_p(pvals) -> float:
    """Fisher combination of independent per region tests."""
    from scipy.stats import chi2
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1.0)
    stat = -2.0 * np.log(p).sum()
    return float(chi2.sf(stat, 2 * len(p)))


def deficit_from(quality: np.ndarray, reference: float = None) -> np.ndarray:
    """Quality deficit of a unit, either against perfection or against a level."""
    if reference is None:
        return 1.0 - quality
    return np.maximum(0.0, reference - quality)


def rank_percentile(values: np.ndarray) -> np.ndarray:
    return (rankdata(values) - 0.5) / len(values)
