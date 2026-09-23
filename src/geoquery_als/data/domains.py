"""Recover FRACTAL's five spatial domains from patch centroids.

The dataset card states that the five acquisition regions are separated by at
least ~100 km, but the GeoPackage carries no explicit domain field. Single
linkage at a threshold far below the inter-region gap and far above the
intra-region nearest-neighbour spacing therefore recovers the regions exactly
and without tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DomainRecovery:
    labels: np.ndarray                # component id per patch
    n_components: int
    link_threshold_m: float
    min_inter_domain_distance_m: float
    max_intra_domain_nn_distance_m: float


def recover_domains(xy: np.ndarray, link_threshold_m: float = 20_000.0) -> DomainRecovery:
    """Single-linkage clustering of patch centroids at a fixed distance threshold.

    Parameters
    ----------
    xy : (N, 2) array of projected coordinates in metres (EPSG:2154).
    link_threshold_m : two patches are linked when closer than this distance.
        The result is stable for any value between the largest intra-region
        nearest-neighbour gap and the smallest inter-region gap; the separation
        of those two scales is reported so the choice can be audited.
    """
    tree = cKDTree(xy)
    pairs = tree.query_pairs(link_threshold_m, output_type="ndarray")
    n = xy.shape[0]
    if pairs.size == 0:
        raise ValueError("no patch pairs within the linkage threshold")
    adj = coo_matrix(
        (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    )
    n_comp, labels = connected_components(adj, directed=False)

    # Relabel by descending component size so ids are stable across reruns.
    order = pd.Series(labels).value_counts().index.to_numpy()
    remap = np.full(n_comp, -1, dtype=np.int64)
    remap[order] = np.arange(n_comp)
    labels = remap[labels]

    nn_dist, _ = tree.query(xy, k=2)
    max_intra = 0.0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() > 1:
            sub = cKDTree(xy[m])
            d, _ = sub.query(xy[m], k=2)
            max_intra = max(max_intra, float(d[:, 1].max()))

    min_inter = np.inf
    uniq = np.unique(labels)
    for i, a in enumerate(uniq):
        ta = cKDTree(xy[labels == a])
        for b in uniq[i + 1:]:
            d, _ = ta.query(xy[labels == b], k=1)
            min_inter = min(min_inter, float(d.min()))

    return DomainRecovery(labels, int(n_comp), float(link_threshold_m),
                          float(min_inter), float(max_intra))


def assign_domain_names(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach stable, geographically interpretable domain identifiers.

    Domains are named D1..D5 in order of increasing centroid easting so that the
    naming does not depend on cluster discovery order.
    """
    out = df.copy()
    out["_comp"] = labels
    centre = out.groupby("_comp")[["centroid_x", "centroid_y"]].mean()
    order = centre.sort_values("centroid_x").index.to_list()
    name = {c: f"D{i + 1}" for i, c in enumerate(order)}
    out["domain"] = out["_comp"].map(name)
    return out.drop(columns="_comp")
