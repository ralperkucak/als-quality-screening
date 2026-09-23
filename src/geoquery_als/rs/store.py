"""Rebuild unit-level quantities from the cached per-patch sufficient statistics.

Every quantity reported by the study is derived here from two cached objects: the
per-patch confusion matrix, which fixes the reference side, and the per-patch
per-class confidence histograms and sums, which fix the prediction side. Nothing
is read from a previously derived table, so an arbitrary grouping of patches into
units, an arbitrary class weighting and an arbitrary quality criterion can be
evaluated without returning to the point store.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "pivot"
FOLDS = ("D1", "D2", "D3", "D4", "D5")
CLASS_NAMES = ("other", "ground", "vegetation", "building", "water", "bridge",
               "permanent_structure")
NUM_CLASSES = len(CLASS_NAMES)
HIST_BINS = 200
# The confidence histogram is accumulated over the reachable range of a maximum
# softmax probability, which starts at one over the number of classes rather
# than at zero; see `reliability.statistics.patch_statistics`. Reading it on a
# unit grid shifts every value and every threshold by about an eighth.
CONF_LO = 1.0 / NUM_CLASSES


@dataclass(frozen=True)
class PatchStats:
    """Per-patch sufficient statistics of one fold and one partition family."""

    patch_id: np.ndarray          # (P,)
    n_full: np.ndarray            # (P,)
    confusion: np.ndarray         # (P, 7, 7), rows reference, columns prediction
    conf_hist: np.ndarray         # (P, 7, 200), point counts by predicted class
    support_hist: np.ndarray      # (P, 7, 200)
    n_pred: np.ndarray            # (P, 7)
    sum_conf: np.ndarray          # (P, 7)
    sum_entropy: np.ndarray       # (P, 7)
    sum_calibrated: np.ndarray    # (P, 7)
    sum_support: np.ndarray       # (P, 7)
    sum_reliability: np.ndarray   # (P, 7)

    @property
    def index(self) -> dict:
        return {pid: i for i, pid in enumerate(self.patch_id)}


def suffix(backbone: str) -> str:
    """Filename suffix that keeps one training run's artefacts from another's."""
    return "" if backbone == "randlanet" else f"_{backbone}"


@lru_cache(maxsize=32)
def load_patch_stats(fold: str, family: str = "target",
                     backbone: str = "randlanet") -> PatchStats:
    z = np.load(BASE / "unit_statistics" / f"{family}_{fold}{suffix(backbone)}.npz",
                allow_pickle=True)
    return PatchStats(
        patch_id=np.asarray([str(p) for p in z["patch_id"]]),
        n_full=z["n_full"].astype(np.int64),
        confusion=z["confusion"].astype(np.int64),
        conf_hist=z["conf_hist"].astype(np.float64),
        support_hist=z["support_hist"].astype(np.float64),
        n_pred=z["n_pred"].astype(np.float64),
        sum_conf=z["sum_conf"].astype(np.float64),
        sum_entropy=z["sum_entropy"].astype(np.float64),
        sum_calibrated=z["sum_calibrated"].astype(np.float64),
        sum_support=z["sum_support"].astype(np.float64),
        sum_reliability=z["sum_reliability"].astype(np.float64),
    )


@lru_cache(maxsize=8)
def load_calibration(fold: str, backbone: str = "randlanet") -> dict:
    z = np.load(BASE / "calibration" / f"fitted_{fold}{suffix(backbone)}.npz",
                allow_pickle=True)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=8)
def load_source_points(fold: str, backbone: str = "randlanet") -> dict:
    z = np.load(BASE / "calibration" / f"source_{fold}{suffix(backbone)}.npz",
                allow_pickle=True)
    return {k: z[k] for k in z.files}


def bin_centres(lo: float = CONF_LO, hi: float = 1.0) -> np.ndarray:
    """Centres of the histogram bins, on the grid the histogram was built on.

    The default is the confidence grid. The support histogram is accumulated on
    the unit interval, so a caller reading that one passes `lo=0.0`.
    """
    edges = np.linspace(lo, hi, HIST_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


@dataclass(frozen=True)
class UnitStats:
    """Unit-level sufficient statistics for one fold, in the order of `unit_id`."""

    fold: str
    unit_id: np.ndarray           # (U,)
    n_patches: np.ndarray         # (U,)
    n_points: np.ndarray          # (U,)
    confusion: np.ndarray         # (U, 7, 7)
    conf_hist: np.ndarray         # (U, 7, 200)
    support_hist: np.ndarray      # (U, 7, 200)
    n_pred: np.ndarray            # (U, 7)
    sum_conf: np.ndarray          # (U, 7)
    sum_entropy: np.ndarray       # (U, 7)
    sum_calibrated: np.ndarray    # (U, 7)
    sum_support: np.ndarray       # (U, 7)
    sum_reliability: np.ndarray   # (U, 7)
    x: np.ndarray = None          # (U,) easting of the unit centre
    y: np.ndarray = None
    sampled_km2: np.ndarray = None

    @property
    def n_units(self) -> int:
        return len(self.unit_id)

    @property
    def n_ref(self) -> np.ndarray:
        """Reference point count per class, (U, 7)."""
        return self.confusion.sum(axis=2)


def group_patches(stats: PatchStats, members: pd.DataFrame, fold: str,
                  geometry: pd.DataFrame = None,
                  min_patches: int = 8) -> UnitStats:
    """Sum per-patch statistics over the patches of each unit.

    `members` needs the columns `patch_id` and `unit_id`. Units holding fewer
    than `min_patches` patches are dropped, which is the inclusion criterion of
    the study. `geometry` optionally supplies `unit_id`, `unit_x`, `unit_y` and
    `sampled_km2`.
    """
    idx = stats.index
    members = members[members["patch_id"].isin(idx)].copy()
    members["row"] = members["patch_id"].map(idx).astype(np.int64)
    counts = members.groupby("unit_id")["row"].size()
    keep = counts[counts >= min_patches].index
    members = members[members["unit_id"].isin(keep)]

    unit_ids = np.asarray(sorted(members["unit_id"].unique()))
    slot = {u: i for i, u in enumerate(unit_ids)}
    target = members["unit_id"].map(slot).to_numpy(dtype=np.int64)
    rows = members["row"].to_numpy(dtype=np.int64)
    u = len(unit_ids)

    def accumulate(source):
        out = np.zeros((u,) + source.shape[1:], dtype=np.float64)
        np.add.at(out, target, source[rows])
        return out

    result = UnitStats(
        fold=fold,
        unit_id=unit_ids,
        n_patches=np.bincount(target, minlength=u).astype(np.int64),
        n_points=accumulate(stats.n_full.astype(np.float64)).astype(np.int64),
        confusion=accumulate(stats.confusion.astype(np.float64)),
        conf_hist=accumulate(stats.conf_hist),
        support_hist=accumulate(stats.support_hist),
        n_pred=accumulate(stats.n_pred),
        sum_conf=accumulate(stats.sum_conf),
        sum_entropy=accumulate(stats.sum_entropy),
        sum_calibrated=accumulate(stats.sum_calibrated),
        sum_support=accumulate(stats.sum_support),
        sum_reliability=accumulate(stats.sum_reliability),
    )
    if geometry is not None:
        g = geometry.drop_duplicates("unit_id").set_index("unit_id").reindex(unit_ids)
        result = replace(result,
                         x=g["unit_x"].to_numpy(dtype=float),
                         y=g["unit_y"].to_numpy(dtype=float),
                         sampled_km2=g["sampled_km2"].to_numpy(dtype=float))
    return result


def regrid(members: pd.DataFrame, scale_m: int, offset=(0, 0)) -> pd.DataFrame:
    """Reassign patches to a regular grid of the given spacing and origin shift."""
    out = members.copy()
    cx = np.floor((out["centroid_x"].to_numpy() - offset[0]) / scale_m).astype(np.int64)
    cy = np.floor((out["centroid_y"].to_numpy() - offset[1]) / scale_m).astype(np.int64)
    out["unit_id"] = [f"{f}_{fam}_{scale_m}_{a}_{b}"
                      for f, fam, a, b in zip(out["fold"], out["family"], cx, cy)]
    out["unit_x"] = (cx + 0.5) * scale_m + offset[0]
    out["unit_y"] = (cy + 0.5) * scale_m + offset[1]
    out["sampled_km2"] = np.nan
    return out


def load_units(fold: str, scale_m: int = 2000, family: str = "target",
               min_patches: int = 8, offset=(0, 0),
               partition: str = "all", backbone: str = "randlanet") -> UnitStats:
    """Unit statistics for one fold at one aggregation scale and grid alignment.

    `partition` selects `all` patches of the family, or `test` to restrict the
    target family to the held-out test split.
    """
    members = pd.read_parquet(BASE / "units" / "unit_members.parquet")
    members = members[(members["fold"] == fold) & (members["family"] == family)
                      & (members["scale_m"] == scale_m)]
    geometry = pd.read_parquet(BASE / "units" / "units.parquet")
    geometry = geometry[(geometry["fold"] == fold) & (geometry["family"] == family)
                        & (geometry["scale_m"] == scale_m)]
    if offset != (0, 0):
        members = regrid(members, scale_m, offset)
        geometry = members[["unit_id", "unit_x", "unit_y", "sampled_km2"]]
    stats = load_patch_stats(fold, family, backbone)
    cols = ["patch_id", "unit_id"]
    return group_patches(stats, members[cols], fold, geometry=geometry,
                         min_patches=min_patches)
