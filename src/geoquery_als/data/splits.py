"""Leave-one-domain-out split construction.

For a target domain d the protocol is:

  source train      patches of the `train` split from the four other domains
  source validation patches of the `val` split from the four other domains
  target pool       an unlabelled contiguous block of `train` patches of d
  target test       `test` patches of d, never touched by training or selection

The candidate pool is a contiguous block rather than a random sample of the
domain because the quantity under study is the geographic scale over which
neighbouring candidates are redundant. Random thinning would inflate the
spacing between candidates and destroy exactly the structure being measured.
The block is placed deterministically at the domain's spatial median, so the
choice involves no tuning, and it is identical for every method and every seed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import PATCH_SIDE_M


@dataclass(frozen=True)
class SplitConfig:
    n_source_train_per_domain: int = 1500
    n_source_val_per_domain: int = 200
    n_target_pool: int = 3000
    n_target_test: int = 800
    pool_block: str = "median"      # "median" or "alt" for the robustness block
    seed: int = 20260819


@dataclass
class FoldSpec:
    target_domain: str
    source_train: list[str]
    source_val: list[str]
    target_pool: list[str]
    target_test: list[str]
    diagnostics: dict


def _take(rng: np.random.Generator, ids: np.ndarray, n: int) -> list[str]:
    if len(ids) <= n:
        return sorted(ids.tolist())
    return sorted(rng.choice(ids, size=n, replace=False).tolist())


def _contiguous_block(df: pd.DataFrame, n: int, mode: str) -> pd.DataFrame:
    """The n patches closest to a deterministic anchor inside the domain."""
    xy = df[["centroid_x", "centroid_y"]].to_numpy()
    if mode == "median":
        anchor = np.median(xy, axis=0)
    elif mode == "alt":
        # Second block for the placement-robustness check: the patch farthest
        # from the median anchor, so the two blocks overlap as little as possible.
        med = np.median(xy, axis=0)
        anchor = xy[int(np.argmax(np.linalg.norm(xy - med, axis=1)))]
    else:
        raise ValueError(f"unknown pool_block mode: {mode}")
    d = np.linalg.norm(xy - anchor, axis=1)
    order = np.argsort(d, kind="stable")[:min(n, len(df))]
    return df.iloc[np.sort(order)]


def build_folds(index: pd.DataFrame, cfg: SplitConfig,
                available: set[str] | None = None) -> list[FoldSpec]:
    """Construct one fold per domain. `available` restricts to downloaded patches."""
    df = index if available is None else index[index["patch_id"].isin(available)]
    domains = sorted(df["domain"].unique())
    folds: list[FoldSpec] = []

    for k, target in enumerate(domains):
        rng = np.random.default_rng(cfg.seed + 1000 * k)
        src = df[(df["domain"] != target)]
        st, sv = [], []
        for dom in sorted(src["domain"].unique()):
            tr = src[(src["domain"] == dom) & (src["split"] == "train")]["patch_id"].to_numpy()
            va = src[(src["domain"] == dom) & (src["split"] == "val")]["patch_id"].to_numpy()
            st += _take(rng, tr, cfg.n_source_train_per_domain)
            sv += _take(rng, va, cfg.n_source_val_per_domain)

        tgt_train = df[(df["domain"] == target) & (df["split"] == "train")]
        block = _contiguous_block(tgt_train, cfg.n_target_pool, cfg.pool_block)
        pool = sorted(block["patch_id"].tolist())

        tgt_test = df[(df["domain"] == target) & (df["split"] == "test")]
        test = _take(rng, tgt_test["patch_id"].to_numpy(), cfg.n_target_test)

        folds.append(FoldSpec(target, sorted(st), sorted(sv), pool, test,
                              _fold_diagnostics(df, block, tgt_test, test)))
    return folds


def _fold_diagnostics(df: pd.DataFrame, block: pd.DataFrame,
                      tgt_test_all: pd.DataFrame, test_ids: list[str]) -> dict:
    bxy = block[["centroid_x", "centroid_y"]].to_numpy()
    nn, _ = cKDTree(bxy).query(bxy, k=2)
    txy = tgt_test_all.set_index("patch_id").loc[test_ids,
                                                 ["centroid_x", "centroid_y"]].to_numpy()
    d_bt, _ = cKDTree(bxy).query(txy, k=1)
    return {
        "pool_n": int(len(block)),
        "pool_median_nn_m": float(np.median(nn[:, 1])),
        "pool_mean_nn_m": float(np.mean(nn[:, 1])),
        "pool_extent_x_km": float((bxy[:, 0].max() - bxy[:, 0].min()) / 1000),
        "pool_extent_y_km": float((bxy[:, 1].max() - bxy[:, 1].min()) / 1000),
        "pool_area_km2": float(len(block) * 0.0025),
        "pool_convex_extent_km2": float(
            (bxy[:, 0].max() - bxy[:, 0].min()) * (bxy[:, 1].max() - bxy[:, 1].min()) / 1e6),
        "test_n": int(len(test_ids)),
        "min_pool_to_test_m": float(d_bt.min()),
        "median_pool_to_test_m": float(np.median(d_bt)),
    }


def folds_to_frame(folds: list[FoldSpec]) -> pd.DataFrame:
    rows = []
    for f in folds:
        for role, ids in (("source_train", f.source_train), ("source_val", f.source_val),
                          ("target_pool", f.target_pool), ("target_test", f.target_test)):
            rows += [{"fold": f.target_domain, "role": role, "patch_id": p} for p in ids]
    return pd.DataFrame(rows)


def check_leakage(folds: list[FoldSpec], index: pd.DataFrame) -> dict:
    """Every assertion a reviewer would want to see verified."""
    xy = index.set_index("patch_id")[["centroid_x", "centroid_y"]]
    report: dict = {"folds": {}, "passed": True}
    for f in folds:
        st, sv = set(f.source_train), set(f.source_val)
        tp, tt = set(f.target_pool), set(f.target_test)
        dom = index.set_index("patch_id")["domain"]
        checks = {
            "source_train_disjoint_from_target_pool": len(st & tp) == 0,
            "source_train_disjoint_from_target_test": len(st & tt) == 0,
            "source_val_disjoint_from_target_test": len(sv & tt) == 0,
            "target_pool_disjoint_from_target_test": len(tp & tt) == 0,
            "source_train_disjoint_from_source_val": len(st & sv) == 0,
            "no_target_domain_patch_in_source_train":
                not any(dom[p] == f.target_domain for p in st),
            "no_target_domain_patch_in_source_val":
                not any(dom[p] == f.target_domain for p in sv),
            "all_pool_in_target_domain": all(dom[p] == f.target_domain for p in tp),
            "all_test_in_target_domain": all(dom[p] == f.target_domain for p in tt),
        }
        pxy = xy.loc[sorted(tp)].to_numpy()
        txy = xy.loc[sorted(tt)].to_numpy()
        dmin = float(cKDTree(pxy).query(txy, k=1)[0].min())
        # Patches are 50 m squares on a regular grid, so two footprints overlap
        # exactly when their centroids are closer than 50 m. Touching neighbours
        # at exactly 50 m share no point and are not leakage.
        checks["pool_and_test_footprints_do_not_overlap"] = dmin >= PATCH_SIDE_M - 1e-6
        sxy = xy.loc[sorted(st | sv)].to_numpy()
        dsrc = float(cKDTree(sxy).query(txy, k=1)[0].min())
        checks["source_and_target_test_geographically_separated"] = dsrc > 50_000.0
        d_all = cKDTree(pxy).query(txy, k=1)[0]
        report["folds"][f.target_domain] = {
            "checks": checks,
            "min_pool_to_test_distance_m": dmin,
            "median_pool_to_test_distance_m": float(np.median(d_all)),
            "share_of_test_within_500m_of_pool": float((d_all < 500.0).mean()),
            "min_source_to_target_test_distance_m": dsrc,
            "sizes": {"source_train": len(st), "source_val": len(sv),
                      "target_pool": len(tp), "target_test": len(tt)},
        }
        report["passed"] &= all(checks.values())
    return report


def config_as_dict(cfg: SplitConfig) -> dict:
    return asdict(cfg)
