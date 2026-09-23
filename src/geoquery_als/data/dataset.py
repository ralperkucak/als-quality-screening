"""Torch dataset that produces RandLA-Net's multi-resolution neighbourhood inputs.

RandLA-Net needs, for every resolution level, the k nearest neighbours of each
point, the neighbours used for random-sample pooling, and the nearest coarse
point used for interpolation during upsampling. These are geometric quantities
that do not depend on the network weights, so they are built on CPU, as in the
reference implementation.

Two implementation details make the active-learning loop affordable.

First, because random sampling keeps the first n/4 entries of an already
shuffled point list, the coordinates at every level are prefixes of the level-0
coordinates, so only one coordinate array has to be stored.

Second, the neighbourhood indices depend on the sampled point subset but not on
the geometric augmentation, which is a rotation, a reflection, a near-unit
anisotropic scaling and a centimetre-scale jitter. Applying the augmentation
after the indices are built therefore leaves the neighbourhood graph valid and
allows the indices to be cached and reused across active-learning rounds.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .store import PatchStore, build_features


@dataclass(frozen=True)
class SamplingConfig:
    num_points: int = 12288
    k_neighbours: int = 16
    sub_ratio: int = 4
    num_layers: int = 4
    coord_scale_m: float = 25.0     # patch half-width, matches the published baseline
    geometry_only: bool = False


def build_hierarchy(xyz: np.ndarray, cfg: SamplingConfig) -> dict[str, list[np.ndarray]]:
    """Neighbour, pooling and interpolation indices for every resolution level."""
    neigh, pool, up = [], [], []
    n = xyz.shape[0]
    for _ in range(cfg.num_layers):
        cur = xyz[:n]
        tree = cKDTree(cur)
        k_eff = min(cfg.k_neighbours, n)
        _, nn_idx = tree.query(cur, k=k_eff, workers=1)
        nn_idx = np.atleast_2d(nn_idx)
        if k_eff < cfg.k_neighbours:
            # See the note in `_knn_indices`: repeat the farthest retained
            # neighbour rather than emit an index outside the level.
            pad = np.repeat(nn_idx[:, -1:], cfg.k_neighbours - k_eff, axis=1)
            nn_idx = np.concatenate([nn_idx, pad], axis=1)
        n_sub = max(n // cfg.sub_ratio, 1)
        neigh.append(nn_idx.astype(np.int32))
        pool.append(nn_idx[:n_sub].astype(np.int32))
        _, up_idx = cKDTree(xyz[:n_sub]).query(cur, k=1, workers=1)
        up.append(np.atleast_1d(up_idx).astype(np.int32))
        n = n_sub
    return {"neigh_idx": neigh, "pool_idx": pool, "up_idx": up}


# Caching the neighbourhood hierarchy pays only while the cached heap stays
# small. Each entry is about 1.5 MB, and randomly reading a multi-gigabyte heap
# of such entries thrashes the address translation buffers: measured
# single-thread throughput falls from 468 items per second at 600 cached patches
# to 47 at 4,000, while rebuilding the hierarchy uncached costs a flat 23 items
# per second per thread and scales across threads, because the k-d tree query
# releases the interpreter lock. Above this many patches the hierarchy is
# therefore recomputed and only the much smaller projection map is cached.
HIERARCHY_CACHE_LIMIT = 1200


class HierarchyCache:
    """Thread-safe store of per-patch point selection and neighbourhood indices.

    `enable_hierarchy` turns off the large part of the cache while keeping the
    projection map, which is two orders of magnitude smaller per patch and is
    always worth keeping.
    """

    def __init__(self, max_entries: int = 8000, enable_hierarchy: bool = True) -> None:
        self._d: dict[str, tuple] = {}
        self._proj: dict[str, tuple] = {}
        self._lock = Lock()
        self.max_entries = max_entries
        self.enable_hierarchy = enable_hierarchy
        self.hits = 0
        self.misses = 0

    @classmethod
    def for_working_set(cls, n_patches: int) -> "HierarchyCache":
        """Cache the hierarchy only when the working set is small enough to pay."""
        return cls(max_entries=n_patches + 200,
                   enable_hierarchy=n_patches <= HIERARCHY_CACHE_LIMIT)

    def get(self, key: str):
        if not self.enable_hierarchy:
            return None
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
            return v

    def put(self, key: str, value: tuple) -> None:
        if not self.enable_hierarchy:
            return
        with self._lock:
            if len(self._d) < self.max_entries:
                self._d[key] = value

    def get_projection(self, key: str):
        """Nearest-neighbour map from every stored point to the sampled subset.

        The subset is deterministic, so this map is too, and evaluating a patch
        at a later active-learning round costs one array lookup instead of a
        fresh k-d tree query over tens of thousands of points.
        """
        with self._lock:
            return self._proj.get(key)

    def put_projection(self, key: str, value: tuple) -> None:
        with self._lock:
            if len(self._proj) < self.max_entries:
                self._proj[key] = value

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._proj.clear()

    def stats(self) -> dict:
        return {"hierarchy_enabled": self.enable_hierarchy, "entries": len(self._d),
                "projections": len(self._proj), "hits": self.hits, "misses": self.misses}


class PatchDataset(Dataset):
    """Patches addressed by identifier, with optional geometric augmentation.

    `patch_ids` may repeat, which is how the adaptation stage mixes a small
    labelled target set with source replay at a fixed ratio.
    """

    def __init__(self, store: PatchStore, patch_ids: list[str], cfg: SamplingConfig,
                 *, augment: bool = False, seed: int = 0,
                 cache: HierarchyCache | None = None,
                 hierarchy_on_device: bool = False) -> None:
        self.store = store
        self.patch_ids = list(patch_ids)
        self.cfg = cfg
        self.augment = augment
        self.seed = seed
        self.cache = cache
        # With the neighbourhood graph built on the device, the loader only has
        # to assemble points and features, which is what turned the CPU from the
        # bottleneck back into a feeder.
        self.hierarchy_on_device = hierarchy_on_device

    def __len__(self) -> int:
        return len(self.patch_ids)

    def _point_selection(self, pid: str, n_available: int) -> np.ndarray:
        """Deterministic point subset, stable across processes and runs.

        A stable digest is used rather than Python's built-in string hash, which
        is salted per process and would silently break reproducibility.
        """
        digest = hashlib.blake2b(
            f"{pid}|{self.seed}|{self.cfg.num_points}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        need = self.cfg.num_points
        if n_available >= need:
            sel = rng.choice(n_available, size=need, replace=False)
        else:
            sel = np.concatenate([np.arange(n_available),
                                  rng.choice(n_available, size=need - n_available,
                                             replace=True)])
        rng.shuffle(sel)                     # random sampling relies on random order
        return sel.astype(np.int64)

    def __getitem__(self, item: int) -> dict:
        pid = self.patch_ids[item]
        raw = self.store.read(pid)
        xyz, feats = build_features(raw, geometry_only=self.cfg.geometry_only)
        labels = raw["label"].astype(np.int64)

        if self.hierarchy_on_device:
            sel, hier = self._point_selection(pid, xyz.shape[0]), None
        else:
            cached = self.cache.get(pid) if self.cache is not None else None
            if cached is None:
                sel = self._point_selection(pid, xyz.shape[0])
                hier = build_hierarchy(xyz[sel] / self.cfg.coord_scale_m, self.cfg)
                if self.cache is not None:
                    self.cache.put(pid, (sel, hier))
            else:
                sel, hier = cached

        xyz_s, feats_s, lab_s = xyz[sel], feats[sel], labels[sel]
        if self.augment:
            xyz_s = _augment(xyz_s, np.random.default_rng())
        norm = xyz_s / self.cfg.coord_scale_m

        common = {
            "features": torch.from_numpy(np.ascontiguousarray(
                np.concatenate([norm, feats_s], axis=1).T.astype(np.float32))),
            "labels": torch.from_numpy(lab_s),
            "point_index": torch.from_numpy(sel.astype(np.int64)),
            "patch_index": item,
        }
        if self.hierarchy_on_device:
            common["xyz_norm"] = torch.from_numpy(
                np.ascontiguousarray(norm.astype(np.float32)))
            return common

        n = norm.shape[0]
        xyz_levels = []
        for _ in range(self.cfg.num_layers):
            xyz_levels.append(torch.from_numpy(
                np.ascontiguousarray(norm[:n].T.astype(np.float32))))
            n = max(n // self.cfg.sub_ratio, 1)

        return {
            **common,
            "xyz": xyz_levels,
            # Indices stay 32-bit on the host and are widened on the device.
            # torch.gather needs 64-bit indices, but converting here would double
            # both the host memory traffic and the transfer for the largest
            # tensors in the batch.
            "neigh_idx": [torch.from_numpy(a) for a in hier["neigh_idx"]],
            "pool_idx": [torch.from_numpy(a) for a in hier["pool_idx"]],
            "up_idx": [torch.from_numpy(a) for a in hier["up_idx"]],
        }


def _augment(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation about the vertical axis, reflection, near-unit anisotropic scaling
    and centimetre jitter.

    Vertical structure carries most of the class signal in ALS data, so rotation
    is restricted to the z axis and the vertical scale stays close to one. All of
    these leave the k-nearest-neighbour graph effectively unchanged, which is why
    the augmentation is applied after the neighbourhood indices are built.
    """
    out = xyz.copy()
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    out[:, :2] = out[:, :2] @ np.array([[c, -s], [s, c]], dtype=np.float32).T
    if rng.random() < 0.5:
        out[:, 0] *= -1.0
    scale = rng.uniform(0.95, 1.05, size=3).astype(np.float32)
    scale[2] = rng.uniform(0.98, 1.02)
    out *= scale
    out += rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
    return out


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    for key in ("xyz", "neigh_idx", "pool_idx", "up_idx"):
        if key in batch[0]:
            out[key] = [torch.stack([b[key][i] for b in batch])
                        for i in range(len(batch[0][key]))]
    if "xyz_norm" in batch[0]:
        out["xyz_norm"] = torch.stack([b["xyz_norm"] for b in batch])
    out["features"] = torch.stack([b["features"] for b in batch])
    out["labels"] = torch.stack([b["labels"] for b in batch])
    out["point_index"] = torch.stack([b["point_index"] for b in batch])
    out["patch_index"] = torch.tensor([b["patch_index"] for b in batch])
    return out


INDEX_KEYS = ("neigh_idx", "pool_idx", "up_idx")


def to_device(batch: dict, device: torch.device,
              cfg: SamplingConfig | None = None) -> dict:
    """Move a batch to the device, building the neighbourhood graph there when
    the loader did not build it on the host."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            moved = [t.to(device, non_blocking=True) for t in v]
            out[k] = [t.long() for t in moved] if k in INDEX_KEYS else moved
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    if "neigh_idx" not in out:
        if cfg is None:
            raise ValueError("batch carries no neighbourhood graph and no "
                             "sampling configuration was given to build one")
        out.update(build_hierarchy_on_device(out["xyz_norm"], cfg))
    return out


# ---------------------------------------------------------------------------
# device-side neighbourhood construction
# ---------------------------------------------------------------------------
def _knn_indices(ref: torch.Tensor, query: torch.Tensor, k: int,
                 chunk: int = 512) -> torch.Tensor:
    """Exact k nearest neighbours of `query` among `ref`, batched.

    Squared distances come from a matrix product rather than an explicit
    difference, which keeps the work on the tensor cores, and the query axis is
    processed in chunks so the distance matrix is never held in full. The term
    that depends only on the query is omitted because it does not change the
    ordering.

    ref   : (B, M, 3)
    query : (B, Q, 3)
    out   : (B, Q, k) int32
    """
    m = ref.shape[1]
    k_eff = min(k, m)
    ref_sq = (ref * ref).sum(-1).unsqueeze(1)                 # (B, 1, M)
    parts = []
    for start in range(0, query.shape[1], chunk):
        q = query[:, start:start + chunk]                     # (B, c, 3)
        d = ref_sq - 2.0 * torch.bmm(q, ref.transpose(1, 2))  # (B, c, M)
        parts.append(d.topk(k_eff, dim=-1, largest=False).indices)
    idx = torch.cat(parts, dim=1)
    if k_eff < k:
        # The coarsest levels can hold fewer points than the neighbourhood size.
        # Repeating the farthest retained neighbour keeps the tensor shape the
        # network expects without inventing an index that does not exist.
        idx = torch.cat([idx, idx[..., -1:].expand(-1, -1, k - k_eff)], dim=-1)
    return idx.to(torch.int32)


def build_hierarchy_on_device(xyz: torch.Tensor, cfg: SamplingConfig) -> dict:
    """The multi-resolution inputs RandLA-Net needs, computed on the device.

    Building this on CPU dominated the data pipeline: profiling attributed 93 %
    of the per-sample cost to the k-d tree, which left the GPU idle. The graph
    is the same exact k-nearest-neighbour graph, so the two paths are
    interchangeable, which `test_device_hierarchy_matches_the_cpu_one` checks.

    xyz : (B, N, 3) normalised coordinates, in the shuffled order that random
          sampling relies on.
    """
    neigh, pool, up, levels = [], [], [], []
    n = xyz.shape[1]
    for _ in range(cfg.num_layers):
        cur = xyz[:, :n]
        levels.append(cur)
        nn_idx = _knn_indices(cur, cur, cfg.k_neighbours)
        n_sub = max(n // cfg.sub_ratio, 1)
        neigh.append(nn_idx)
        pool.append(nn_idx[:, :n_sub])
        up.append(_knn_indices(xyz[:, :n_sub], cur, 1).squeeze(-1))
        n = n_sub
    return {"xyz": [t.transpose(1, 2).contiguous() for t in levels],
            "neigh_idx": [t.long() for t in neigh],
            "pool_idx": [t.long() for t in pool],
            "up_idx": [t.long() for t in up]}
