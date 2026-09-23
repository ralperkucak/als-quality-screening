"""Random-access reader over the preprocessed patch store.

Each archive becomes one shard directory holding a concatenated array per
field plus an offsets vector. The arrays are memory-mapped, so reading a single
patch costs one seek regardless of how many patches the store holds. This
matters because active learning touches a few hundred scattered patches per
round rather than streaming the store in order.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NUM_CLASSES, PROCESSED_DIR
from .preprocess import FIELDS, XY_QUANT, Z_QUANT

STORE_DIR = PROCESSED_DIR / "patches"


class PatchStore:
    """Memory-mapped access to preprocessed FRACTAL patches."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = Path(store_dir or STORE_DIR)
        self.index = self._build_index()
        self._maps: dict[tuple[str, str], np.memmap] = {}

    # -- index ------------------------------------------------------------
    def _build_index(self) -> pd.DataFrame:
        cache = self.store_dir / "_index.parquet"
        shards = sorted(d for d in self.store_dir.iterdir()
                        if d.is_dir() and (d / "meta.json").exists())
        if cache.exists():
            idx = pd.read_parquet(cache)
            if set(idx["shard"].unique()) == {d.name for d in shards}:
                return idx

        rows = []
        for d in shards:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            offsets = np.load(d / "offsets.npy")
            for k, m in enumerate(meta):
                rows.append({
                    "patch_id": m["patch_id"], "shard": d.name, "local_index": k,
                    "offset": int(offsets[k]), "count": int(offsets[k + 1] - offsets[k]),
                    "split": d.name.split("-")[0],
                    "centroid_x": m["centroid_x"], "centroid_y": m["centroid_y"],
                    "z_min": m["z_min"], "z_max": m["z_max"],
                    "n_raw": m["n_raw"], "n_voxel": m["n_voxel"],
                    "n_dropped_noise": m["n_dropped_noise"],
                    **{f"stored_n_{c}": m[f"stored_n_{c}"] for c in range(NUM_CLASSES)},
                })
        idx = pd.DataFrame(rows)
        if not idx.empty:
            idx.to_parquet(cache, index=False)
        return idx

    def available_patches(self) -> set[str]:
        return set(self.index["patch_id"])

    # -- reading ----------------------------------------------------------
    def _map(self, shard: str, field: str) -> np.memmap:
        key = (shard, field)
        if key not in self._maps:
            self._maps[key] = np.load(self.store_dir / shard / f"{field}.npy",
                                      mmap_mode="r")
        return self._maps[key]

    @lru_cache(maxsize=1)
    def _lookup(self) -> dict[str, tuple[str, int, int]]:
        return {r.patch_id: (r.shard, r.offset, r.count)
                for r in self.index.itertuples(index=False)}

    def read(self, patch_id: str) -> dict[str, np.ndarray]:
        shard, off, cnt = self._lookup()[patch_id]
        sl = slice(off, off + cnt)
        return {f: np.asarray(self._map(shard, f)[sl]) for f in FIELDS}

    def read_xyz_label(self, patch_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Metric coordinates relative to the patch origin, plus labels."""
        d = self.read(patch_id)
        xyz = np.empty((len(d["z"]), 3), dtype=np.float32)
        xyz[:, :2] = d["xy"].astype(np.float32) * XY_QUANT
        xyz[:, 2] = d["z"].astype(np.float32) * Z_QUANT
        return xyz, d["label"].astype(np.int64)


def build_features(d: dict[str, np.ndarray], *, geometry_only: bool = False
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz in metres, per-point feature matrix).

    Feature channels, in order:
      0 intensity, scaled and clipped
      1 return number / number of returns
      2 number of returns, scaled
      3 red, 4 green, 5 blue, 6 near infrared, all scaled to [0, 1]
      7 NDVI in [-1, 1]

    In the geometry-only configuration the four colour channels and NDVI are
    replaced by zeros so that the network sees exactly the same tensor shape and
    the comparison isolates the information content, not the architecture.
    """
    n = len(d["z"])
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, :2] = d["xy"].astype(np.float32) * XY_QUANT
    xyz[:, 2] = d["z"].astype(np.float32) * Z_QUANT

    feats = np.zeros((n, 8), dtype=np.float32)
    feats[:, 0] = np.clip(d["intensity"].astype(np.float32) / 2000.0, 0.0, 1.0)
    nr = np.maximum(d["returns"][:, 1].astype(np.float32), 1.0)
    feats[:, 1] = np.clip(d["returns"][:, 0].astype(np.float32) / nr, 0.0, 1.0)
    feats[:, 2] = np.clip(nr / 7.0, 0.0, 1.0)
    if not geometry_only:
        rgbn = d["rgbn"].astype(np.float32) / 255.0
        feats[:, 3:7] = rgbn
        red, nir = rgbn[:, 0], rgbn[:, 3]
        feats[:, 7] = (nir - red) / np.maximum(nir + red, 1e-6)
    return xyz, feats
