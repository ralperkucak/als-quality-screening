"""Convert FRACTAL LAZ patches into a compact, fixed-schema binary store.

Each patch is voxel-subsampled at 0.25 m (the spacing used by the published
FRACTAL baseline) and then randomly thinned to a maximum point count. Points
are stored as small integers relative to the patch origin, which keeps the whole
benchmark on a laptop-sized disk while preserving centimetre geometry.

Classification codes 65 and 66 (noise and virtual/synthetic points) are outside
the 7-class FRACTAL nomenclature and are dropped before subsampling; their share
is recorded per patch so the loss can be reported.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import laspy
import numpy as np

from ..config import ASPRS_TO_TRAIN_ID, NUM_CLASSES

VOXEL_SIZE_M = 0.25
XY_QUANT = 0.01          # metres per int16 step for x, y (relative to centroid)
Z_QUANT = 0.02           # metres per int16 step for z (relative to patch minimum)
DROP_CODES = (65, 66)    # noise / virtual points, outside the nomenclature

_LUT = np.full(256, -1, dtype=np.int8)
for _code, _tid in ASPRS_TO_TRAIN_ID.items():
    _LUT[_code] = _tid


@dataclass
class PatchArrays:
    xy: np.ndarray        # (M, 2) int16, centimetres relative to patch centroid
    z: np.ndarray         # (M,)   int16, 2 cm steps above the patch minimum
    intensity: np.ndarray  # (M,)  uint16
    returns: np.ndarray   # (M, 2) uint8, (return_number, number_of_returns)
    rgbn: np.ndarray      # (M, 4) uint8, red green blue nir
    label: np.ndarray     # (M,)   int8, 0..6
    meta: dict


def _voxel_indices(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Index of one representative point per occupied voxel.

    The representative is the point closest to its voxel centre, chosen
    deterministically so that the store does not depend on point order.
    """
    grid = np.floor(pts / voxel)
    key = np.ascontiguousarray(grid.astype(np.int64))
    # Fold the 3D grid coordinate into a single sortable integer.
    key -= key.min(axis=0, keepdims=True)
    span = key.max(axis=0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    centre = (grid + 0.5) * voxel
    d2 = np.sum((pts - centre) ** 2, axis=1)
    order = np.lexsort((d2, flat))
    flat_sorted = flat[order]
    first = np.ones(len(flat_sorted), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    return order[first]


def preprocess_laz(raw: bytes, patch_id: str, max_points: int,
                   voxel: float = VOXEL_SIZE_M) -> PatchArrays:
    las = laspy.read(io.BytesIO(raw))
    cls = np.asarray(las.classification)
    n_raw = int(len(cls))

    keep = ~np.isin(cls, DROP_CODES)
    label = _LUT[cls[keep]]
    unmapped = int((label < 0).sum())
    if unmapped:
        keep_idx = np.flatnonzero(keep)[label >= 0]
        keep = np.zeros(n_raw, dtype=bool)
        keep[keep_idx] = True
        label = _LUT[cls[keep]]

    xyz = np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], axis=1)[keep]
    if len(xyz) == 0:
        raise ValueError(f"{patch_id}: no points left after class filtering")

    sel = _voxel_indices(xyz, voxel)
    n_voxel = int(len(sel))
    if n_voxel > max_points:
        rng = np.random.default_rng(
            int.from_bytes(hashlib.blake2b(patch_id.encode(), digest_size=8).digest(), "little")
        )
        sel = np.sort(rng.choice(sel, size=max_points, replace=False))

    idx_full = np.flatnonzero(keep)[sel]

    x = np.asarray(las.x)[idx_full]
    y = np.asarray(las.y)[idx_full]
    z = np.asarray(las.z)[idx_full]
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    zmin = float(z.min())

    xy_i = np.rint(np.stack([(x - cx), (y - cy)], axis=1) / XY_QUANT)
    z_i = np.rint((z - zmin) / Z_QUANT)
    np.clip(xy_i, -32768, 32767, out=xy_i)
    np.clip(z_i, 0, 32767, out=z_i)

    rn = np.asarray(las.return_number)[idx_full]
    nr = np.asarray(las.number_of_returns)[idx_full]
    rgbn = np.stack([np.asarray(las.red)[idx_full], np.asarray(las.green)[idx_full],
                     np.asarray(las.blue)[idx_full], np.asarray(las.nir)[idx_full]], axis=1)

    lab = _LUT[np.asarray(las.classification)[idx_full]]
    hist = np.bincount(lab.astype(np.int64), minlength=NUM_CLASSES)

    meta = {
        "patch_id": patch_id,
        "n_raw": n_raw,
        "n_dropped_noise": int(n_raw - int(keep.sum())),
        "n_voxel": n_voxel,
        "n_stored": int(len(idx_full)),
        "centroid_x": float(cx), "centroid_y": float(cy), "z_min": zmin,
        "z_max": float(z.max()),
        **{f"stored_n_{i}": int(hist[i]) for i in range(NUM_CLASSES)},
    }
    return PatchArrays(
        xy=xy_i.astype(np.int16), z=z_i.astype(np.int16),
        intensity=np.asarray(las.intensity)[idx_full].astype(np.uint16),
        returns=np.stack([np.clip(rn, 0, 255), np.clip(nr, 0, 255)], axis=1).astype(np.uint8),
        rgbn=(rgbn >> 8).astype(np.uint8), label=lab.astype(np.int8), meta=meta,
    )


FIELDS = ("xy", "z", "intensity", "returns", "rgbn", "label")


def write_shard(patches: list[PatchArrays], shard_dir) -> dict:
    """Write one shard as separate .npy files so it can be memory-mapped.

    Random patch access during active learning touches a few hundred patches
    scattered over the whole store, so the store must support cheap random
    reads rather than bulk loading.
    """
    import json
    from pathlib import Path

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    counts = np.array([len(p.label) for p in patches], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    for field in FIELDS:
        np.save(shard_dir / f"{field}.npy",
                np.concatenate([getattr(p, field) for p in patches]))
    np.save(shard_dir / "offsets.npy", offsets)
    (shard_dir / "meta.json").write_text(
        json.dumps([p.meta for p in patches]), encoding="utf-8")
    return {"n_patches": len(patches), "n_points": int(offsets[-1]),
            "bytes": sum(f.stat().st_size for f in shard_dir.glob("*.npy"))}
