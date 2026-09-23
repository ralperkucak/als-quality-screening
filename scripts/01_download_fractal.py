"""Phase 1: stream FRACTAL from the Hugging Face Hub into the local point store.

Each archive is downloaded, checksummed, converted to a compact shard and then
deleted, so peak disk use stays at one archive plus the shard being written.
The script is restartable: completed shards are skipped.

Shards are numbered in the repository's own archive order. Because the archives
interleave patches from every region (verified in outputs/audit/shard_audit.json),
any prefix of the archive list is an unbiased random sample of the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoquery_als.config import PROCESSED_DIR, RAW_DIR, OUTPUT_DIR  # noqa: E402
from geoquery_als.data.preprocess import preprocess_laz, write_shard  # noqa: E402

BASE = "https://huggingface.co/datasets/IGNF/FRACTAL/resolve/main/data/{split}/{name}"
N_ARCHIVES = {"train": 80, "val": 10, "test": 10}
# Test patches carry the reported metrics, so they are stored at the full
# 0.25 m voxel resolution; train/val patches are thinned to fit on disk.
MAX_POINTS = {"train": 16384, "val": 16384, "test": 40960}

STORE = PROCESSED_DIR / "patches"


def _one(args):
    name, data, split = args
    try:
        return preprocess_laz(data, Path(name).stem.split("-", 1)[1], MAX_POINTS[split])
    except Exception as exc:  # noqa: BLE001
        return (name, f"{type(exc).__name__}: {exc}")


def process_archive(zip_path: Path, split: str, shard_dir: Path, workers: int) -> dict:
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        payload = [(n, zf.read(n), split) for n in names]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, payload, chunksize=8))

    ok = [r for r in results if not isinstance(r, tuple)]
    bad = [r for r in results if isinstance(r, tuple)]
    ok.sort(key=lambda p: p.meta["patch_id"])
    stats = write_shard(ok, shard_dir)
    return {
        "shard": shard_dir.name, "split": split, "n_failed": len(bad),
        "failures": bad[:20], "seconds": round(time.time() - t0, 1), **stats,
    }


def download(url: str, dest: Path, *, attempts: int = 8) -> str:
    """Download with resume and backoff.

    The ingestion runs for hours over a slow link, so a dropped connection has
    to be survivable: a partial file is resumed with a range request rather than
    restarted, and only a genuinely unreachable host ends the attempt.
    """
    # A partial file left by an interrupted run may already be complete; asking
    # for a range past the end returns 416 rather than a helpful answer.
    try:
        head = requests.head(url, timeout=60, allow_redirects=True)
        remote_size = int(head.headers.get("Content-Length", 0))
    except Exception:                                                # noqa: BLE001
        remote_size = 0
    if remote_size and dest.exists() and dest.stat().st_size >= remote_size:
        if dest.stat().st_size > remote_size:
            dest.unlink()
        else:
            h = hashlib.sha256()
            with dest.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

    last_error: Exception | None = None
    for attempt in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers=headers) as r:
                if have and r.status_code == 200:
                    have = 0                      # server ignored the range request
                elif have and r.status_code != 206:
                    r.raise_for_status()
                else:
                    r.raise_for_status()
                mode = "ab" if have else "wb"
                with dest.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except Exception as exc:                                     # noqa: BLE001
            last_error = exc
            if "416" in str(exc) and dest.exists():
                dest.unlink()                    # partial file no longer usable
            wait = min(60, 5 * 2 ** attempt)
            print(f"    download attempt {attempt + 1}/{attempts} failed "
                  f"({type(exc).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError(f"download failed after {attempts} attempts: {last_error}")

    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max archives per split (default: all)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    STORE.mkdir(parents=True, exist_ok=True)
    zips = RAW_DIR / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "audit" / "shard_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        n = N_ARCHIVES[split] if args.limit is None else min(args.limit, N_ARCHIVES[split])
        for i in range(n):
            name = f"{split}-{i:02d}.zip"
            shard_dir = STORE / f"{split}-{i:02d}"
            if (shard_dir / "meta.json").exists() and (shard_dir / "label.npy").exists():
                continue
            zp = zips / name
            t0 = time.time()
            sha = download(BASE.format(split=split, name=name), zp)
            dl = time.time() - t0
            rec = process_archive(zp, split, shard_dir, args.workers)
            rec.update({"sha256_zip": sha, "download_seconds": round(dl, 1),
                        "zip_bytes": zp.stat().st_size})
            zp.unlink()
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{name}: {rec['n_patches']} patches, {rec['n_points']/1e6:.1f} Mpts, "
                  f"{rec['bytes']/1e6:.0f} MB, dl {dl:.0f}s, proc {rec['seconds']}s, "
                  f"fail {rec['n_failed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
