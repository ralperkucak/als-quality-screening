"""Source-side pass: class prototypes and the pairs every calibrator is fitted on.

This is the only place the reliability estimator learns anything, and it learns
it entirely from labelled source validation data. No target patch is read here.

Two things come out of the pass. Class prototypes are accumulated exactly, as
weighted sums over every correctly predicted full point, so nothing per point
has to be written to get them. Alongside them a seeded subsample of points is
kept with its embedding, confidence, entropy, predicted class and correctness,
which is what the class-wise calibrators and the empirical distance distribution
are fitted on later. A subsample suffices there because those are
one-dimensional distributions per class, while a prototype is a mean in 32
dimensions and is worth having exact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoquery_als.config import MODEL_DIR, NUM_CLASSES, OUTPUT_DIR  # noqa: E402
from geoquery_als.adaptation.engine import _loader  # noqa: E402
from geoquery_als.data.dataset import HierarchyCache, SamplingConfig, to_device  # noqa: E402
from geoquery_als.data.store import PatchStore  # noqa: E402
from geoquery_als.models import (backbone_suffix,  # noqa: E402
                                 build_backbone)
from geoquery_als.reliability.statistics import (PrototypeAccumulator,  # noqa: E402
                                                 point_entropy)

POINTS_PER_PATCH = 200        # calibration subsample; 240,000 points per fold


def build_model(conf: dict, device: torch.device, backbone='randlanet'):
    return build_backbone(backbone, conf["model"]).to(device)


@torch.no_grad()
def source_pass(model, store: PatchStore, ids: list[str],
                scfg: SamplingConfig, device: torch.device, cache: HierarchyCache,
                seed: int, workers: int) -> tuple[PrototypeAccumulator, dict]:
    """Accumulate prototypes and collect the calibration subsample."""
    model.eval()
    proto = PrototypeAccumulator(dim=32)
    keep = {k: [] for k in ("emb", "conf", "entropy", "pred", "correct", "weight")}
    rng = np.random.default_rng(seed)
    dl = _loader(store, ids, scfg, batch_size=8, shuffle=False, augment=False,
                 workers=workers, seed=0, cache=cache)
    done = 0
    for batch in dl:
        gpu = to_device(batch, device, scfg)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, emb = model(gpu, return_embedding=True)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()      # (B, C, N)
        embs = emb.float().cpu().numpy()                                # (B, D, N)
        labs = batch["labels"].numpy()
        pidx = batch["point_index"].numpy()
        for b in range(probs.shape[0]):
            pid = ids[int(batch["patch_index"][b])]
            hit = cache.get_projection(pid)
            if hit is None:
                xyz_full, lab_full = store.read_xyz_label(pid)
                _, nn = cKDTree(xyz_full[pidx[b]]).query(xyz_full, k=1, workers=-1)
                nn = nn.astype(np.int32)
                cache.put_projection(pid, (nn, lab_full.astype(np.int8)))
            else:
                nn, _ = hit
            weight = np.bincount(nn, minlength=probs.shape[2]).astype(np.float64)
            p = probs[b]
            pred = p.argmax(axis=0)
            e = embs[b].T                                               # (N, D)
            lab = labs[b]
            proto.update(e, pred, lab, weight)

            take = rng.choice(len(pred), size=min(POINTS_PER_PATCH, len(pred)),
                              replace=False)
            keep["emb"].append(e[take].astype(np.float16))
            keep["conf"].append(p.max(axis=0)[take].astype(np.float16))
            keep["entropy"].append(point_entropy(p)[take].astype(np.float16))
            keep["pred"].append(pred[take].astype(np.uint8))
            keep["correct"].append((pred[take] == lab[take]).astype(np.uint8))
            keep["weight"].append(weight[take].astype(np.float32))
            done += 1
    return proto, {k: np.concatenate(v) for k, v in keep.items()}, done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--backbone", default="randlanet")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    conf = yaml.safe_load((root / "configs/experiment/main.yaml").read_text("utf-8"))
    scfg = SamplingConfig(**conf["sampling"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = PatchStore()

    members = pd.read_parquet(OUTPUT_DIR / "pivot" / "units" / "unit_members.parquet")
    src = members[members.family == "source"]
    out_dir = OUTPUT_DIR / "pivot" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in (args.folds or sorted(src["fold"].unique())):
        sfx = backbone_suffix(args.backbone)
        path = out_dir / f"source_{fold}{sfx}.npz"
        if path.exists() and not args.force:
            print(f"[{fold}] already done")
            continue
        ids = sorted(src[src.fold == fold]["patch_id"].unique().tolist())
        ckpt = MODEL_DIR / "source_lodo" / f"{fold}{sfx}.pt"
        model = build_model(conf, device, args.backbone)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)["state_dict"])
        cache = HierarchyCache.for_working_set(len(ids))
        t0 = time.time()
        proto, sample, done = source_pass(model, store, ids, scfg, device, cache,
                                          seed=conf["seed_base"] + 3,
                                          workers=args.workers)
        prototypes = proto.prototypes()
        np.savez_compressed(
            path, prototypes=prototypes, proto_count=proto.count,
            **{f"sample_{k}": v for k, v in sample.items()})
        cache.clear()
        del model
        torch.cuda.empty_cache()
        seen = int(np.isfinite(prototypes[:, 0]).sum())
        print(f"[{fold}] {done} patches, {len(sample['conf']):,} calibration points, "
              f"{seen}/{NUM_CLASSES} classes with a prototype, "
              f"accuracy {sample['correct'].mean():.3f} "
              f"({(time.time() - t0) / 60:.1f} min)")
        print("      prototype support per class: "
              + " ".join(f"{int(c):,}" for c in proto.count))

    (out_dir / "meta.json").write_text(json.dumps(
        {"points_per_patch": POINTS_PER_PATCH, "embedding_dim": 32,
         "note": "fitted on labelled source validation only; no target patch read"},
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
