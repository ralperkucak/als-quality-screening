"""Phase 6: train one source-only model per leave-one-domain-out fold.

No patch of the target domain, and no target label, enters this stage. The
resulting checkpoint is the zero-budget starting point shared by every
acquisition method, so it is trained once per fold and reused.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoquery_als.config import MANIFEST_DIR, MODEL_DIR, NUM_CLASSES, OUTPUT_DIR  # noqa: E402
from geoquery_als.adaptation.engine import TrainConfig, evaluate, train  # noqa: E402
from geoquery_als.data.dataset import HierarchyCache, SamplingConfig  # noqa: E402
from geoquery_als.data.store import PatchStore  # noqa: E402
from geoquery_als.models import (backbone_suffix, variant_index, build_backbone,  # noqa: E402
                                 count_parameters)


def class_counts_for(store: PatchStore, ids: list[str]) -> np.ndarray:
    idx = store.index.set_index("patch_id")
    cols = [f"stored_n_{c}" for c in range(NUM_CLASSES)]
    return idx.loc[ids, cols].to_numpy().sum(axis=0).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--geometry-only", action="store_true")
    # A "<backbone>_sN" name repeats the same architecture from a different
    # initialisation, for the seed sensitivity run.
    ap.add_argument("--backbone", default="randlanet")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    conf = yaml.safe_load((root / "configs/experiment/main.yaml").read_text(encoding="utf-8"))
    splits = pd.read_parquet(MANIFEST_DIR / "lodo_splits.parquet")
    store = PatchStore()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scfg = SamplingConfig(**{**conf["sampling"],
                            "geometry_only": args.geometry_only or conf["sampling"]["geometry_only"]})
    tcfg_kwargs = dict(conf["source_training"])
    if args.epochs is not None:
        tcfg_kwargs["epochs"] = args.epochs
    suffix = ("_geomonly" if scfg.geometry_only else "") + backbone_suffix(args.backbone)

    folds = args.folds or sorted(splits["fold"].unique())
    out_dir = MODEL_DIR / "source_lodo"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "metrics" / f"source_only{suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    for fold in folds:
        ckpt = out_dir / f"{fold}{suffix}.pt"
        if ckpt.exists():
            print(f"[{fold}] checkpoint exists, skipping")
            continue
        sel = splits[splits["fold"] == fold]
        st = sel[sel.role == "source_train"]["patch_id"].tolist()
        sv = sel[sel.role == "source_val"]["patch_id"].tolist()
        tt = sel[sel.role == "target_test"]["patch_id"].tolist()
        counts = class_counts_for(store, st)
        print(f"[{fold}] source train {len(st)}  source val {len(sv)}  target test {len(tt)}")
        print(f"[{fold}] source class point counts: {counts.astype(int).tolist()}")

        seed = conf["seed_base"] + 10_000 * variant_index(args.backbone)
        torch.manual_seed(seed)
        model = build_backbone(args.backbone, conf["model"]).to(device)
        if fold == folds[0]:
            print(f"{args.backbone}: {count_parameters(model):,} trainable parameters")

        tcfg = TrainConfig(seed=seed % (2 ** 31), **tcfg_kwargs)
        cache = HierarchyCache.for_working_set(len(st) + len(sv) + len(tt))
        t0 = time.time()
        resume = out_dir / f".{fold}{suffix}.resume"
        hist = train(model, store, st, sv, scfg, tcfg, device, counts, cache=cache,
                     resume_path=resume)
        met = evaluate(model, store, tt, scfg, device, batch_size=tcfg.batch_size,
                       workers=tcfg.num_workers, project_to_full=True, cache=cache)
        print(f"[{fold}] neighbourhood cache: {cache.stats()}")
        cache.clear()
        torch.save({"state_dict": model.state_dict(), "config": conf,
                    "fold": fold, "geometry_only": scfg.geometry_only,
                    "backbone": args.backbone,
                    "class_counts": counts.tolist()}, ckpt)
        resume.unlink(missing_ok=True)     # the fold is complete
        summary[fold] = {"training": hist, "target_test": met,
                         "wall_seconds": round(time.time() - t0, 1),
                         "n_source_train": len(st), "n_source_val": len(sv),
                         "n_target_test": len(tt)}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{fold}] source-only target mIoU {met['mIoU']:.4f}  OA {met['OA']:.4f}  "
              f"({(time.time()-t0)/60:.0f} min)")
        print(f"[{fold}] per-class IoU: "
              + ", ".join(f"{k} {v:.3f}" for k, v in met["per_class_IoU"].items()
                          if v is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
