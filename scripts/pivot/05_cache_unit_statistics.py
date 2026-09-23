"""Per-patch sufficient statistics for every unit, source and target.

One inference pass per fold and family. The source-fitted prototypes, calibrator
and support distribution are loaded and applied unchanged; nothing is estimated
here. The reference labels are read only to build the per-patch confusion matrix,
which is the evaluation target and never reaches a feature; the separation is a
contract asserted in `tests/pivot/test_no_target_label_in_features.py`.

The source family is passed over as well as the target because the mapping from
a reliability score to an expected mIoU has to be fitted on units whose true mIoU
is known without touching the target, and source validation units are the only
such units in the primary setting.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from geoquery_als.config import MODEL_DIR, OUTPUT_DIR  # noqa: E402
from geoquery_als.adaptation.engine import _loader  # noqa: E402
from geoquery_als.data.dataset import HierarchyCache, SamplingConfig, to_device  # noqa: E402
from geoquery_als.data.store import PatchStore  # noqa: E402
from geoquery_als.models import (backbone_suffix,  # noqa: E402
                                 build_backbone)
from geoquery_als.reliability.calibration import (ClasswiseCalibrator,  # noqa: E402
                                                  SupportDistribution)
from geoquery_als.reliability.statistics import (accumulate_patch,  # noqa: E402
                                                 cosine_distance_to_prototype)

FIELDS = ("n_pred", "sum_conf", "sum_conf_sq", "sum_entropy", "sum_calibrated",
          "sum_support", "sum_reliability")


def build_model(conf: dict, device: torch.device, backbone='randlanet'):
    return build_backbone(backbone, conf["model"]).to(device)


def make_support_fn(prototypes: np.ndarray, support: SupportDistribution):
    """Turn a raw embedding into the source-empirical support score."""
    def fn(emb: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return support(cosine_distance_to_prototype(emb, prototypes, pred), pred)
    return fn


@torch.no_grad()
def run_pass(model, store: PatchStore, ids: list[str],
             scfg: SamplingConfig, device: torch.device, cache: HierarchyCache,
             prototypes: np.ndarray, calibrator, support, workers: int) -> dict:
    model.eval()
    out = {k: [] for k in ("patch_id", "n_full", "confusion", "conf_hist",
                           "support_hist", *FIELDS)}
    support_fn = make_support_fn(prototypes, support)
    dl = _loader(store, ids, scfg, batch_size=8, shuffle=False, augment=False,
                 workers=workers, seed=0, cache=cache)
    for batch in dl:
        gpu = to_device(batch, device, scfg)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, emb = model(gpu, return_embedding=True)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        embs = emb.float().cpu().numpy()
        pidx = batch["point_index"].numpy()
        for b in range(probs.shape[0]):
            pid = ids[int(batch["patch_index"][b])]
            hit = cache.get_projection(pid)
            if hit is None:
                xyz_full, lab_full = store.read_xyz_label(pid)
                _, nn = cKDTree(xyz_full[pidx[b]]).query(xyz_full, k=1, workers=-1)
                nn = nn.astype(np.int32)
                lab_full = lab_full.astype(np.int8)
                cache.put_projection(pid, (nn, lab_full))
            else:
                nn, lab_full = hit
            st = accumulate_patch(pid, probs[b], embs[b], nn, lab_full,
                                  prototypes=None, calibrator=calibrator)
            # The support term needs the empirical source distribution, which
            # accumulate_patch cannot know about, so it is applied here and the
            # two dependent fields are recomputed.
            pred = probs[b].argmax(axis=0)
            weight = np.bincount(nn, minlength=probs.shape[2]).astype(np.float64)
            sup = support_fn(embs[b].T, pred)
            cal = calibrator(probs[b].max(axis=0), pred)
            rel = np.sqrt(np.clip(cal, 0, 1) * sup)
            st.sum_support = _class_sums(sup, pred, weight)
            st.sum_reliability = _class_sums(rel, pred, weight)
            st.support_hist = _hist(sup, pred, weight)

            out["patch_id"].append(pid)
            out["n_full"].append(st.n_full)
            out["confusion"].append(st.confusion)
            out["conf_hist"].append(st.conf_hist.astype(np.float32))
            out["support_hist"].append(st.support_hist.astype(np.float32))
            for f in FIELDS:
                out[f].append(getattr(st, f))
    return {k: (np.asarray(v) if k != "patch_id" else np.asarray(v, dtype=object))
            for k, v in out.items()}


def _class_sums(values: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> np.ndarray:
    from geoquery_als.config import NUM_CLASSES
    out = np.zeros(NUM_CLASSES)
    np.add.at(out, pred, values * weight)
    return out


def _hist(values: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> np.ndarray:
    from geoquery_als.reliability.statistics import weighted_histogram
    return weighted_histogram(values, pred, weight, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--backbone", default="randlanet")
    ap.add_argument("--families", nargs="*", default=["source", "target"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    conf = yaml.safe_load((root / "configs/experiment/main.yaml").read_text("utf-8"))
    scfg = SamplingConfig(**conf["sampling"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = PatchStore()

    members = pd.read_parquet(OUTPUT_DIR / "pivot" / "units" / "unit_members.parquet")
    cal_dir = OUTPUT_DIR / "pivot" / "calibration"
    out_dir = OUTPUT_DIR / "pivot" / "unit_statistics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in (args.folds or sorted(members["fold"].unique())):
        sfx = backbone_suffix(args.backbone)
        fitted_path = cal_dir / f"fitted_{fold}{sfx}.npz"
        if not fitted_path.exists():
            print(f"[{fold}] no fitted calibration, run script 04 first")
            continue
        fitted = np.load(fitted_path)
        prototypes = fitted["prototypes"]
        calibrator = ClasswiseCalibrator.from_npz(fitted)
        support = SupportDistribution.from_npz(fitted)

        model = None
        for family in args.families:
            path = out_dir / f"{family}_{fold}{sfx}.npz"
            if path.exists() and not args.force:
                print(f"[{fold}/{family}] already done")
                continue
            ids = sorted(members[(members.fold == fold)
                                 & (members.family == family)]["patch_id"].unique())
            if not ids:
                continue
            if model is None:
                model = build_model(conf, device, args.backbone)
                model.load_state_dict(torch.load(
                    MODEL_DIR / "source_lodo" / f"{fold}{sfx}.pt", map_location=device,
                    weights_only=False)["state_dict"])
            cache = HierarchyCache.for_working_set(len(ids))
            t0 = time.time()
            res = run_pass(model, store, ids, scfg, device, cache, prototypes,
                           calibrator, support, args.workers)
            np.savez_compressed(path, **res)
            cache.clear()
            oa = np.stack(res["confusion"]).sum(axis=0)
            print(f"[{fold}/{family}] {len(ids)} patches, "
                  f"{res['n_full'].sum():,} points, "
                  f"OA {np.diag(oa).sum() / oa.sum():.4f}, "
                  f"mean support {res['sum_support'].sum() / res['n_full'].sum():.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)")
        if model is not None:
            del model
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
