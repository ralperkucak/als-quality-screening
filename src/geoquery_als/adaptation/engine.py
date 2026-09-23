"""Training, inference and evaluation loops shared by every method.

Acquisition strategies differ only in which target patches they request. The
adaptation schedule, the optimiser, the loss, the replay ratio and the evaluation
protocol are identical for all of them, so any difference in the reported
metrics is attributable to the query strategy.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from ..config import NUM_CLASSES
from ..data.dataset import (HierarchyCache, PatchDataset, SamplingConfig, collate,
                            to_device)
from ..data.loader import ThreadedLoader
from ..data.store import PatchStore
from ..evaluation.metrics import (ConfusionAccumulator, expected_calibration_error,
                                  negative_log_likelihood)
from ..models.randlanet import RandLANet, RandLANetConfig
from ..uncertainty.scores import (mean_posterior, patch_scores,
                                  voxel_label_disparity)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 6
    lr: float = 4e-3
    weight_decay: float = 0.0
    num_workers: int = 6
    grad_clip: float = 5.0
    amp: bool = True
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    class_weighting: str = "sqrt_inv_freq"   # or "none"
    log_every: int = 50
    seed: int = 0


@dataclass
class AdaptConfig:
    epochs: int = 8
    batch_size: int = 6
    lr: float = 5e-4
    replay_ratio: float = 2.0     # source patches per target patch in an epoch
    finetune_mode: str = "last_encoder_decoder_head"
    warm_start: bool = True
    num_workers: int = 5
    amp: bool = True
    grad_clip: float = 5.0


def make_model(cfg: RandLANetConfig, device: torch.device) -> RandLANet:
    return RandLANet(cfg).to(device)


def class_weights_from_counts(counts: np.ndarray, scheme: str) -> torch.Tensor:
    if scheme == "none":
        return torch.ones(NUM_CLASSES, dtype=torch.float32)
    freq = counts / max(counts.sum(), 1)
    with np.errstate(divide="ignore"):
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    w[counts == 0] = 0.0
    w = w / np.maximum(w[counts > 0].mean(), 1e-8)
    return torch.tensor(w, dtype=torch.float32)


# Where the neighbourhood graph is built. The two paths produce the same exact
# graph, so this is purely a performance choice, and on this hardware the host
# wins. Measured on identical batches under identical load: host 21.4 training
# and 70.7 inference patches per second, device 13.3 and 14.8. The device
# version has to materialise the full pairwise distance matrix, which is memory
# bound rather than arithmetic bound, and it competes for the same device that
# the network already saturates. It is kept because a machine with a larger GPU
# and fewer CPU cores would invert that balance.
HIERARCHY_ON_DEVICE = False


def _loader(store: PatchStore, ids: list[str], scfg: SamplingConfig, *, batch_size: int,
            shuffle: bool, augment: bool, workers: int, seed: int = 0,
            cache: HierarchyCache | None = None) -> ThreadedLoader:
    """A thread-pool loader; see `geoquery_als.data.loader` for why not processes."""
    ds = PatchDataset(store, ids, scfg, augment=augment, seed=0, cache=cache,
                      hierarchy_on_device=HIERARCHY_ON_DEVICE)
    return ThreadedLoader(ds, batch_size, collate, shuffle=shuffle, num_threads=workers,
                          drop_last=shuffle and len(ids) > batch_size, seed=seed)


MAX_NONFINITE_STEPS = 25


def train(model: RandLANet, store: PatchStore, train_ids: list[str], val_ids: list[str],
          scfg: SamplingConfig, cfg: TrainConfig, device: torch.device,
          class_counts: np.ndarray, *, parameters=None, log=print,
          cache: HierarchyCache | None = None, resume_path=None) -> dict:
    """Supervised training with source-side validation for the schedule only.

    When `resume_path` is given, the optimiser, schedule, scaler and best-so-far
    state are written there after every epoch and read back on start, so an
    interrupted run loses at most one epoch instead of a whole fold.
    """
    torch.manual_seed(cfg.seed)
    params = parameters if parameters is not None else list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr)
    weights = class_weights_from_counts(class_counts, cfg.class_weighting).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    tl = _loader(store, train_ids, scfg, batch_size=cfg.batch_size, shuffle=True,
                 augment=True, workers=cfg.num_workers, seed=cfg.seed, cache=cache)
    history, best = [], {"miou": -1.0, "state": None, "epoch": -1}
    start_epoch, elapsed_before = 0, 0.0

    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimiser"])
        sched.load_state_dict(ck["schedule"])
        scaler.load_state_dict(ck["scaler"])
        history, best = ck["history"], ck["best"]
        start_epoch = ck["epoch"] + 1
        elapsed_before = ck["elapsed"]
        log(f"  resuming from epoch {start_epoch} "
            f"(best val mIoU so far {best['miou']:.4f})")

    t0 = time.time() - elapsed_before
    for epoch in range(start_epoch, cfg.epochs):
        tl.seed = cfg.seed * 100003 + epoch       # a fresh shuffle every epoch
        model.train()
        running, nb, nonfinite = 0.0, 0, 0
        for step, batch in enumerate(tl):
            batch = to_device(batch, device, scfg)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(batch)
                loss = F.cross_entropy(logits, batch["labels"], weight=weights)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            value = float(loss.detach())
            if not math.isfinite(value):
                nonfinite += 1
                if nonfinite > MAX_NONFINITE_STEPS:
                    raise RuntimeError(
                        f"loss was not finite on {nonfinite} steps by epoch "
                        f"{epoch}; the run is dead and further epochs would "
                        f"only consume time")
            else:
                nonfinite = 0
                running += value
                nb += 1
        val = evaluate(model, store, val_ids, scfg, device, batch_size=cfg.batch_size,
                       workers=cfg.num_workers, project_to_full=False, cache=cache)
        sched.step(val["mIoU"])
        history.append({"epoch": epoch, "train_loss": running / max(nb, 1),
                        "val_mIoU": val["mIoU"], "lr": opt.param_groups[0]["lr"],
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {epoch:3d} loss {running/max(nb,1):.4f} "
            f"val mIoU {val['mIoU']:.4f} lr {opt.param_groups[0]['lr']:.2e} "
            f"[{time.time()-t0:.0f}s]")
        if val["mIoU"] > best["miou"]:
            best = {"miou": val["mIoU"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if resume_path is not None:
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"model": model.state_dict(), "optimiser": opt.state_dict(),
                        "schedule": sched.state_dict(), "scaler": scaler.state_dict(),
                        "history": history, "best": best, "epoch": epoch,
                        "elapsed": time.time() - t0}, tmp)
            tmp.replace(resume_path)      # atomic, so a crash cannot corrupt it
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"history": history, "best_val_mIoU": best["miou"], "best_epoch": best["epoch"],
            "wall_seconds": round(time.time() - t0, 1), "config": asdict(cfg)}


@torch.no_grad()
def evaluate(model: RandLANet, store: PatchStore, ids: list[str], scfg: SamplingConfig,
             device: torch.device, *, batch_size: int = 6, workers: int = 4,
             project_to_full: bool = True, votes: int = 1,
             cache: HierarchyCache | None = None) -> dict:
    """Accumulate a point-level confusion matrix over the evaluation clouds.

    When `project_to_full` is set, the network's prediction on its subsampled
    input is transferred to every stored point of the patch by nearest-neighbour
    assignment in 3D, which is the inference protocol of the original
    RandLA-Net. Otherwise the confusion matrix is accumulated on the subsample.
    """
    model.eval()
    acc = ConfusionAccumulator()
    conf_all, corr_all, nll_terms = [], [], []
    for v in range(votes):
        dl = _loader(store, ids, scfg, batch_size=batch_size, shuffle=False, augment=False,
                     workers=workers, seed=v, cache=cache)
        for batch in dl:
            gpu = to_device(batch, device, scfg)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(gpu)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()   # (B, C, N)
            pred = probs.argmax(axis=1)
            lab = batch["labels"].numpy()
            pidx = batch["point_index"].numpy()
            for b in range(len(pred)):
                pid = ids[int(batch["patch_index"][b])]
                if project_to_full:
                    hit = cache.get_projection(pid) if cache is not None else None
                    if hit is None:
                        xyz_full, lab_full = store.read_xyz_label(pid)
                        _, nn = cKDTree(xyz_full[pidx[b]]).query(xyz_full, k=1, workers=-1)
                        nn = nn.astype(np.int32)
                        if cache is not None:
                            cache.put_projection(pid, (nn, lab_full.astype(np.int8)))
                    else:
                        nn, lab_full = hit
                    acc.update(lab_full, pred[b][nn])
                else:
                    acc.update(lab[b], pred[b])
                p_true = probs[b][lab[b], np.arange(probs.shape[2])]
                conf_all.append(probs[b].max(axis=0))
                corr_all.append((pred[b] == lab[b]).astype(np.float64))
                nll_terms.append(p_true)
    out = acc.as_dict()
    conf = np.concatenate(conf_all) if conf_all else np.zeros(1)
    corr = np.concatenate(corr_all) if corr_all else np.zeros(1)
    out["ECE"] = expected_calibration_error(conf, corr)
    out["NLL"] = negative_log_likelihood(np.concatenate(nll_terms)) if nll_terms else None
    out["n_patches"] = len(ids)
    return out


@torch.no_grad()
def infer_pool(model: RandLANet, store: PatchStore, ids: list[str], scfg: SamplingConfig,
               device: torch.device, *, batch_size: int = 8, workers: int = 5,
               top_q: float = 0.1, cache: HierarchyCache | None = None
               ) -> dict[str, np.ndarray]:
    """Per-patch uncertainty scores, mean posterior and pooled embedding.

    Labels are read by the data loader but never returned here: this function is
    the only bridge between the model and the acquisition step, and it emits no
    label-derived quantity.
    """
    model.eval()
    dl = _loader(store, ids, scfg, batch_size=batch_size, shuffle=False, augment=False,
                 workers=workers, seed=0, cache=cache)
    n = len(ids)
    scores = {k: np.zeros(n) for k in
              ("entropy_mean", "entropy_topq", "margin", "least_confidence",
               "voxel_disparity")}
    post = np.zeros((n, NUM_CLASSES))
    emb = None
    t0 = time.time()
    for batch in dl:
        gpu = to_device(batch, device, scfg)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, e = model(gpu, return_embedding=True)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        pooled = torch.nn.functional.normalize(e.float().mean(dim=2), dim=1).cpu().numpy()
        if emb is None:
            emb = np.zeros((n, pooled.shape[1]), dtype=np.float32)
        xyz_m = (batch["features"][:, :3].numpy().transpose(0, 2, 1)
                 * scfg.coord_scale_m)                       # (B, N, 3) in metres
        pred = probs.argmax(axis=1)                          # (B, N)
        for b in range(probs.shape[0]):
            i = int(batch["patch_index"][b])
            p = probs[b].T                                   # (N, C)
            for k, v in patch_scores(p, top_q).items():
                scores[k][i] = v
            scores["voxel_disparity"][i] = voxel_label_disparity(xyz_m[b], pred[b])
            post[i] = mean_posterior(p)
            emb[i] = pooled[b]
    return {"scores": scores, "mean_posterior": post, "embedding": emb,
            "seconds": round(time.time() - t0, 2)}
