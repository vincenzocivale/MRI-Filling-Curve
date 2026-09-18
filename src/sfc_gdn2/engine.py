from __future__ import annotations

import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .data.dataset import MRIPatchDataset, balanced_split, load_manifests
from .io import seed_all
from .metrics import auc, masked_errors
from .model import MRIProbe


def collate(batch):
    return {"patches": torch.stack([x["patches"] for x in batch]),
            "dataset": [x["dataset"] for x in batch], "sample_id": [x["sample_id"] for x in batch]}


def _mask(batch: int, n: int, ratio: float, device, generator: torch.Generator):
    return torch.rand((batch, n), generator=generator, device=device) < ratio

def prediction_batch(model, x, ratio, device, generator):
    if model.objective == "next_patch":
        target = x[:, model.perm[1:]]
        mask = torch.ones(target.shape[:2], dtype=torch.bool, device=device)
        return model(x), target, mask
    mask = _mask(len(x), x.shape[1], ratio, device, generator)
    return model(x, mask), x, mask


@torch.no_grad()
def evaluate(model, loader, ratio, device, seed):
    model.eval(); by = defaultdict(lambda: [0.,0.,0])
    g = torch.Generator(device=device).manual_seed(seed + 991)
    for batch in loader:
        x = batch["patches"].to(device)
        p, target, m = prediction_batch(model, x, ratio, device, g)
        for ds in set(batch["dataset"]):
            idx = torch.tensor([v == ds for v in batch["dataset"]], device=device)
            mse, mae = masked_errors(p[idx], target[idx], m[idx])
            by[ds][0] += mse; by[ds][1] += mae; by[ds][2] += 1
    rows = [{"dataset": ds, "mse": a/n, "mae": b/n} for ds,(a,b,n) in by.items()]
    return rows


def microtrain(cfg: dict, curve: str):
    seed = int(cfg["seed"]); seed_all(seed)
    data_cfg, mt = cfg["data"], cfg["microtrain"]
    df = load_manifests(cfg["manifests"])
    tr, va = balanced_split(df, seed, data_cfg["max_train_per_dataset"], data_cfg["max_val_per_dataset"])
    ds_tr = MRIPatchDataset(tr, data_cfg["target_shape"], data_cfg["patch_size"], data_cfg["cache_dir"])
    ds_va = MRIPatchDataset(va, data_cfg["target_shape"], data_cfg["patch_size"], data_cfg["cache_dir"])
    loader = DataLoader(ds_tr, batch_size=mt["batch_size"], shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(ds_va, batch_size=mt["batch_size"], shuffle=False, num_workers=0, collate_fn=collate)
    shape = data_cfg["target_shape"]; patch = data_cfg["patch_size"]
    if len(set(shape)) != 1: raise ValueError("Pilot currently requires cubic target_shape.")
    grid = shape[0] // patch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRIProbe(patch**3, grid, curve, seed, cfg["model"], mt.get("objective", "masked")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=mt["lr"], weight_decay=mt["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(mt.get("amp", True) and device.type == "cuda"))
    g = torch.Generator(device=device).manual_seed(seed + 123)
    hist, it = [], iter(loader); t0 = time.time()
    for step in range(1, mt["steps"] + 1):
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)
        x = batch["patches"].to(device)
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler.is_enabled()):
            pred, target, m = prediction_batch(model, x, mt.get("mask_ratio", 0.5), device, g)
            loss = ((pred - target)[m] ** 2).mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % mt["eval_every"] == 0 or step == mt["steps"]:
            rows = evaluate(model, val_loader, mt.get("mask_ratio", 0.5), device, seed)
            hist.append({"step": step, "train_mse": float(loss.item()),
                         "val_mse": sum(r["mse"] for r in rows)/len(rows),
                         "seconds": time.time()-t0})
    final = evaluate(model, val_loader, mt.get("mask_ratio", 0.5), device, seed)
    return model, hist, final, {"val_mse_auc": auc(hist)}
