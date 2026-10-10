from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .volume import VolumeStore, patch_dims, patch_grid, patchify


def cohort_key(df: pd.DataFrame) -> pd.Series:
    """Per-source grouping key: `cohort` when the scanner emitted one, else `dataset`."""
    return df["cohort"] if "cohort" in df.columns else df["dataset"]


class PatchDataset(Dataset):
    """Scans as their native voxels (stored dtype; `to_device` makes the values and the [N, P] patches,
    P = k0*k1*k2, k = the voxels per axis closest to `patch_mm`: `patch_dims`), with spacing (mm), k,
    patch grid [3], value `affine` (a, c) and the row index (callers look up per-row labels themselves)
    and, optionally, `target(row)` computed in the worker (e.g. per-patch segmentation labels)."""

    def __init__(self, rows: pd.DataFrame, store: VolumeStore, patch_mm: float,
                 target: Callable[[pd.Series], torch.Tensor] | None = None):
        self.rows = rows.reset_index(drop=True)
        self.paths = self.rows["path"].tolist()
        self.store, self.patch_mm, self.target = store, patch_mm, target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        arr, spacing, affine = self.store.load(self.paths[i])
        k = patch_dims(spacing, self.patch_mm)
        item = {"volume": torch.from_numpy(np.array(arr)), "grid": patch_grid(arr.shape, k), "k": torch.tensor(k),
                "spacing": torch.tensor(spacing, dtype=torch.float32),
                "affine": torch.tensor(affine, dtype=torch.float64), "index": i}
        if self.target is not None:
            item["target"] = self.target(self.rows.iloc[i])
        return item


def collate(items: list[dict]) -> dict:
    """Volumes stay a list (shapes differ); per-patch targets padded (0 = background) to the batch's
    largest grid."""
    out = {k: torch.stack([it[k] for it in items]) for k in ("grid", "k", "spacing", "affine")}
    out |= {"volumes": [it["volume"] for it in items], "index": torch.tensor([it["index"] for it in items])}
    if "target" in items[0]:
        out["target"] = torch.nn.utils.rnn.pad_sequence([it["target"] for it in items], batch_first=True)
    return out


def to_device(batch: dict, device) -> dict:
    """Stored voxels -> values a * raw + c (fp32: exact for 16-bit scans) on the device -> per scan
    [N_b, P_b] patches (end padding = 0), with grid, k, spacing."""
    patches = []
    for v, (a, c), k in zip(batch["volumes"], batch["affine"].tolist(), batch["k"].tolist()):
        patches.append(patchify(v.to(device, non_blocking=True).float() * a + c, k))
    return {"patches": patches, **{k: batch[k].to(device) for k in ("grid", "k", "spacing")}}


def per_patch(vols: dict, n_patch: int, reduce) -> torch.Tensor:
    """reduce([N_b, P_b]) -> [N_b] per scan, padded with 0 to [B, n_patch]."""
    out = torch.zeros(len(vols["patches"]), n_patch, device=vols["grid"].device)
    for i, x in enumerate(vols["patches"]):
        out[i, : len(x)] = reduce(x)
    return out


class BudgetBatches:
    """Shuffled (seed + epoch) batches of whole scans filled up to `budget` patches (NaViT-style packing) and
    at most `max_batch` scans; a scan larger than the budget gets a batch of its own. Last partial batch dropped."""

    def __init__(self, sizes: list[int], budget: int, max_batch: int, seed: int):
        self.sizes, self.budget, self.max_batch, self.seed, self.epoch = sizes, budget, max_batch, seed, 0

    def __iter__(self):
        order = torch.randperm(len(self.sizes), generator=torch.Generator().manual_seed(self.seed + self.epoch))
        self.epoch += 1
        batch, total = [], 0
        for i in order.tolist():
            if batch and (total + self.sizes[i] > self.budget or len(batch) == self.max_batch):
                yield batch
                batch, total = [], 0
            batch.append(i)
            total += self.sizes[i]

    def __len__(self) -> int:
        return sum(1 for _ in BudgetBatches(self.sizes, self.budget, self.max_batch, self.seed))


def loader(rows: pd.DataFrame, data: dict, batch_size: int, train: bool, seed: int = 0,
           **ds_kwargs) -> DataLoader:
    """Training: shuffled (seeded), worker-persistent; with `data.patches_per_batch`, batches of up to batch_size
    whole scans holding at most that many patches. Evaluation: ordered, batch_size scans."""
    store = VolumeStore(data["cache_dir"])
    ds = PatchDataset(rows, store, data["patch_mm"], **ds_kwargs)
    workers = int(data.get("num_workers", 8))
    common = {"num_workers": workers, "collate_fn": collate, "pin_memory": True,
              "persistent_workers": train and workers > 0, "prefetch_factor": 4 if workers else None}
    if train and data.get("patches_per_batch"):
        sizes = []
        for p in ds.paths:
            arr, spacing, _ = store.load(p)
            sizes.append(int(patch_grid(arr.shape, patch_dims(spacing, data["patch_mm"])).prod()))
        return DataLoader(ds, batch_sampler=BudgetBatches(sizes, int(data["patches_per_batch"]), batch_size, seed),
                          **common)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train,
                      generator=torch.Generator().manual_seed(seed), **common)
