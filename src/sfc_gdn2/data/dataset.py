from __future__ import annotations

import itertools
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .volume import VolumeStore, patch_dims, patch_grid, patchify


class PatchDataset(Dataset):
    """Scans as their native voxels (stored dtype; `to_device` makes the values and the [N, P] patches,
    P = k0*k1*k2, k = the voxels per axis closest to `patch_mm`: `patch_dims`), with spacing (mm), k,
    patch grid [3], value `affine` (a, c) and the row index (callers look up per-row labels themselves)
    and, optionally, `target(row)` computed in the worker (e.g. per-patch segmentation labels).
    With `sizes` (training): a scan that fails to load is logged and replaced by a random scan of at most its
    patches (the batch keeps its budget), so one corrupt file of ~1M does not stop the run."""

    def __init__(self, rows: pd.DataFrame, store: VolumeStore, patch_mm: float,
                 target: Callable[[pd.Series], torch.Tensor] | None = None, sizes: list[int] | None = None):
        self.rows = rows.reset_index(drop=True)
        self.paths = self.rows["path"].tolist()
        self.frames = self.rows["frames"].tolist() if "frames" in self.rows else None
        self.store, self.patch_mm, self.target, self.sizes = store, patch_mm, target, sizes

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, i: int):
        # 4D+ scans (pool column `frames`): a random frame per access (torch RNG: seeded per worker by the loader)
        n = int(self.frames[i]) if self.frames is not None else 1
        return self.store.load(self.paths[i], int(torch.randint(n, ())) if n > 1 else None)

    def __getitem__(self, i: int) -> dict:
        for _ in range(10):
            try:
                arr, spacing, affine = self._load(i)
                break
            except Exception as e:  # unreadable scan: logged, replaced
                if self.sizes is None:
                    raise
                print(f"[data] skip {self.paths[i]}: {e!r}", file=sys.stderr, flush=True)
                fits = np.flatnonzero(np.asarray(self.sizes) <= self.sizes[i])
                i = int(fits[torch.randint(len(fits), ())])
        else:
            raise RuntimeError(f"10 unreadable scans in a row, last {self.paths[i]}")
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
    at most `max_batch` scans; a scan larger than the budget gets a batch of its own. Last partial batch dropped.
    With `weights`: an epoch is len(sizes) scans drawn with replacement in proportion to them (mixing).
    One endless stream for every rank: epochs (seed + epoch) back to back, rank r of `world` takes batches
    r, r + world, ...; the first `start` batches of the stream are skipped. A step consumes `world` batches, so a
    run resumes on any number of GPUs from the count consumed so far.
    With `groups` (session per scan): each drawn scan comes with a random other scan of its group in the same batch
    (alone if it has none, or if the two exceed the budget)."""

    def __init__(self, sizes: list[int], budget: int, max_batch: int, seed: int,
                 weights: torch.Tensor | None = None, rank: int = 0, world: int = 1, groups=None):
        self.sizes, self.budget, self.max_batch, self.seed, self.weights = sizes, budget, max_batch, seed, weights
        self.rank, self.world, self.start = rank, world, 0
        self.groups = None if groups is None else torch.tensor(groups)
        if self.groups is not None:  # scans sorted by group: a group's scans are members[first : first + count]
            self.members = torch.argsort(self.groups, stable=True)
            self.count = torch.bincount(self.groups)
            self.first = self.count.cumsum(0) - self.count
            self.slot = torch.empty_like(self.groups)
            self.slot[self.members] = torch.arange(len(self.groups)) - self.first[self.groups[self.members]]

    def batches(self, epoch: int):
        g = torch.Generator().manual_seed(self.seed + epoch)
        order = (torch.randperm(len(self.sizes), generator=g) if self.weights is None
                 else torch.multinomial(self.weights, len(self.sizes), replacement=True, generator=g))
        units = [[i] for i in order.tolist()]
        if self.groups is not None:
            grp = self.groups[order]
            n = self.count[grp]
            j = (torch.rand(len(order), generator=g) * (n - 1)).long()
            j += j >= self.slot[order]                                    # any member but the scan itself
            mate = self.members[self.first[grp] + j.minimum(n - 1)]
            units = [[i, m] if k > 1 and self.sizes[i] + self.sizes[m] <= self.budget else [i]
                     for i, m, k in zip(order.tolist(), mate.tolist(), n.tolist())]
        batch, total = [], 0
        for u in units:
            size = sum(self.sizes[i] for i in u)
            if batch and (total + size > self.budget or len(batch) + len(u) > self.max_batch):
                yield batch
                batch, total = [], 0
            batch += u
            total += size

    def __iter__(self):
        stream = itertools.chain.from_iterable(self.batches(e) for e in itertools.count())
        return itertools.islice(stream, self.start + self.rank, None, self.world)


def mix_weights(rows: pd.DataFrame, by: list[str], alpha: float) -> torch.Tensor | None:
    """Per-scan sampling weights giving each stratum (`by`, e.g. dataset x modality) of n scans a share
    proportional to n^alpha: 1 = as in the pool (None), 0 = equal strata."""
    if alpha == 1:
        return None
    n = rows.groupby(by, dropna=False)[by[0]].transform("size").to_numpy(dtype=np.float64)
    return torch.from_numpy(n ** (alpha - 1))


def loader(rows: pd.DataFrame, data: dict, batch_size: int, train: bool, seed: int = 0, rank: int = 0,
           world: int = 1, **ds_kwargs) -> DataLoader:
    """Training: endless `BudgetBatches` over all rows, this rank's share (`rank` of `world`), worker-persistent;
    with `data.patches_per_batch`,
    batches of up to batch_size whole scans holding at most that many patches, else batch_size scans; with
    `data.mix_alpha` (and `mix_by`), strata reweighted; with `subject`/`session` columns, scans batched in pairs of
    one session (`session_ids`). Evaluation: ordered, batch_size scans."""
    store = VolumeStore(data["cache_dir"])
    workers = int(data.get("num_workers", 8))
    common = {"num_workers": workers, "collate_fn": collate, "pin_memory": True,
              "persistent_workers": train and workers > 0,
              "prefetch_factor": int(data.get("prefetch", 4)) if workers else None}  # batches in flight per worker
    if not train:
        return DataLoader(PatchDataset(rows, store, data["patch_mm"], **ds_kwargs), batch_size=batch_size, **common)
    rows = rows.reset_index(drop=True)
    if not data.get("patches_per_batch"):
        sizes, budget = [1] * len(rows), batch_size
    elif "shape" in rows:  # pool manifests carry the header (data/pool.py): no scan is read, any patch_mm
        from .pool import n_patch
        sizes, budget = [n_patch(tuple(map(int, s.split("x"))), tuple(map(float, z.split("x"))), data["patch_mm"])
                         for s, z in zip(rows["shape"], rows["spacing"])], int(data["patches_per_batch"])
    else:
        sizes, budget = [], int(data["patches_per_batch"])
        for p in rows["path"]:
            arr, spacing, _ = store.load(p)
            sizes.append(int(patch_grid(arr.shape, patch_dims(spacing, data["patch_mm"])).prod()))
    weights = mix_weights(rows, data.get("mix_by", ["dataset", "modality"]), float(data.get("mix_alpha", 1.0)))
    ds = PatchDataset(rows, store, data["patch_mm"], sizes=sizes, **ds_kwargs)
    return DataLoader(ds, batch_sampler=BudgetBatches(sizes, budget, batch_size, seed, weights, rank, world,
                                                      session_ids(rows)), **common)


def session_ids(rows: pd.DataFrame) -> np.ndarray | None:
    """Session index per row (subject x session; a missing session is one per subject), None without those columns."""
    if not {"subject", "session"} <= set(rows):
        return None
    key = [rows["subject"], rows["session"].astype(str)]  # read_csv may mix int and str sessions across chunks
    return rows.groupby(key, dropna=False, sort=False).ngroup().to_numpy()
