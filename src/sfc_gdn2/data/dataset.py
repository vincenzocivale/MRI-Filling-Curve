from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .pool import n_patch
from .volume import decode, patch_dims, patch_grid, patchify


class PatchDataset(Dataset):
    """Scans as their native voxels (stored dtype; `to_device` makes the values and the [N, P] patches,
    P = k0*k1*k2, k = the voxels per axis closest to `patch_mm`: `patch_dims`), with spacing (mm), k,
    patch grid [3], value `affine` (a, c) and the row index (callers look up per-row labels themselves).
    A scan that fails to load is logged and replaced by a random scan of at most its patches (`sizes`: the batch
    keeps its budget), so one corrupt file of ~1M does not stop the run."""

    def __init__(self, rows: pd.DataFrame, patch_mm: float, sizes: list[int]):
        self.rows = rows.reset_index(drop=True)
        self.paths = self.rows["path"].tolist()
        self.frames = self.rows["frames"].tolist()
        self.patch_mm, self.sizes = patch_mm, sizes

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, i: int):
        # 4D+ scans (pool column `frames`): a random frame per access (torch RNG: seeded per worker by the loader)
        n = int(self.frames[i])
        return decode(self.paths[i], int(torch.randint(n, ())) if n > 1 else None)

    def __getitem__(self, i: int) -> dict:
        for _ in range(10):
            try:
                arr, spacing, affine = self._load(i)
                break
            except Exception as e:  # noqa: BLE001 -- unreadable scan: logged, replaced
                print(f"[data] skip {self.paths[i]}: {e!r}", file=sys.stderr, flush=True)
                fits = np.flatnonzero(np.asarray(self.sizes) <= self.sizes[i])
                i = int(fits[torch.randint(len(fits), ())])
        else:
            raise RuntimeError(f"10 unreadable scans in a row, last {self.paths[i]}")
        k = patch_dims(spacing, self.patch_mm)
        return {"volume": torch.from_numpy(np.array(arr)), "grid": patch_grid(arr.shape, k), "k": torch.tensor(k),
                "spacing": torch.tensor(spacing, dtype=torch.float32),
                "affine": torch.tensor(affine, dtype=torch.float64), "index": i}


def collate(items: list[dict]) -> dict:
    """Volumes stay a list (shapes differ)."""
    out = {k: torch.stack([it[k] for it in items]) for k in ("grid", "k", "spacing", "affine")}
    return out | {"volumes": [it["volume"] for it in items], "index": torch.tensor([it["index"] for it in items])}


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
    `groups` (session per scan): each drawn scan comes with a random other scan of its group in the same batch
    (alone if it has none, or if the two exceed the budget)."""

    def __init__(self, sizes: list[int], budget: int, max_batch: int, seed: int, groups,
                 weights: torch.Tensor | None = None, rank: int = 0, world: int = 1):
        self.sizes, self.budget, self.max_batch, self.seed, self.weights = sizes, budget, max_batch, seed, weights
        self.rank, self.world, self.start = rank, world, 0
        self.groups = torch.tensor(groups)  # scans sorted by group: a group's scans are members[first : first + count]
        self.members = torch.argsort(self.groups, stable=True)
        self.count = torch.bincount(self.groups)
        self.first = self.count.cumsum(0) - self.count
        self.slot = torch.empty_like(self.groups)
        self.slot[self.members] = torch.arange(len(self.groups)) - self.first[self.groups[self.members]]

    def batches(self, epoch: int):
        g = torch.Generator().manual_seed(self.seed + epoch)
        order = (torch.randperm(len(self.sizes), generator=g) if self.weights is None
                 else torch.multinomial(self.weights, len(self.sizes), replacement=True, generator=g))
        grp = self.groups[order]
        n = self.count[grp]
        j = (torch.rand(len(order), generator=g) * (n - 1)).long()
        j += j >= self.slot[order]                                        # any member but the scan itself
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


class _Scans(torch.utils.data.Sampler):
    """The scans of a batch plan, one after the other (what the per-scan loader's workers fetch)."""

    def __init__(self, plan: BudgetBatches):
        self.plan = plan

    def __iter__(self):
        return itertools.chain.from_iterable(self.plan)


class ScanBatches:
    """Training batches: a per-scan DataLoader (each worker loads single scans, so host RAM follows the scans in flight (workers x prefetch, at least two batches), not a whole batch per
    worker, and the workers' I/O latencies overlap: a pool scan is ~0.5 s of Lustre reads and gunzip)
    regrouped in the main process by the same deterministic plan. `batch_sampler` is the plan: its `start` places a
    resumed run. A whole batch per worker had both costs grow with the batch (2 GPUs x 72 scans: OOM at 240 GB, and
    25-46% of each step waiting for data)."""

    def __init__(self, ds: Dataset, plan: BudgetBatches, **common):
        self.batch_sampler = plan
        self.loader = DataLoader(ds, sampler=_Scans(plan), batch_size=None, **common)

    def __iter__(self):
        scans = iter(self.loader)
        for b in self.batch_sampler:
            yield collate([next(scans) for _ in b])


def mix_weights(rows: pd.DataFrame, by: list[str], alpha: float) -> torch.Tensor | None:
    """Per-scan sampling weights giving each stratum (`by`, e.g. dataset x modality) of n scans a share
    proportional to n^alpha: 1 = as in the pool (None), 0 = equal strata."""
    if alpha == 1:
        return None
    n = rows.groupby(by, dropna=False)[by[0]].transform("size").to_numpy(dtype=np.float64)
    return torch.from_numpy(n ** (alpha - 1))


def loader(rows: pd.DataFrame, data: dict, batch_size: int, sessions, seed: int = 0, rank: int = 0,
           world: int = 1) -> ScanBatches:
    """Endless `BudgetBatches` over all rows, this rank's share (`rank` of `world`), worker-persistent: batches of
    up to batch_size whole scans holding at most `data.patches_per_batch` patches (sized from the pool manifest's
    header `shape` / `spacing`, data/pool.py: no scan is read, any patch_mm); with `data.mix_alpha` (and `mix_by`),
    strata reweighted; scans batched in pairs of one session (`sessions`: `session_ids`); scans loaded one per
    worker (`ScanBatches`)."""
    workers = int(data.get("num_workers", 8))
    # scans in flight: at least two batches (m7: 32 workers x 4 = 128 < a 144-scan batch, 33% of each step waiting)
    prefetch = max(int(data.get("prefetch", 4)), -(-2 * batch_size // max(workers, 1)))
    common = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0,
              "prefetch_factor": prefetch if workers else None}
    rows = rows.reset_index(drop=True)
    budget_mm = data.get("budget_patch_mm") or data["patch_mm"]  # with patch_mm_choices: the smallest (most tokens)
    sizes = [n_patch(tuple(map(int, s.split("x"))), tuple(map(float, z.split("x"))), budget_mm)
             for s, z in zip(rows["shape"], rows["spacing"])]
    weights = mix_weights(rows, data.get("mix_by", ["dataset", "modality"]), float(data.get("mix_alpha", 1.0)))
    ds = PatchDataset(rows, data["patch_mm"], sizes)
    plan = BudgetBatches(sizes, int(data["patches_per_batch"]), batch_size, seed, sessions, weights, rank, world)
    return ScanBatches(ds, plan, **common)


def session_ids(rows: pd.DataFrame) -> np.ndarray:
    """Session index per row (subject x session; a missing session is one per subject)."""
    key = [rows["subject"], rows["session"].astype(str)]  # read_csv may mix int and str sessions across chunks
    return rows.groupby(key, dropna=False, sort=False).ngroup().to_numpy()
