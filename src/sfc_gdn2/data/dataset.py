from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .volume import VolumeStore, patchify


def cohort_key(df: pd.DataFrame) -> pd.Series:
    """Per-source grouping key: `cohort` when the scanner emitted one, else `dataset`."""
    return df["cohort"] if "cohort" in df.columns else df["dataset"]


class PatchDataset(Dataset):
    """Scans as [N, patch^3] float16 patch tensors in canonical raster order, with the row index
    (callers look up per-row labels themselves) and, optionally, `target(row)` computed in the
    worker (e.g. per-patch segmentation labels)."""

    def __init__(self, rows: pd.DataFrame, store: VolumeStore, patch: int,
                 target: Callable[[pd.Series], torch.Tensor] | None = None):
        self.rows = rows.reset_index(drop=True)
        self.paths = self.rows["path"].tolist()
        self.store, self.patch, self.target = store, patch, target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        vol = torch.from_numpy(np.array(self.store.load(self.paths[i]), dtype=np.float16))
        item = {"patches": patchify(vol, self.patch), "index": i}
        if self.target is not None:
            item["target"] = self.target(self.rows.iloc[i])
        return item


def loader(rows: pd.DataFrame, data: dict, batch_size: int, train: bool, seed: int = 0,
           **ds_kwargs) -> DataLoader:
    """Shuffled (seeded) and worker-persistent for training, ordered for evaluation."""
    ds = PatchDataset(rows, VolumeStore(data["cache_dir"], data["target_shape"]), data["patch_size"],
                      **ds_kwargs)
    workers = int(data.get("num_workers", 8))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=workers,
                      pin_memory=True, persistent_workers=train and workers > 0,
                      prefetch_factor=4 if workers else None,
                      generator=torch.Generator().manual_seed(seed))
