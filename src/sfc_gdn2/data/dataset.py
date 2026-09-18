from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .volume import load_volume, patchify


def load_manifests(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def balanced_split(df: pd.DataFrame, seed: int, max_train_per_dataset: int, max_val_per_dataset: int):
    rng = np.random.default_rng(seed)
    train_frames, val_frames = [], []
    for _, group in df.groupby("dataset", sort=True):
        subjects = sorted(group["subject"].astype(str).unique())
        perm = rng.permutation(len(subjects))
        subjects = [subjects[i] for i in perm]
        n_val_subj = max(1, round(0.2 * len(subjects))) if len(subjects) > 1 else 0
        val_subjects = set(subjects[:n_val_subj])
        train_subjects = set(subjects) - val_subjects
        g_train = group[group["subject"].astype(str).isin(train_subjects)].iloc[:max_train_per_dataset]
        g_val = group[group["subject"].astype(str).isin(val_subjects)].iloc[:max_val_per_dataset]
        if g_val.empty and len(g_train) > 1:
            g_val = g_train.tail(1)
            g_train = g_train.iloc[:-1]
        train_frames.append(g_train)
        val_frames.append(g_val)
    tr = pd.concat(train_frames, ignore_index=True) if train_frames else df.iloc[0:0]
    va = pd.concat(val_frames, ignore_index=True) if val_frames else df.iloc[0:0]
    return tr, va


class MRIPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_shape, patch_size: int, cache_dir: str):
        self.rows = df.reset_index(drop=True)
        self.target_shape = tuple(target_shape)
        self.patch_size = patch_size
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        vol = load_volume(row["path"], self.target_shape, self.cache_dir)
        patches = patchify(vol, self.patch_size)
        return {"patches": patches, "dataset": row["dataset"], "sample_id": row["sample_id"]}
