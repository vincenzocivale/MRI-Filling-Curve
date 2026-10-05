"""TotalSegmentator MRI (v3: <root>/sNNNN/{mri.nii.gz, segmentations/<class>.nii.gz}, meta.csv)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .volume import VolumeStore, _atomic_write, fit_cube, load_canonical, patchify


def scan(cfg: dict) -> pd.DataFrame:
    """One row per case; `mask_path` is the case's segmentations/ directory. Case = subject (the
    release has no patient id)."""
    meta = pd.read_csv(Path(cfg["root"]) / "meta.csv", sep=";", encoding="utf-8-sig")
    return pd.DataFrame({
        "dataset": cfg["name"], "cohort": cfg["name"], "sample_id": meta["image_id"], "subject": meta["image_id"],
        "session": "", "modality": "mri", "path": [str(Path(cfg["root"]) / i / "mri.nii.gz") for i in meta["image_id"]],
        "mask_path": [str(Path(cfg["root"]) / i / "segmentations") for i in meta["image_id"]],
        "split": "unused", "age": pd.to_numeric(meta["age"], errors="coerce"), "sex": meta["gender"]})


def attach(df: pd.DataFrame, root, columns: list[str]) -> pd.DataFrame:
    """`segmentation` = the segmentations directory (every case has one)."""
    return df.assign(segmentation=df["mask_path"])


def classes(seg_dir: str | Path) -> list[str]:
    return sorted(p.name.removesuffix(".nii.gz") for p in Path(seg_dir).glob("*.nii.gz"))


class PatchLabels:
    """Per-patch majority class (0 = background, i + 1 = classes[i]) in the same canonical,
    aspect-preserving geometry as VolumeStore, cached as [N] int16. Picklable (DataLoader workers)."""

    def __init__(self, store: VolumeStore, patch: int, class_names: list[str]):
        self.store, self.patch, self.classes = store, patch, class_names

    def __call__(self, row: pd.Series) -> torch.Tensor:
        key = hashlib.sha1(f"{row['mask_path']}|{self.store.shape}|{self.patch}|iso-majority".encode())
        dest = self.store.root / "patch_labels" / f"{key.hexdigest()[:20]}.npy"
        if not dest.exists():
            arr = self._compute(row["mask_path"])

            def write(tmp):
                with open(tmp, "wb") as f:
                    np.save(f, arr)
            _atomic_write(dest, write)
        return torch.from_numpy(np.load(dest).astype(np.int64))

    def _compute(self, seg_dir: str) -> np.ndarray:
        label, zooms = None, None
        for i, name in enumerate(self.classes, start=1):
            m, zooms = load_canonical(Path(seg_dir) / f"{name}.nii.gz")
            label = np.zeros(m.shape, np.float32) if label is None else label
            label[m > 0] = i
        cube = torch.from_numpy(fit_cube(label, zooms, self.store.side, mode="nearest")).long()
        counts = torch.zeros(cube.numel() // self.patch ** 3, len(self.classes) + 1, dtype=torch.long)
        p = patchify(cube, self.patch)
        counts.scatter_add_(1, p, torch.ones_like(p))
        return counts.argmax(1).numpy().astype(np.int16)
