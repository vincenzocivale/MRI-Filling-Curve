from __future__ import annotations

import os
import uuid
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def materialize(path: str, cache_dir: str | Path) -> Path:
    if not path.startswith("zip://"):
        return Path(path)
    archive_str, member = path[len("zip://"):].split("::", 1)
    archive = Path(archive_str)
    # archive.parent.name disambiguates layouts where the archive stem alone repeats
    # across subjects (e.g. every subject's `ses-01.zip`).
    dest = Path(cache_dir) / archive.parent.name / archive.stem / member
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Unique per-call tmp name + atomic replace: safe if multiple processes/workers
        # materialize the same member concurrently (shared cache_dir across parallel runs).
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
        with zipfile.ZipFile(archive) as zf, zf.open(member) as src, open(tmp, "wb") as out:
            out.write(src.read())
        os.replace(tmp, dest)
    return dest


def load_volume(path: str, target_shape: tuple[int, int, int], cache_dir: str | Path) -> torch.Tensor:
    local = materialize(path, cache_dir)
    img = nib.as_closest_canonical(nib.load(str(local)))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / max(hi - lo, 1e-6)
    t = torch.from_numpy(vol).float()[None, None]
    t = F.interpolate(t, size=tuple(target_shape), mode="trilinear", align_corners=False)
    return t[0, 0]


def patchify(vol: torch.Tensor, patch_size: int) -> torch.Tensor:
    g = vol.shape[0] // patch_size
    v = vol.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return v.reshape(g * g * g, patch_size ** 3).contiguous()
