from __future__ import annotations

import hashlib
import os
import uuid
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _atomic_write(dest: Path, write) -> None:
    """Unique tmp name + os.replace: safe when several workers/runs fill the same cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    write(tmp)
    os.replace(tmp, dest)


def foreground_percentile(vol: np.ndarray) -> np.ndarray:
    """Clip to the 1st/99th percentile of voxels > 0 and rescale to [0, 1]."""
    fg = vol[vol > 0]
    lo, hi = np.percentile(fg if fg.size else vol, [1, 99])
    return (np.clip(vol, lo, hi) - lo) / max(hi - lo, 1e-6)


def fit_cube(vol: np.ndarray, zooms, side: int, mode: str = "trilinear") -> np.ndarray:
    """Aspect-preserving resample: isotropic voxels with the longest physical side spanning `side`,
    zero-padded (centred) to a `side`^3 cube. Thick-slice stacks keep their true proportions
    instead of being stretched to fill the cube."""
    extent = np.asarray(vol.shape) * np.asarray(zooms[:3], dtype=float)
    shape = tuple(int(s) for s in np.clip(np.round(extent / extent.max() * side), 1, side))
    kw = {"align_corners": False} if mode == "trilinear" else {}
    t = F.interpolate(torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))[None, None],
                      size=shape, mode=mode, **kw)[0, 0]
    out = torch.zeros((side,) * 3)
    lo = [(side - s) // 2 for s in shape]
    out[lo[0]:lo[0] + shape[0], lo[1]:lo[1] + shape[1], lo[2]:lo[2] + shape[2]] = t
    return out.numpy()


def load_canonical(path: str | Path) -> tuple[np.ndarray, tuple]:
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(img.dataobj, dtype=np.float32), img.header.get_zooms()


class VolumeStore:
    """Loads a scan as a canonical, normalised, isotropically resampled float16 cube.

    The first access decodes the NIfTI (extracting it from its zip if needed) and writes the
    result to `cache_dir/cubes/`; every later access is a single memory-mapped .npy read, which
    is what keeps the data loader off the critical path.
    """

    def __init__(self, cache_dir: str | Path, target_shape: tuple[int, int, int]):
        self.root = Path(cache_dir)
        self.shape = tuple(target_shape)
        if len(set(self.shape)) != 1:
            raise ValueError(f"target_shape must be cubic, got {self.shape}")
        self.side = self.shape[0]

    def _cube_path(self, path: str) -> Path:
        key = hashlib.sha1(f"{path}|{self.shape}|iso-fg-p1-p99".encode()).hexdigest()[:20]
        return self.root / "cubes" / key[:2] / f"{key}.npy"

    def _materialize(self, path: str) -> Path:
        if not path.startswith("zip://"):
            return Path(path)
        archive, member = path[len("zip://"):].split("::", 1)
        archive = Path(archive)
        # archive.parent.name disambiguates layouts where every subject's archive is `ses-01.zip`.
        dest = self.root / archive.parent.name / archive.stem / member
        if not dest.exists():
            def write(tmp):
                with zipfile.ZipFile(archive) as zf, zf.open(member) as src, open(tmp, "wb") as out:
                    out.write(src.read())
            _atomic_write(dest, write)
        return dest

    def _decode(self, path: str) -> np.ndarray:
        vol, zooms = load_canonical(self._materialize(path))
        return fit_cube(foreground_percentile(vol), zooms, self.side).astype(np.float16)

    def load(self, path: str) -> np.ndarray:
        cube = self._cube_path(path)
        if not cube.exists():
            arr = self._decode(path)
            def write(tmp):
                with open(tmp, "wb") as f:
                    np.save(f, arr)
            _atomic_write(cube, write)
            return arr
        return np.load(cube, mmap_mode="r")


def patchify(vol: torch.Tensor, patch: int) -> torch.Tensor:
    """[D,H,W] cube -> [N, patch^3] non-overlapping patches in canonical raster order."""
    g = vol.shape[0] // patch
    v = vol.reshape(g, patch, g, patch, g, patch).permute(0, 2, 4, 1, 3, 5)
    return v.reshape(g ** 3, patch ** 3)


def unpatchify(patches: torch.Tensor, grid: int) -> torch.Tensor:
    """Inverse of `patchify`, batched: [B, grid^3, p^3] canonical-order patches -> [B, S, S, S] cubes."""
    b, _, v = patches.shape
    p = round(v ** (1 / 3))
    x = patches.reshape(b, grid, grid, grid, p, p, p).permute(0, 1, 4, 2, 5, 3, 6)
    return x.reshape(b, grid * p, grid * p, grid * p)
