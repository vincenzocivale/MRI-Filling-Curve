from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import torch
import torch.nn.functional as F


def _atomic_write(dest: Path, write) -> None:
    """Unique tmp name + os.replace: safe when several workers/runs fill the same cache."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    write(tmp)
    os.replace(tmp, dest)


def patch_dims(spacing, patch_mm: float) -> tuple[int, int, int]:
    """Native voxels per patch axis: the whole number closest to `patch_mm` mm (>= 1)."""
    return tuple(max(1, round(patch_mm / float(z))) for z in spacing[:3])


def load_canonical(path: str | Path) -> tuple[np.ndarray, tuple, float, float]:
    """RAS-closest orientation (axis permutation and flips only, as nib.as_closest_canonical: no value
    changed) of the voxels as stored (dtype kept, header scaling not applied) -> (raw, zooms in the
    new axis order, slope, inter); the scan's values are raw * slope + inter."""
    img = nib.load(str(path))
    ornt = nio.io_orientation(img.affine)
    raw = nio.apply_orientation(img.dataobj.get_unscaled(), ornt)
    if raw.dtype == np.uint16:   # torch has no uint16 tensors
        raw = raw.astype(np.int32)
    elif raw.dtype == np.float64:
        raw = raw.astype(np.float32)
    zooms = [0.0] * 3
    for i, z in enumerate(img.header.get_zooms()[:3]):
        zooms[int(ornt[i, 0])] = float(z)
    slope, inter = (float(v) if v is not None and np.isfinite(v) else d
                    for v, d in ((img.dataobj.slope, 1.0), (img.dataobj.inter, 0.0)))
    return np.ascontiguousarray(raw), tuple(zooms), slope or 1.0, inter


class VolumeStore:
    """A scan exactly as acquired: canonical orientation (no value changed), stored dtype, native voxels.
    `load` -> (raw array, spacing mm [3], (a, c)): a * raw + c is the scan's value (header slope and
    intercept) divided by the 99th percentile of its voxels > 0, i.e. ~[0, 1] without clipping anything.

    The first access decodes the NIfTI (extracting it from its zip if needed) and writes the array to
    `cache_dir/native/` (+ a .json with spacing and scale); every later access is a memory-mapped
    .npy read, which is what keeps the data loader off the critical path.
    """

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)

    def _cache_path(self, path: str) -> Path:
        key = hashlib.sha1(f"{path}|native-v1".encode()).hexdigest()[:20]
        return self.root / "native" / key[:2] / f"{key}.npy"

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

    def load(self, path: str) -> tuple[np.ndarray, tuple, float]:
        arr_path = self._cache_path(path)
        meta_path = arr_path.with_suffix(".json")
        if not meta_path.exists():
            arr, spacing, slope, inter = load_canonical(self._materialize(path))
            val = arr.astype(np.float64) * slope + inter
            fg = val[val > 0]
            scale = 1.0 / max(float(np.percentile(fg if fg.size else val, 99)), 1e-6)
            meta = {"spacing": spacing, "affine": [slope * scale, inter * scale]}

            def write_arr(tmp):
                with open(tmp, "wb") as f:
                    np.save(f, arr)
            _atomic_write(arr_path, write_arr)
            _atomic_write(meta_path, lambda tmp: Path(tmp).write_text(json.dumps(meta)))
            return arr, tuple(spacing), tuple(meta["affine"])
        meta = json.loads(meta_path.read_text())
        return np.load(arr_path, mmap_mode="r"), tuple(meta["spacing"]), tuple(meta["affine"])


def patchify(vol: torch.Tensor, k) -> torch.Tensor:
    """[D,H,W] volume -> [N, k0*k1*k2] non-overlapping patches of k = (k0, k1, k2) voxels (an int: a
    cube), in raster order of the patch grid. The end of each axis is zero-padded to a multiple of k."""
    k = (k,) * 3 if isinstance(k, int) else tuple(int(v) for v in k)
    vol = F.pad(vol, [p for n, q in zip(reversed(vol.shape), reversed(k)) for p in (0, -n % q)])
    (gx, gy, gz), (a, b, c) = (n // q for n, q in zip(vol.shape, k)), k
    return vol.reshape(gx, a, gy, b, gz, c).permute(0, 2, 4, 1, 3, 5).reshape(gx * gy * gz, a * b * c)


def patch_grid(shape, k) -> torch.Tensor:
    """Patch grid [3] of a volume of `shape` cut into patches of k voxels (end padded)."""
    return torch.tensor([-(-int(n) // int(q)) for n, q in zip(shape[:3], k)])
