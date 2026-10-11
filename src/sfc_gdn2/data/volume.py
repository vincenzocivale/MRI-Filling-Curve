from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import torch
import torch.nn.functional as F


def patch_dims(spacing, patch_mm: float) -> tuple[int, int, int]:
    """Native voxels per patch axis: the whole number closest to `patch_mm` mm (>= 1)."""
    return tuple(max(1, round(patch_mm / float(z))) for z in spacing[:3])


def open_image(path: str | Path):
    """A NIfTI file, or a `zip://archive::member` read in memory (nothing extracted to disk)."""
    path = str(path)
    if not path.startswith("zip://"):
        return nib.load(path)
    archive, member = path[len("zip://"):].split("::", 1)
    with zipfile.ZipFile(archive) as zf:
        data = zf.read(member)
    return nib.Nifti1Image.from_bytes(gzip.decompress(data) if member.endswith(".gz") else data)


def load_canonical(path, frame: int | None = None) -> tuple[np.ndarray, tuple, float, float]:
    """RAS-closest orientation (axis permutation and flips only, as nib.as_closest_canonical: no value
    changed) of the voxels as stored (dtype kept, header scaling not applied) -> (raw, zooms in the
    new axis order, slope, inter); the scan's values are raw * slope + inter. `path`: a file, a zip member
    or a loaded image. 4D+ (DWI, ASL, ...): the 3D volume `frame` of the flattened trailing axes (required when
    there is more than one)."""
    img = path if isinstance(path, nib.spatialimages.SpatialImage) else open_image(path)
    ornt = nio.io_orientation(img.affine)
    raw = img.dataobj.get_unscaled()
    if raw.ndim > 3:
        raw = raw.reshape(*raw.shape[:3], -1)
        if frame is None and raw.shape[3] > 1:
            raise ValueError(f"{path}: {raw.shape[3]} frames, pass `frame`")
        raw = raw[..., frame or 0]
    raw = nio.apply_orientation(raw, ornt)
    raw = raw.astype(raw.dtype.newbyteorder("="), copy=False)   # big-endian files: torch needs native order
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


def decode(path, frame: int | None = None) -> tuple[np.ndarray, tuple, tuple]:
    """A scan exactly as acquired: canonical orientation (no value changed), stored dtype, native voxels.
    `path`: a file, a zip member (read in memory, nothing written) or a loaded image -> (raw, spacing mm [3], (a, c)):
    a * raw + c is the scan's value (header slope and intercept) divided by the 99th percentile of its voxels > 0,
    i.e. ~[0, 1] without clipping anything; a 4D+ scan's `frame`, scaled on its own (b0 and DWI frames differ)."""
    arr, spacing, slope, inter = load_canonical(path, frame)
    # the 99th percentile of the voxels > 0, on every 4th voxel in float32 (in float64 on all of them it was a
    # quarter of the loader's time and up to 1 GB per worker on the big scans)
    val = arr.reshape(-1)[::4].astype(np.float32) * slope + inter
    fg = val[val > 0]
    scale = 1.0 / max(float(np.percentile(fg if fg.size else val, 99)), 1e-6)
    return arr, tuple(spacing), (slope * scale, inter * scale)


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
