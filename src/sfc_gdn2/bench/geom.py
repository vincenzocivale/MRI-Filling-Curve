"""Source voxel -> model input grid, for segmentation through each model's own network (fm/*.py `seg_input`,
fm/segrun.py), and the GT reader.

Each wrapper's `seg_input` returns the 4x4 affine from a source voxel index (i, j, k of the NIfTI the model was
given) to the continuous voxel index of the model's network input grid; the shared pieces are here.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import from_matvec


def _scale(src_shape, dst_shape, align_corners: bool = False) -> np.ndarray:
    """Index map of an independent per-axis resize src_shape -> dst_shape (inverse of the output -> input map)."""
    m = np.eye(4)
    for a, (n, k) in enumerate(zip(src_shape, dst_shape)):
        if align_corners:  # scipy.ndimage.zoom (grid_mode=False): out o <- in o (n-1)/(k-1)
            m[a, a] = (k - 1) / max(n - 1, 1)
        else:              # pixel centres (monai / skimage / torch align_corners=False)
            m[a, a], m[a, 3] = k / n, 0.5 * k / n - 0.5
    return m


def _nnunet_crop_resample(props: dict, preprocessed_shape, transpose_forward) -> np.ndarray:
    """NIfTI (i, j, k) -> SimpleITK array (k, j, i) of the read image -> nnU-Net transpose -> crop -> resample ->
    preprocessed index."""
    bbox = [b[0] for b in props["bbox_used_for_cropping"]]
    reverse_then_transpose = np.eye(4)[[*transpose_forward, 3]] @ np.eye(4)[[2, 1, 0, 3]]
    return (_scale(props["shape_after_cropping_and_before_resampling"], preprocessed_shape)
            @ from_matvec(np.eye(3), [-b for b in bbox]) @ reverse_then_transpose)


def brainfm_input_affine(source_affine, source_shape, crop_start) -> np.ndarray:
    """Voxel -> world of BrainFM's network input. `prepare_image` (BrainFM utils/test_utils.py:253-276) resizes to 1 mm
    (misc.py:1124-1126 size, myzoom affine update :1108-1113) and then "aligns" `final` with the ALREADY aligned
    affine (:272-276), a no-op: the network sees the source axis order, while the affine it returns is the
    RAS-aligned one (equal only for near-RAS inputs). Then the zero crop starting at `crop_start`."""
    a = np.asarray(source_affine, dtype=np.float64)
    shape = np.asarray(source_shape[:3], dtype=np.float64)
    f = np.round(shape * np.sqrt((a[:3, :3] ** 2).sum(0))) / shape
    r = a.copy()
    r[:3, :3] = a[:3, :3] / f
    r[:3, 3] = a[:3, 3] - a[:3, :3] @ (0.5 - 0.5 / f) + (a[:3, :3] / f) @ np.asarray(crop_start, dtype=np.float64)
    return r


def gt_labels(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(labels 0..K, affine, class names): a mask file (any > 0 = 1) or a directory of one mask per class."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.nii*"))
        ref = nib.load(files[0])
        lab = np.zeros(ref.shape, dtype=np.uint8)
        for k, f in enumerate(files, 1):
            lab[np.asarray(nib.load(f).dataobj) > 0] = k
        return lab, ref.affine, [f.name.split(".")[0] for f in files]
    img = nib.load(p)
    return (np.asarray(img.dataobj) > 0).astype(np.uint8), img.affine, ["lesion"]
