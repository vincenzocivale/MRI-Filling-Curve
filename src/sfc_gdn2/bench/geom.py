"""Source voxel -> model input grid, for segmentation through each model's own network (fm/*.py `seg_input`,
fm/segrun.py), and the GT reader.

`source_to_input(model, meta)` = 4x4 affine from a source voxel index (i, j, k of the NIfTI the model was given) to
the continuous voxel index of the model's network input grid, rebuilt from the geometry its wrapper stores in `meta`.
`None` = not invertible (BrainIAC registers to a template and keeps no transform).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

# model (config stem) -> family
FAMILY = {"fomo26": "asparagus", "brainfm": "brainfm", "bsf_ukb_pretrain": "bsf_ukb", "bsf_atlas_pretrain": "bsf_atlas",
          "medicalnet_resnet50_23dataset": "medicalnet", "nnunet_totalseg_mr_870": "nnunet",
          "openmind_resencl_mae": "nnssl", "openmind_resencl_simclr": "nnssl", "openmind_primusm_mae": "nnssl",
          "nnfoundation_cnn": "nnssl", "nnfoundation_vit": "nnssl", "brainiac": None}


def _np(a) -> np.ndarray:
    return np.asarray(a.numpy() if isinstance(a, torch.Tensor) else a, dtype=np.float64)


def _scale(src_shape, dst_shape, align_corners: bool = False) -> np.ndarray:
    """Index map of an independent per-axis resize src_shape -> dst_shape (inverse of the output -> input map)."""
    m = np.eye(4)
    for a, (n, k) in enumerate(zip(src_shape, dst_shape)):
        if align_corners:  # scipy.ndimage.zoom (grid_mode=False): out o <- in o (n-1)/(k-1)
            m[a, a] = (k - 1) / max(n - 1, 1)
        else:              # pixel centres (monai / skimage / torch align_corners=False)
            m[a, a], m[a, 3] = k / n, 0.5 * k / n - 0.5
    return m


def _shift(offset) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def _perm(order) -> np.ndarray:
    m = np.zeros((4, 4))
    m[3, 3] = 1
    for a, b in enumerate(order):
        m[a, b] = 1
    return m


def _nnunet_crop_resample(props_or_meta: dict, reverse_then_transpose: np.ndarray) -> np.ndarray:
    """SimpleITK array (z, y, x) of the read image -> nnU-Net transpose -> crop -> resample -> preprocessed index."""
    bbox = [b[0] for b in props_or_meta["bbox_used_for_cropping"]]
    crop_shape = props_or_meta["shape_after_cropping_and_before_resampling"]
    pre_shape = props_or_meta.get("preprocessed_shape") or props_or_meta["shape_after_resampling"]
    return _scale(crop_shape, pre_shape) @ _shift([-b for b in bbox]) @ reverse_then_transpose


def brainfm_input_affine(source_affine, source_shape, crop_start) -> np.ndarray:
    """Voxel -> world of BrainFM's network input. `prepare_image` (BrainFM utils/test_utils.py:253-276) resizes to 1 mm
    (misc.py:1124-1126 size, myzoom affine update :1108-1113) and then "aligns" `final` with the ALREADY aligned
    affine (:272-276), a no-op: the network sees the source axis order, while the affine it returns is the
    RAS-aligned one (equal only for near-RAS inputs). Then the zero crop starting at `crop_start`."""
    a = _np(source_affine)
    shape = np.asarray(source_shape[:3], dtype=np.float64)
    f = np.round(shape * np.sqrt((a[:3, :3] ** 2).sum(0))) / shape
    r = a.copy()
    r[:3, :3] = a[:3, :3] / f
    r[:3, 3] = a[:3, 3] - a[:3, :3] @ (0.5 - 0.5 / f) + (a[:3, :3] / f) @ np.asarray(crop_start, dtype=np.float64)
    return r


def source_to_input(model: str, meta: dict) -> np.ndarray | None:
    """`model` = a config stem of FAMILY, or a family name (the wrappers' own segmentation inputs). A wrapper may
    store the map itself in `meta["source_to_input"]` (ours, fm/sfc.py)."""
    if "source_to_input" in meta:
        return _np(meta["source_to_input"])
    fam = FAMILY.get(model, model)
    if fam == "asparagus":
        return np.linalg.inv(_np(meta["affine"])) @ _np(meta["source_affine"])
    if fam == "brainfm":
        a = brainfm_input_affine(meta["source_affine"], meta["source_shape"], meta["bbox"][0])
        return np.linalg.inv(a) @ _np(meta["source_affine"])
    if fam == "bsf_ukb":
        return np.linalg.inv(_np(meta["input_affine"])) @ _np(meta["source_affine"])
    if fam == "bsf_atlas":
        return _scale(meta["source_shape"], meta["input_shape"])
    if fam == "medicalnet":
        return _scale(meta["source_shape"], meta["input_shape"], align_corners=True)
    if fam == "nnssl":
        rt = _perm(meta["transpose_forward"]) @ _perm([2, 1, 0])  # NIfTI (i,j,k) -> sitk (k,j,i) -> transpose
        pre = _nnunet_crop_resample(meta, rt)
        return _shift([r[0] for r in meta["revert_padding"]]) @ pre  # into the padded grid of the sliding window
    if fam == "nnunet":
        g, props = meta["totalseg"], meta["nnunet_properties"]
        to_rsp = np.linalg.inv(_np(g["resampled_affine"])) @ _np(g["original_affine"]) if g else np.eye(4)
        rt = _perm(meta["transpose_forward"]) @ _perm([2, 1, 0])
        return _nnunet_crop_resample(props | {"preprocessed_shape": meta["preprocessed_shape"]}, rt) @ to_rsp
    return None


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
