"""Where each stored feature map sits on the source NIfTI (for the segmentation probes).

`source_to_input(model, meta)` = 4x4 affine from a source voxel index (i, j, k of the NIfTI the model was given) to
the continuous voxel index of the model's network input grid, rebuilt from the geometry its wrapper stores in `meta`
(fm/*.py). A map at stride s covers input voxels [c*s, (c+1)*s) per axis, so its continuous cell coordinate is
(input + 0.5) / s - 0.5. `None` = not invertible (BrainIAC registers to a template and keeps no transform).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

# model (config stem) -> family; family -> {map name: stride on the input grid}
FAMILY = {"fomo26": "asparagus", "brainfm": "brainfm", "bsf_ukb_pretrain": "bsf_ukb", "bsf_atlas_pretrain": "bsf_atlas",
          "medicalnet_resnet50_23dataset": "medicalnet", "nnunet_totalseg_mr_870": "nnunet",
          "openmind_resencl_mae": "nnssl", "openmind_resencl_simclr": "nnssl", "openmind_primusm_mae": "nnssl",
          "nnfoundation_cnn": "nnssl", "nnfoundation_vit": "nnssl", "brainiac": None}
STRIDES = {
    "asparagus": {"stage3": 8, "stage4": 16, "stage5": 32},
    "brainfm": {"feat_2": 8, "feat_last_pool8": 8, "feat_1": 16, "feat_0": 32},
    "bsf_ukb": {"stage2": 8, "stage3": 16, "stage4": 32},
    "bsf_atlas": {"stage2": 8, "stage3": 16, "stage4": 32},
    "medicalnet": {"layer4_pool144": (8, 32, 32)},
    "nnunet": {"stage2": 4, "stage3": 8, "stage4": 16},
    "nnssl": {"stitched_stage3": 8, "stitched_tokens": 8, "stitched_stage4": 16, "stitched_tokens_pool2": 16,
              "stitched_stage5": 32},
}


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


def source_to_input(model: str, meta: dict) -> np.ndarray | None:
    """`model` = a config stem of FAMILY, or a family name (the wrappers' own segmentation inputs)."""
    fam = FAMILY.get(model, model)
    if fam in ("asparagus", "brainfm"):
        return np.linalg.inv(_np(meta["affine"])) @ _np(meta["source_affine"])
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


def cell_coords(m: np.ndarray, stride, src_shape) -> torch.Tensor:
    """[3, *src_shape] continuous map-cell coordinate of every source voxel."""
    s = np.broadcast_to(np.asarray(stride, dtype=np.float64), (3,))
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in src_shape], indexing="ij"), 0).reshape(3, -1)
    inp = m[:3, :3] @ idx + m[:3, 3:]
    return torch.from_numpy(((inp + 0.5) / s[:, None] - 0.5).reshape(3, *src_shape)).float()


def cell_fractions(lab: torch.Tensor, coords: torch.Tensor, grid: tuple[int, ...], k: int) -> torch.Tensor:
    """[k+1, *grid] fraction of each label 0..k among the source voxels falling in each map cell (NaN: none)."""
    cell = coords.round().long()
    ok = ((cell >= 0) & (cell < torch.tensor(grid)[:, None, None, None])).all(0)
    flat = ((cell[0] * grid[1] + cell[1]) * grid[2] + cell[2])[ok]
    counts = torch.zeros(int(np.prod(grid)), k + 1).index_put_((flat, lab[ok].long()), torch.ones(len(flat)),
                                                               accumulate=True)
    return (counts / counts.sum(1, keepdim=True)).T.reshape(k + 1, *grid)


def from_cells(values: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """[C, *grid] cell values -> [C, *src_shape] by trilinear interpolation at each source voxel (border clamped)."""
    grid = torch.tensor(values.shape[1:], dtype=torch.float32, device=values.device)
    c = coords.to(values.device)
    norm = (2 * (c + 0.5) / grid[:, None, None, None] - 1)  # align_corners=False normalisation
    g = norm.flip(0).permute(1, 2, 3, 0)[None]               # grid_sample wants (x=W, y=H, z=D)
    return F.grid_sample(values[None].float(), g, mode="bilinear", padding_mode="border", align_corners=False)[0]


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
