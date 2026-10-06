"""BrainIAC (AIM-KannLab/BrainIAC): SimCLR MONAI ViT-B/16 on 96^3 brain MRI, run with the repo's own code.

All references are to the original repo (`cfg["repo"]`, verified at ba60f45), under `src/`:
- preprocessing (default, `args.preprocessed: false`): `preprocessing/mri_preprocess_3d_simple.py:main`
  (:125), exactly as quickstart.ipynb cell 4 runs it: SimpleITK N4 (:74), resampling of the template to
  1 mm + rigid Euler3D Mattes-MI registration to `preprocessing/atlases/temp_head.nii.gz` (:76-143), then the
  vendored HD-BET in fast mode, no TTA (:25 -> `preprocessing/HD_BET/hd_bet.py:10`). Registration samples
  1% of voxels at random with a wall-clock seed (:113), so this step is not bitwise reproducible.
  `args.preprocessed: true` skips it, for images already produced by that pipeline (logged).
- model input: `dataset.py:37` `get_validation_transform()` (LoadImaged -> EnsureChannelFirstd ->
  trilinear Resized to 96^3 -> NormalizeIntensityd(nonzero, channel_wise)); no reorientation, the array
  axes of the preprocessed NIfTI are fed as they are.
- network / checkpoint: `load_brainiac.py:4` -> `model.py:7` `ViTBackboneNet` (strict load of `backbone.*`).
- inference: `get_brainiac_features.py:30-38` (eval, no_grad, fp32, one volume per batch).
- canonical embedding: `model.py:44` `features[0][:, 0]`. The MONAI ViT is built without a CLS token
  (classification=False), so this is the first PATCH token (voxel block [0:16,0:16,0:16]) after the final
  LayerNorm; the repo's probes / heads use it (`model.py:56-66`).
Other representations exposed by the same forward: all 216 final tokens (`tokens`, and `map` = the same
tokens as a 768x6x6x6 grid over the 96^3 array axes, C order of the patch-embedding conv) and the
pre-norm outputs of blocks 3/6/9, which UNETR consumes for segmentation (`segmentation_model.py:25-39`).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .base import Image, Wrapper, add_to_path

log = logging.getLogger(__name__)

GRID, PATCH, SIZE = 6, 16, 96
HIDDEN = (3, 6, 9)  # 1-based transformer blocks whose outputs UNETR uses


def plain(t: torch.Tensor) -> torch.Tensor:
    """MetaTensor -> torch.Tensor (the .pt files must load without monai)."""
    return t.as_tensor() if hasattr(t, "as_tensor") else t


class BrainIAC(Wrapper):
    """args: preprocessed (bool, default False), template (path, default the repo's temp_head.nii.gz),
    keep_dir (optional dir where the preprocessed NIfTI is kept; otherwise it is deleted)."""

    name = "brainiac"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        src = Path(cfg["repo"]) / "src"
        add_to_path(src, src / "preprocessing")
        from dataset import get_validation_transform
        from load_brainiac import load_brainiac

        self.transform = get_validation_transform(image_size=(SIZE,) * 3)
        self.model = load_brainiac(cfg["checkpoint"], device=device).eval()
        self.preprocessed = bool(self.args.get("preprocessed", False))
        self.template = str(self.args.get("template") or src / "preprocessing/atlases/temp_head.nii.gz")
        self.keep_dir = self.args.get("keep_dir")

    def _run_repo_preprocessing(self, image: str, work: Path) -> Path:
        """The repo's mri_preprocess_3d_simple.main on a one-file input dir (it globs `*.nii.gz` and takes the
        ID as the name up to the first dot, :49), returning `<out>/<ID>_0000.nii.gz`."""
        import mri_preprocess_3d_simple as mp

        ident = Path(image).name.split(".")[0].replace("_mask", "_msk") or "image"  # "_mask" files are skipped (:66)
        src_dir, out_dir = work / "in", work / "out"
        src_dir.mkdir()
        (src_dir / f"{ident}.nii.gz").symlink_to(Path(image).resolve())
        mp.main(temp_img=self.template, input_dir=str(src_dir), output_dir=str(out_dir))
        out = out_dir / f"{ident}_0000.nii.gz"
        if not out.exists():
            raise RuntimeError(f"brainiac: repo preprocessing produced no output for {image}")
        return out

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("brainiac is single-modality: pass a NIfTI path")
        work = Path(tempfile.mkdtemp(prefix="brainiac_"))
        try:
            if self.preprocessed:
                log.warning("brainiac: preprocessed=true, skipping N4/registration/HD-BET for %s", image)
                path = Path(image)
            else:
                path = self._run_repo_preprocessing(image, work)
                if self.keep_dir:
                    Path(self.keep_dir).mkdir(parents=True, exist_ok=True)
                    path = Path(shutil.copy2(path, Path(self.keep_dir) / path.name))
            x = self.transform({"image": str(path)})["image"]  # MetaTensor [1, 96, 96, 96]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        kept = self.preprocessed or bool(self.keep_dir)
        return {"image": x, "source": image, "preprocessed_path": str(path) if kept else None}

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        x = prepared["image"]
        inputs = default_collate([x]).to(self.device)  # what DataLoader(batch_size=1) feeds: a MetaTensor batch
        out, hidden = self.model.backbone(inputs)
        embedding = self.model(inputs)                         # ViTBackboneNet.forward, model.py:39-46
        if not torch.equal(embedding, out[:, 0]):
            raise RuntimeError("brainiac: ViTBackboneNet.forward != backbone(x)[0][:, 0]")
        tokens = plain(out[0])
        feats = {"embedding": plain(embedding[0]), "tokens": tokens,
                 "map": tokens.transpose(0, 1).reshape(-1, GRID, GRID, GRID)}
        feats |= {f"hidden_{i}": plain(hidden[i - 1][0]) for i in HIDDEN}
        meta = {
            "source": prepared["source"],
            "preprocessed_path": prepared["preprocessed_path"],  # None unless args.keep_dir is set
            "preprocessed": self.preprocessed,
            "template": None if self.preprocessed else self.template,
            # grid of the preprocessed NIfTI (template grid when the repo preprocessing ran)
            "source_shape": tuple(int(s) for s in x.meta["spatial_shape"]),
            "source_affine": torch.as_tensor(np.asarray(x.meta["original_affine"]), dtype=torch.float64),
            # monai Resize(trilinear, align_corners=False) source_shape -> 96^3, independently per axis
            "input_shape": (SIZE,) * 3,
            "input_affine": x.affine.to(torch.float64),
            "patch_size": PATCH, "grid": (GRID,) * 3,
            "token_order": "C order over the 96^3 array axes: token t <-> block (t//36, t//6%6, t%6)",
            "embedding_token": "token 0 = block (0,0,0) after final LayerNorm (no CLS token in this ViT)",
            "hidden": "pre-LayerNorm outputs of transformer blocks 3/6/9 (1-based), [216, 768]",
        }
        return {"features": feats, "canonical": "embedding", "meta": meta}


def build(cfg: dict, device: str) -> BrainIAC:
    return BrainIAC(cfg, device)
