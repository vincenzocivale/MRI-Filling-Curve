"""BrainIAC (AIM-KannLab/BrainIAC): SimCLR MONAI ViT-B/16 on 96^3 brain MRI, run with the repo's own code.

All references are to the original repo (`cfg["repo"]`, verified at ba60f45), under `src/`:
- preprocessing (default, `args.preprocessed: false`): `preprocessing/mri_preprocess_3d_simple.py:main`
  (:125), exactly as quickstart.ipynb cell 4 runs it, split in its two steps (`preprocess_cpu` / `preprocess_gpu`): SimpleITK N4 (:74), resampling of the template to
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

import contextlib
import logging
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
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

    def preprocess_cpu(self, image: Image) -> dict:
        """Step 1 of the repo's mri_preprocess_3d_simple.main (:155-180) on a one-file input dir (it globs
        `*.nii.gz` and takes the ID as the name up to the first dot, :49): `registration` = N4 + rigid registration
        (SimpleITK, CPU). The worker runs this ahead in CPU processes; `preprocess_gpu` does the rest."""
        if not isinstance(image, str):
            raise TypeError("brainiac is single-modality: pass a NIfTI path")
        if self.preprocessed:
            return {"source": image, "work": None}
        import mri_preprocess_3d_simple as mp

        work = Path(tempfile.mkdtemp(prefix="brainiac_"))
        ident = Path(image).name.split(".")[0].replace("_mask", "_msk") or "image"  # "_mask" files are skipped (:66)
        (work / "in").mkdir()
        (work / "out" / "temp_registered").mkdir(parents=True)
        (work / "in" / f"{ident}.nii.gz").symlink_to(Path(image).resolve())
        with self._n4(), self._capture_registration() as reg:
            mp.registration(input_dir=str(work / "in"), output_dir=str(work / "out" / "temp_registered"),
                            temp_img=self.template)
        return {"source": image, "work": str(work), "ident": ident, "registration": reg[0] if reg else None}

    @contextlib.contextmanager
    def _capture_registration(self):
        """Keeps the rigid transform the repo's registration finds (fixed = template -> moving = input, LPS mm) as a
        4x4, for mapping segmentations back; the repo discards it after resampling (:129-138). Nothing else changes."""
        import SimpleITK as sitk

        box, orig = [], sitk.ImageRegistrationMethod

        class Capturing(orig):
            def Execute(self, *a):
                t = super().Execute(*a)
                p0 = np.array(t.TransformPoint((0.0, 0.0, 0.0)))
                m = np.eye(4)
                m[:3, :3] = np.stack([np.array(t.TransformPoint(tuple(e))) - p0 for e in np.eye(3)], 1)
                m[:3, 3] = p0
                box.append(m)
                return t

        sitk.ImageRegistrationMethod = Capturing
        try:
            yield box
        finally:
            sitk.ImageRegistrationMethod = orig

    @contextlib.contextmanager
    def _n4(self):
        """`args.n4_shrink` = s (OURS, off by default): the N4 bias field is fitted on the image shrunk by s per axis
        and applied at full resolution, instead of the repo's full-resolution N4 (mri_preprocess_3d_simple.py:74)."""
        s = self.args.get("n4_shrink")
        if not s:
            yield
            return
        import SimpleITK as sitk
        orig = sitk.N4BiasFieldCorrection

        def n4(img, *_, **__):
            f = sitk.N4BiasFieldCorrectionImageFilter()
            f.Execute(sitk.Shrink(img, [int(s)] * img.GetDimension()))
            # `/` is real division (float64): cast back to the input type, as the repo's N4 returns (registration needs it)
            return sitk.Cast(img / sitk.Exp(f.GetLogBiasFieldAsImage(img)), img.GetPixelID())
        sitk.N4BiasFieldCorrection = n4
        try:
            yield
        finally:
            sitk.N4BiasFieldCorrection = orig

    def preprocess_gpu(self, staged: dict) -> dict:
        """Step 2 of main (:182-190): HD-BET (fast, no TTA) on device "0" if CUDA else "cpu", as main picks it;
        then the model input transform."""
        image, work = staged["source"], staged["work"]
        try:
            if work is None:
                log.warning("brainiac: preprocessed=true, skipping N4/registration/HD-BET for %s", image)
                path = Path(image)
            else:
                import mri_preprocess_3d_simple as mp

                out = Path(work) / "out"
                if not (out / "temp_registered" / f"{staged['ident']}_0000.nii.gz").exists():
                    raise RuntimeError(f"brainiac: repo registration produced no output for {image}")
                mp.brain_extraction(input_dir=str(out / "temp_registered"), output_dir=str(out),
                                    device="0" if torch.cuda.is_available() else "cpu")
                path = out / f"{staged['ident']}_0000.nii.gz"
                if not path.exists():
                    raise RuntimeError(f"brainiac: repo preprocessing produced no output for {image}")
                if self.keep_dir:
                    Path(self.keep_dir).mkdir(parents=True, exist_ok=True)
                    path = Path(shutil.copy2(path, Path(self.keep_dir) / path.name))
            x = self.transform({"image": str(path)})["image"]  # MetaTensor [1, 96, 96, 96]
        finally:
            if work:
                shutil.rmtree(work, ignore_errors=True)
        kept = self.preprocessed or bool(self.keep_dir)
        return {"image": x, "source": image, "preprocessed_path": str(path) if kept else None,
                "registration": staged.get("registration")}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        """The 96^3 model input; source voxel -> world (RAS) -> LPS -> inverse of the captured rigid transform ->
        template grid of the registered file (HD-BET keeps it) -> trilinear Resize to 96^3 (align_corners=False)."""
        from ..bench_geom import _scale

        x, reg = prepared["image"], prepared["registration"]
        if reg is None:
            raise RuntimeError("brainiac: no registration transform (preprocessed=true inputs cannot be mapped back)")
        lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        a_reg = np.asarray(x.meta["original_affine"], dtype=np.float64)
        shape = [int(s) for s in x.meta["spatial_shape"]]
        m = (_scale(shape, x.shape[1:]) @ np.linalg.inv(a_reg) @ lps @ np.linalg.inv(reg) @ lps
             @ nib.load(prepared["source"]).affine)
        return x.as_tensor().float(), m

    def seg_net(self, n_out: int, pretrained: bool = True):
        """The repo's segmentation network (segmentation_model.py `ViTUNETRSegmentationModel`: MONAI UNETR, feature_size 16,
        instance norm, res blocks, 96^3) with n_out classes; its ViT from the SimCLR `backbone.*` weights (strict), the
        UNETR conv path from scratch. The repo's `freeze: "yes"` (train_lightning_segmentation.py:32-34) freezes the
        unused standalone `model.vit`, not `unetr.vit`; the protocol freezes `unetr.vit`, the encoder actually run."""
        from monai.networks.nets import UNETR

        net = UNETR(in_channels=1, out_channels=n_out, img_size=(SIZE,) * 3, feature_size=16, hidden_size=768,
                    mlp_dim=3072, num_heads=12, norm_name="instance", res_block=True, dropout_rate=0.0)
        if pretrained:
            sd = torch.load(self.cfg["checkpoint"], map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            net.vit.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")},
                                    strict=True)
        return net, ["vit"], (SIZE,) * 3

    def preprocess(self, image: Image) -> dict:
        return self.preprocess_gpu(self.preprocess_cpu(image))

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
            "n4_shrink": self.args.get("n4_shrink"),  # ours when set: N4 field fitted at 1/s resolution
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
