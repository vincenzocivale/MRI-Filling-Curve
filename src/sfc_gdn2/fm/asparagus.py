"""FOMO26 baseline / AMAES ResEnc-U-Net-B (Asparagus), run with the official FOMO26 embedding pipeline.

Checkpoints (Lightning, keys `model.encoder.*` / `model.decoder.*`): AMAES pretraining (MAE, 60% masking,
4^3 units, random 160^3 crops of RAS / native-spacing / not skull-stripped FOMO300K scans) of
`asparagus/modules/networks/resenc_unet.py:143` `resenc_unet_b(use_skip_connections=False)`
(`configs/projects/datapaper/pretrain/resenc_6m_amaes.yaml`). `fomo26` = FOMO26BrainAI/fomo26-baseline
`resenc_unet_b.ckpt` (identical tensors to FOMO-MRI/AMAES_resenc_b_fomo300k); `amaes_fomo260k` = the
FOMO260K run. The `_5ch` file is the 1-channel stem repeated x5 and divided by 5 (`base_module.py:182`
applied offline): no config, and any stem/channel mismatch is rejected here (gardening_tools
`BaseNet.load_state_dict`, `BaseNet.py:22-31`, would silently drop it and keep a random stem).

Everything is the repo's own code: fomo26/fomo-lp `pipeline/embed_all.py` (`cfg["repo"]`, verified at
f915d5a) on top of asparagus @ 16e00d7 (installed in the env; embed_all needs its `late_fusion` / `_encode`,
absent at 72f333b which the HF model cards link) and gardening_tools 0.3.5.
- reorientation (`args.reorient_ras`, default true): NOT part of the repo code. The organisers deliver the
  FOMO26 Task 6/7 data already in RAS (fomo26.github.io FAQ) and FOMO300K was reoriented to RAS by
  asparagus_preprocessing (`utils/nifti.py:32-37`); we reproduce that delivery with the same gardening_tools
  `reorient_nib_image` (`functional/nibabel_utils.py:15-26`), only when the file is not RAS already, and
  hand the RAS volume (float32 NIfTI) to the repo's loader. Recorded in `meta`.
- preprocessing: `embed_all.load_scan` (:81, nibabel get_fdata -> float32, no resampling) and
  `embed_all.preprocess` (:92) = `CPU_clsreg_val_test_transforms_crop(160^3)` (`presets/train.py:158`):
  whole-volume z-score (`gardening_tools/functional/normalization.py:4`), pad to 160 with the volume min
  (`asparagus/modules/transforms/pad.py:253`), centre crop 160^3 (`cropping_and_padding.py:223`).
- network / checkpoint: `embed_all.load_backbone` (:40) = `resenc_unet_b_clsreg(late_fusion=True)`
  (`resenc_unet.py:166`), whose residual shortcuts use MaxPool3d (`resenc_unet.py:59`) where pretraining
  used AvgPool3d (gardening_tools `resunet.py:47`); this is the official path and the default.
- inference / canonical embedding: `embed_all.embed_scan` (:104-120): fp32, single crop, `_encode` ->
  deepest stage (320 ch, stride 32) -> global average pool -> `embedding` [320]. All six encoder stages of
  the same forward (captured from `_encode`, no second pass) are exposed as `stage0..stage5`.
- `args.encoder: pretrain` (opt-in): the pretraining network's own encoder (AvgPool shortcuts), strict load
  of `model.encoder.*`; its repo output is `SelfSupervisedModule.predict_step` (`self_supervised.py:183-186`)
  = `encoder(x)[-1]` (`deepest`, canonical); the GAP `embedding` is then ours (`derived`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .base import Image, Wrapper, add_to_path, git_commit

SIZE = 160
STEM = "encoder.stem.conv1.conv.weight"
STRIDES = (1, 2, 4, 8, 16, 32)


class Asparagus(Wrapper):
    """args: encoder (official | pretrain, default official), reorient_ras (bool, default true),
    target_size (int, default 160 = embed_all's default)."""

    name = "asparagus"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        add_to_path(Path(cfg["repo"]) / "pipeline")
        import embed_all

        self.ea = embed_all
        self.encoder = self.args.get("encoder", "official")
        if self.encoder not in ("official", "pretrain"):
            raise ValueError(f"asparagus: encoder must be 'official' or 'pretrain', got {self.encoder!r}")
        self.reorient_ras = bool(self.args.get("reorient_ras", True))
        self.size = int(self.args.get("target_size", SIZE))
        self.dev = torch.device(device)

        sd = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
        enc = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.encoder.")}
        self._enc = enc
        if enc[STEM].shape[1] != 1:
            raise ValueError(f"asparagus: checkpoint stem has {enc[STEM].shape[1]} input channels; the "
                             f"embedding pipeline is single-modality (1). The _5ch file is not supported.")
        if self.encoder == "official":
            self.model = embed_all.load_backbone(Path(cfg["checkpoint"]), device=self.dev)
            self._skips: list[torch.Tensor] = []
            encode = self.model._encode

            def capture(x: torch.Tensor) -> list[torch.Tensor]:  # records the stages of embed_scan's forward
                self._skips = encode(x)
                return self._skips

            self.model._encode = capture
        else:
            from asparagus.modules.networks.resenc_unet import resenc_unet_b

            self.model = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=1,
                                       use_skip_connections=False)
            torch.nn.Module.load_state_dict(self.model.encoder, {k[len("encoder."):]: v for k, v in enc.items()},
                                            strict=True)
            self.model.eval().to(self.dev)
        got = self.model.state_dict()  # every encoder tensor must be the checkpoint's (no silent drop)
        bad = [k for k, v in enc.items() if k not in got or not torch.equal(got[k].cpu(), v)]
        if bad:
            raise RuntimeError(f"asparagus: {len(bad)} encoder tensors not loaded from the checkpoint, e.g. {bad[:3]}")

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("asparagus: single-modality, pass a NIfTI path")
        img = nib.load(image)
        from gardening_tools.functional.nibabel_utils import get_nib_orientation, reorient_nib_image

        orient = get_nib_orientation(img)
        meta = {"source_affine": torch.from_numpy(img.affine.copy()), "source_shape": tuple(img.shape[:3]),
                "source_orientation": orient, "reoriented_to_ras": False}
        path = Path(image)
        with tempfile.TemporaryDirectory(prefix="asparagus_") as tmp:
            if self.reorient_ras and orient != "RAS":
                ras = reorient_nib_image(img, orient, "RAS")
                path = Path(tmp) / "ras.nii"
                nib.save(nib.Nifti1Image(np.asarray(ras.get_fdata(), dtype=np.float32), ras.affine), path)
                img, meta["reoriented_to_ras"] = ras, True
            scan = self.ea.load_scan(path)
        x = self.ea.preprocess(scan, target_size=(self.size,) * 3)
        # geometry of the network input grid (pad.py:66-78 symmetric pad, cropping_and_padding.py:251-262 floor crop)
        shape = np.array(scan.shape[1:])
        pad_lb = np.maximum(self.size - shape, 0) // 2
        start = (np.maximum(shape, self.size) - self.size) // 2
        offset = start - pad_lb  # input voxel i <-> (RAS) array voxel i + offset
        shift = np.eye(4)
        shift[:3, 3] = offset
        meta |= {"affine": torch.from_numpy(img.affine @ shift),  # voxel -> world of the 160^3 input grid
                 "array_shape": tuple(int(s) for s in shape), "input_offset": tuple(int(o) for o in offset),
                 "spacing": tuple(float(z) for z in img.header.get_zooms()[:3]), "input_shape": tuple(x.shape[1:]),
                 "stage_strides": STRIDES}
        return {"x": x, "meta": meta}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_preprocess(self, image: str) -> dict:
        """asparagus' segmentation data at 1 mm (asparagus_preprocessing `get_iso_preprocessing_config`, the `_ISO`
        variant of its segmentation datasets, configs/preprocessing_presets.py:14-21; the FOMO25 baseline segments at
        1 mm too): RAS as above, size round(spacing / 1 mm * shape) (utils/process_case.py:670), skimage `resize(order=3)`
        (utils/resample.py:46, "yucca"; that module is re-done here since it imports pandas, absent from the env),
        no norm at preprocessing; then the training-time whole-volume z-score (`Torch_Normalize(normalize=True)`,
        CPU_seg_val/test_transforms, presets/train.py:178-196). No crop: training samples patches, inference slides."""
        from gardening_tools.functional.nibabel_utils import get_nib_orientation, reorient_nib_image
        from gardening_tools.modules.transforms.normalize import Torch_Normalize
        from skimage.transform import resize

        from ..bench_geom import _scale

        img = nib.load(image)
        src = img.affine.copy()
        orient = get_nib_orientation(img)
        if self.reorient_ras and orient != "RAS":
            img = reorient_nib_image(img, orient, "RAS")
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        size = np.round(np.array(img.header.get_zooms()[:3], dtype=float) * arr.shape).astype(int)
        arr = resize(arr, output_shape=tuple(size), order=3).astype(np.float32)
        x = Torch_Normalize(normalize=True)({"image": torch.from_numpy(arr)[None], "transforms_applied": {}})["image"]
        return {"x": x, "m": _scale(img.shape[:3], size) @ np.linalg.inv(img.affine) @ src}

    def seg_input(self, prepared: dict):
        return prepared["x"], prepared["m"]

    def seg_net(self, n_out: int, pretrained: bool = True):
        """`resenc_unet_b` with skip connections (the seg_net of configs/model resenc_unet_b), encoder from the
        checkpoint, decoder from scratch (`load_decoder: False`), training patch 160^3 (default_finetune_seg.yaml)."""
        from asparagus.modules.networks.resenc_unet import resenc_unet_b

        net = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=n_out, use_skip_connections=True)
        if pretrained:
            torch.nn.Module.load_state_dict(net.encoder, {k[len("encoder."):]: v for k, v in self._enc.items()},
                                            strict=True)
        return net, ["encoder"], (160, 160, 160)

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        x, meta = prepared["x"], prepared["meta"]
        if self.encoder == "official":
            emb = self.ea.embed_scan(self.model, x, device=self.dev)
            skips, canonical, derived = self._skips, "embedding", []
        else:
            skips = self.model.encoder(x.unsqueeze(0).to(self.dev))
            emb = torch.nn.functional.adaptive_avg_pool3d(skips[-1], (1, 1, 1)).flatten(1).squeeze(0)
            canonical, derived = "deepest", ["embedding"]
        feats = {"embedding": emb, **{f"stage{i}": s[0] for i, s in enumerate(skips)}}
        if self.encoder == "pretrain":
            feats["deepest"] = skips[-1][0]
        return {"features": feats, "canonical": canonical, "derived": derived,
                "meta": meta | {"encoder": self.encoder}}

    def provenance(self) -> dict:
        from importlib.metadata import version

        import asparagus

        return super().provenance() | {"asparagus_commit": git_commit(Path(asparagus.__file__).parents[1]),
                                       "gardening_tools": version("gardening_tools"), "torch": torch.__version__}


def build(cfg: dict, device: str) -> Asparagus:
    return Asparagus(cfg, device)
