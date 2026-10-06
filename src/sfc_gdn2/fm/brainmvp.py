"""BrainMVP (shaohao011/BrainMVP, CVPR 2025): multi-modal MRI pretraining, UniFormer-S or 3D U-Net encoder.

Everything below is the original repo's code, imported from `cfg["repo"]` (refs @ a466bb2):

- Network = the pretraining `RecModel` the released checkpoints were saved from (`main.py:217-222`):
  `arch: uniformer` -> `models/Uniformer.py:RecModel` (encoder `uniformer_small(in_chans=1)`,
  `models/uniformer_blocks.py:336`); `arch: unet` -> `models/Unet.py:RecModel` (residual GroupNorm U-Net,
  `init_channels=32`, `Unet.py:134,155-177`). Constructor args are the pretraining ones
  (`main.py:23,25`, `do_pretrain.sh`: 8 templates `flair t1 t1c t2 mra pd dwi adc`, template 240x240x155);
  they only shape `rep_template` / the decoder, which are loaded but unused.
- Checkpoint: `{"state_dict": module.*}` (DDP), loaded strictly after the repo's own `module.` strip
  (`main.py:266-270`, there with `strict=False` only because `rep_template` is deleted for resume).
- Input: ONE modality, one channel. Pretraining feeds a single (masked) modality per pass
  (`main.py:23`, `utils/ops.py:70-97`); the 4-channel fine-tuning model drops the pretrained stem
  (`Downstream/train_script.py:75-78`), so the pretrained-stem encoder is the 1-channel one.
- Preprocessing, `mode: downstream` (default): the repo's validation/test transform
  `Downstream/dataset/transforms.py:41-49` (`custom_transform(mode="val")`, imported): LoadImaged,
  ScaleIntensityRangePercentilesd(5, 95 -> 0, 1, channel_wise, no clip; on the raw grid), Orientationd(RAS),
  Spacingd(1 mm, bilinear), CropForegroundd(margin=1). Label-only steps are dropped and the image-only
  steps get `allow_missing_keys`. One addition: `EnsureChannelFirstd` after LoadImaged -- the repo always
  loads a LIST of >=2 modality files, which MONAI stacks into a channel axis; a single file gets none.
  `mode: pretrain`: the pretraining transform `utils/data_utils.py:104-113` (inline in `get_loader`, so
  restated here with the repo's `CenterCropForegroundd`, `utils/custom_trans.py:95`): LoadImaged,
  EnsureChannelFirstd, RAS, 1 mm bilinear, CenterCropForeground, percentiles 5/95 clip=True,
  CenterCropForeground (the random 96^3 crop / pad of training is replaced by the sliding window).
- Inference: the repo's procedure `Downstream/train_utils.py:103-113`: sliding window, roi 96^3
  (`patch_shape`), overlap 0.5, sw_batch_size 1, constant blending, under `torch.cuda.amp.autocast()`.
  The repo's own `Downstream/inference_util.py` assumes output size == window size, so for the
  multi-resolution feature maps MONAI's `sliding_window_inference` (which rescales outputs) is used.
  Caveat (MONAI): the last window of an axis starts at `size - 96`, not on the 16-voxel grid; its outputs
  are written at floor(start / stride).

Output (`features`, all [1, C, h, w, d] on the preprocessed RAS grid, axes (x, y, z) as the repo's decoder
permutes them back, `Downstream/model/Uni_unet.py:68-71`):
- uniformer: `x1` 64@/2, `x2` 128@/4, `x3` 320@/8, `x4` 512@/16 (after the final BatchNorm),
  = `UniFormer.forward` outputs (`Downstream/model/uniformer.py:278-304`) used by the segmentation decoder.
- unet: `c1d` 64@/2, `c2d` 128@/4, `c3d` 256@/8, `c4d` 256@/8 (`models/Unet.py:210-228`).
`canonical` = deepest map (x4 / c4d): the repo defines no volume embedding (no classification code;
the paper resizes to 128x128x64 for classification, without a head spec). `global` (mean of the canonical
map) is ours -> `derived`.
"""
from __future__ import annotations

import types
from pathlib import Path

import torch

from .base import Image, Wrapper, add_to_path

PRETRAIN_ARGS = {"in_channels": 1, "roi_x": 96, "initial_checkpoint": "",
                 "template_index": ["flair", "t1", "t1c", "t2", "mra", "pd", "dwi", "adc"],
                 "dst_h": 240, "dst_w": 240, "dst_d": 155}
STRIDES = {"uniformer": {"x1": 2, "x2": 4, "x3": 8, "x4": 16}, "unet": {"c1d": 2, "c2d": 4, "c3d": 8, "c4d": 8}}


def build_recmodel(repo: str | Path, arch: str, checkpoint: str | Path | None) -> torch.nn.Module:
    """The pretraining RecModel (main.py:217-222), strictly loaded from a released checkpoint."""
    add_to_path(repo)
    if arch == "uniformer":
        from models.Uniformer import RecModel
    elif arch == "unet":
        from models.Unet import RecModel
    else:
        raise ValueError(f"brainmvp: arch must be uniformer|unet, got {arch!r}")
    model = RecModel(types.SimpleNamespace(device="cpu", **PRETRAIN_ARGS), dim=512)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu")["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(sd, "module.")
        model.load_state_dict(sd, strict=True)
    return model.eval()


def encoder_maps(model: torch.nn.Module, arch: str, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """RecModel.encoder on one window -> named maps in (x, y, z) order."""
    if arch == "uniformer":
        _, x1, x2, x3, x4 = model.encoder(x)  # (D, H, W) order inside UniFormer (uniformer_blocks.py:304)
        return {k: v.permute(0, 1, 3, 4, 2) for k, v in zip(STRIDES[arch], (x1, x2, x3, x4))}
    return dict(zip(STRIDES[arch], model.encoder(x)))


def preprocess_transform(repo: str | Path, mode: str):
    import monai.transforms as T
    add_to_path(repo, Path(repo) / "Downstream")
    if mode == "downstream":
        from dataset.transforms import custom_transform
        steps = []
        for t in custom_transform(patch_shape=96, mode="val").transforms:
            if "image" not in t.keys:
                continue  # label-only: EnsureChannelFirstd(label), ConvertToMultiChannelBasedOnBratsClassesd
            t.allow_missing_keys = True
            steps.append(t)
            if isinstance(t, T.LoadImaged):
                steps.append(T.EnsureChannelFirstd(keys=["image"]))
        return T.Compose(steps)
    if mode == "pretrain":
        from utils.custom_trans import CenterCropForegroundd
        return T.Compose([
            T.LoadImaged(keys=["image"]),
            T.EnsureChannelFirstd(keys=["image"]),
            T.Orientationd(keys=["image"], axcodes="RAS"),
            T.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            CenterCropForegroundd(keys=["image"], source_key="image"),
            T.ScaleIntensityRangePercentilesd(keys=["image"], lower=5, upper=95, b_min=0.0, b_max=1.0, clip=True,
                                              channel_wise=True),
            CenterCropForegroundd(keys=["image"], source_key="image"),
        ])
    raise ValueError(f"brainmvp: mode must be downstream|pretrain, got {mode!r}")


class BrainMVP(Wrapper):
    """args: arch (uniformer|unet), mode (downstream|pretrain), roi (96), overlap (0.5), amp (true)."""

    name = "brainmvp"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        self.arch, self.mode = self.args.get("arch", "uniformer"), self.args.get("mode", "downstream")
        self.roi, self.overlap = int(self.args.get("roi", 96)), float(self.args.get("overlap", 0.5))
        self.amp = bool(self.args.get("amp", True))
        self.transform = preprocess_transform(cfg["repo"], self.mode)
        self.model = build_recmodel(cfg["repo"], self.arch, cfg.get("checkpoint")).to(device)

    def preprocess(self, image: Image) -> dict:
        return self.transform({"image": [str(image)]})

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        from monai.inferers import sliding_window_inference
        img = prepared["image"]
        x = img.as_tensor()[None].float().to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp):  # Downstream/train_utils.py:112
            maps = sliding_window_inference(x, roi_size=(self.roi,) * 3, sw_batch_size=1, overlap=self.overlap,
                                            mode="constant", predictor=lambda w: encoder_maps(self.model, self.arch, w))
        maps = {k: v.float() for k, v in maps.items()}
        canonical = list(STRIDES[self.arch])[-1]
        feats = maps | {"global": maps[canonical].mean((2, 3, 4))}
        meta = {
            "affine": img.affine.clone(),                 # voxel -> world (RAS) of the preprocessed input
            "input_shape": tuple(img.shape[1:]),
            "original_affine": torch.as_tensor(img.meta["original_affine"]),
            "spatial_shape": tuple(int(s) for s in img.meta["spatial_shape"]),
            "strides": STRIDES[self.arch],                # input voxels per feature voxel, per axis
            "axes": "x,y,z of the preprocessed RAS 1 mm grid",
            "mode": self.mode, "arch": self.arch,
            "sliding_window": {"roi": self.roi, "overlap": self.overlap, "blend": "constant", "amp": self.amp},
        }
        return {"features": feats, "canonical": canonical, "meta": meta, "derived": ["global"]}


def build(cfg: dict, device: str) -> BrainMVP:
    return BrainMVP(cfg, device)
