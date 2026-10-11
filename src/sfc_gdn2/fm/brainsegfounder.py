"""BrainSegFounder (lab-smile/BrainSegFounder): MONAI Swin-ViT (SwinUNETR encoder), run with the repo's own code.

All references are to the original repo (`cfg["repo"]`, verified at 0bbf432). One wrapper for the stage-1 (UKB)
and ATLAS-pretrain checkpoints; `args.pipeline` picks the code path of the stage that produced the checkpoint:

- `ukb` (stage 1, model_weights_UKB-pretrain.pt): transforms of `pretrain/utils/data_utils.py:147-166`, taken
  from the repo's own `get_T1T2_dataloaders` (modality T1, 1 channel): LoadImaged(NibabelReader) ->
  AddChanneld -> Orientationd(RAS) -> ScaleIntensityRanged(a_min, a_max -> [0, 1], clip) -> SpatialPadd(96) ->
  CropForegroundd(k_divisible=96); the trailing RandSpatialCropSamplesd / ToTensord are dropped.
  The stage-1 a_min/a_max were never shipped (no stage-1 launch script): the argparse defaults -1000/1000
  (`pretrain/main_T1T2.py`, `--a_min/--a_max`, inherited from MONAI's CT recipe) are used and flagged.
- `atlas` (model_weights_ATLAS-pretrain.pt): `downstream/ATLAS/pretrain.py:101-109`:
  bidsio `BIDSLoader.load_sample` (float32 `get_fdata()` with a leading entity axis; it needs a BIDS tree, so the
  two lines are re-done here and checked against `ATLASDataset` in the parity script) -> ToTensor -> Resize(96^3,
  MONAI default mode "area"); raw intensities, no reorientation. ATLAS-pretrain has 2 input channels although
  ATLAS is T1w only; per paper §1.3 ("configuring the two input channels to process the same type of data")
  the T1w is duplicated (`args.duplicate_channels`). The code that produced this checkpoint is not in the repo.

Networks / checkpoints:
- `pretrain/models/ssl_head.py:19-40` `SSLHead`; only its `swinViT` is loaded (strict) from `state_dict`
  (`module.swinViT.*` or `swinViT.*`); the remaining keys must be SSL heads (`rotation_head`, `contrastive_head`,
  `conv`). ATLAS-pretrain's reconstruction decoder (`conv.20` 48->48, `conv.23`) matches no SSLHead in the repo,
  hence encoder-only loading.
- MONAI 1.2.0 is required: MONAI >= 1.5 changed `PatchMerging` (#8285), which changes every stage >= 1.

Inference / features:
- `swinViT(x, normalize=True)` hidden states (`ssl_head.py:81`): `stage0..stage4` (48/96/192/384/768 ch at
  strides 2/4/8/16/32). canonical = `stage4`, the bottleneck SSLHead and the SwinUNETR decoder (`encoder10`)
  consume. `stage4_gap` (spatial mean) is ours -> `derived`.
- `ukb`: the SSL code only samples random 96^3 crops, so the deterministic procedure is MONAI
  `sliding_window_inference` (roi 96, sw_batch_size 1, overlap 0.5 = finetuning `--infer_overlap` default,
  `main_FinetuningSwinUNETR_4Channels.py:84`) over the k_divisible crop. `atlas`: one forward on the 96^3 volume.
  fp32, no AMP (`--noamp` in the launch scripts).

Not wrapped: SwinUNETR-SSL `model_swinvit.pt` (MONAI research-contributions SwinUNETR/Pretrain): a CT model
(5 CT datasets, HU window -1000..1000, 1.5x1.5x2 mm), not a brain-MRI foundation model, and not part of this repo.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import nibabel as nib
import numpy as np
import torch
from torch import nn

from .base import Image, Wrapper, add_to_path, plain

STRIDES = (2, 4, 8, 16, 32)
PIPELINES = {"ukb", "atlas"}
SSL_HEAD_KEYS = ("rotation_head.", "rotation_pre.", "contrastive_head.", "contrastive_pre.", "conv.")


def load_file(name: str, path: Path) -> ModuleType:
    """Import one repo file under a unique name (the repo has three different top-level `utils` packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ssl_args(**kw) -> Namespace:
    """Namespace read by the repo's data loaders (only transform-relevant fields matter)."""
    base = {"rank": 0, "distributed": False, "cache_dataset": False, "smartcache_dataset": False,
            "batch_size": 1, "sw_batch_size": 1, "roi_x": 96, "roi_y": 96, "roi_z": 96, "a_min": -1000.0,
            "a_max": 1000.0, "b_min": 0.0, "b_max": 1.0, "T1T2_10k": False, "T1T2_10k_mixed": False,
            "T1T2_40k_matched": False, "T1T2_target_Brats": False}
    return Namespace(**(base | kw))


class BrainSegFounder(Wrapper):
    """args: pipeline (ukb | atlas), in_channels, depths, feature_size (48), a_min / a_max (ScaleIntensityRanged,
    default -1000 / 1000), roi (sliding-window roi or ATLAS resize, 96), overlap (0.5), sw_batch_size (1),
    duplicate_channels (atlas, 2-channel checkpoint)."""

    name = "brainsegfounder"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        a = self.args
        self.pipeline = a["pipeline"]
        if self.pipeline not in PIPELINES:
            raise ValueError(f"brainsegfounder: pipeline must be one of {sorted(PIPELINES)}")
        self.repo = Path(cfg["repo"])
        add_to_path(self.repo)
        self.in_channels = int(a.get("in_channels", 1))
        self.roi = int(a.get("roi", 96))
        self.overlap = float(a.get("overlap", 0.5))
        self.sw_batch_size = int(a.get("sw_batch_size", 1))
        self.transform = self._build_transform()
        self.model, self.swin = self._build_model()
        self.model.eval().to(device)

    # ---------------------------------------------------------------- preprocessing
    def _build_transform(self):
        from monai import transforms as T
        a = self.args
        if self.pipeline == "atlas":   # ATLAS/pretrain.py:105-108
            return T.Compose([T.ToTensor(), T.Resize(spatial_size=[self.roi] * 3)])
        rng = {"a_min": float(a.get("a_min", -1000.0)), "a_max": float(a.get("a_max", 1000.0)),
               "roi_x": self.roi, "roi_y": self.roi, "roi_z": self.roi}
        with tempfile.TemporaryDirectory() as tmp:  # the loader reads a split json; paths are placeholders
            split = Path(tmp) / "split.json"
            du = load_file("bsf_pretrain_data_utils", self.repo / "pretrain/utils/data_utils.py")
            split.write_text(json.dumps({"training": [{"image": ["t1", "t2"]}],
                                         "validation": [{"image": ["t1", "t2"]}]}))
            args = ssl_args(split_json=str(split), modality="T1", in_channels=1, **rng)
            _, val_loader = du.get_T1T2_dataloaders(args, num_workers=0)
        steps = val_loader.dataset.transform.transforms
        dropped = [type(t).__name__ for t in steps[-2:]]
        if dropped != ["RandSpatialCropSamplesd", "ToTensord"]:
            raise RuntimeError(f"brainsegfounder: unexpected repo transform tail {dropped}")
        return T.Compose(list(steps[:-2]))

    def preprocess(self, image: Image) -> dict:
        if self.pipeline == "atlas":
            img = nib.load(image)
            data = np.zeros((1, *img.shape), dtype=np.float32)  # bidsio BIDSLoader.load_sample
            data[0, ...] = img.get_fdata()
            x = plain(self.transform(data))
            if self.in_channels == 2 and self.args.get("duplicate_channels", True):
                x = torch.cat([x, x])  # paper §1.3: both channels get the same modality
            return {"image": x, "path": image, "source_shape": tuple(img.shape),
                    "source_affine": torch.as_tensor(img.affine, dtype=torch.float64)}
        d = self.transform({"image": image})
        return {"image": d["image"], "path": image,
                "crop_start": d.get("foreground_start_coord"), "crop_end": d.get("foreground_end_coord")}

    # ---------------------------------------------------------------- network
    def _build_model(self) -> tuple[nn.Module, nn.Module]:
        ckpt = torch.load(self.cfg["checkpoint"], map_location="cpu")
        sd = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
        depths = list(self.args.get("depths", [2, 2, 2, 2]))
        head = load_file("bsf_ssl_head", self.repo / "pretrain/models/ssl_head.py")
        model = head.SSLHead(Namespace(spatial_dims=3, in_channels=self.in_channels,
                                       feature_size=int(self.args.get("feature_size", 48)), bottleneck_depth=768,
                                       num_swin_blocks_per_stage=depths, num_heads_per_stage=[3, 6, 12, 24],
                                       dropout_path_rate=0.0, use_checkpoint=False))
        swin_sd = {k.removeprefix("swinViT."): v for k, v in sd.items() if k.startswith("swinViT.")}
        rest = [k for k in sd if not k.startswith("swinViT.") and not k.startswith(SSL_HEAD_KEYS)]
        if rest:
            raise RuntimeError(f"brainsegfounder: unexpected checkpoint keys {rest[:4]}")
        model.swinViT.load_state_dict(swin_sd, strict=True)
        return model, model.swinViT

    # ---------------------------------------------------------------- segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        """The pipeline's network input (ukb: RAS, intensity-scaled, foreground crop; atlas: the 96^3 resize)."""
        from ..bench.geom import _scale

        x = prepared["image"]
        if self.pipeline == "atlas":
            return torch.as_tensor(x).float(), _scale(prepared["source_shape"], x.shape[1:])
        m = np.linalg.inv(x.affine.double().numpy()) @ np.asarray(x.meta["original_affine"], dtype=np.float64)
        return x.as_tensor().float(), m

    def seg_net(self, n_out: int, pretrained: bool = True):
        """The repo's fine-tuning network (SwinUNETR, downstream/ATLAS/finetune.py:115-125, BraTS/finetuning): swinViT
        from the checkpoint, the conv encoders/decoders (incl. the full-resolution `encoder1` on the image) from
        scratch; patch = the roi (96^3; ATLAS: the whole resized volume)."""
        from monai.networks.nets import SwinUNETR

        net = SwinUNETR(img_size=self.roi, in_channels=self.in_channels, out_channels=n_out,
                        feature_size=int(self.args.get("feature_size", 48)), use_checkpoint=False,
                        depths=list(self.args.get("depths", [2, 2, 2, 2])))
        if pretrained:
            net.swinViT.load_state_dict(self.swin.state_dict(), strict=True)
        return net, ["swinViT"], (self.roi,) * 3

    # ---------------------------------------------------------------- inference
    def _predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One window (or the whole ATLAS volume) -> hidden states."""
        return {f"stage{i}": h for i, h in enumerate(self.swin(x.contiguous()))}  # SSLHead.forward, ssl_head.py:81

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        from monai.inferers import sliding_window_inference
        x = prepared["image"]
        inputs = torch.stack([plain(x)]).to(self.device)
        if self.pipeline == "atlas":
            out = self._predict(inputs)
            inference = {"mode": "single forward", "input_size": (self.roi,) * 3}
        else:
            out = sliding_window_inference(inputs, roi_size=[self.roi] * 3, sw_batch_size=self.sw_batch_size,
                                           predictor=self._predict, overlap=self.overlap)
            inference = {"mode": "sliding_window_inference", "roi": (self.roi,) * 3, "overlap": self.overlap,
                         "sw_batch_size": self.sw_batch_size, "blend": "constant"}
        feats = {k: v[0].float() for k, v in out.items()}
        feats["stage4_gap"] = feats["stage4"].mean((1, 2, 3))
        meta = {"pipeline": self.pipeline, "input_path": prepared["path"], "inference": inference,
                "input_shape": tuple(inputs.shape[2:]), "strides": {f"stage{i}": s for i, s in enumerate(STRIDES)},
                "grid": "feature voxel (i,j,k) of stage s covers input voxels [i*stride, (i+1)*stride) per axis; "
                        "with sliding windows the stage grid is int(input_shape / stride) per axis"}
        if self.pipeline == "atlas":
            meta |= {"source_shape": prepared["source_shape"], "source_affine": prepared["source_affine"],
                     "array_axes": "NIfTI array axes as stored (no reorientation)",
                     "resize": "monai Resize(mode='area') source_shape -> input_shape, independently per axis",
                     "channels": "T1w duplicated into both channels" if inputs.shape[1] == 2 else "T1w"}
        else:
            meta |= {"input_affine": x.affine.to(torch.float64),  # input voxel -> world (after reorient/crop)
                     "source_affine": torch.as_tensor(np.asarray(x.meta["original_affine"]), dtype=torch.float64),
                     "source_shape": tuple(int(s) for s in x.meta["spatial_shape"])}
            if prepared.get("crop_start") is not None:
                meta |= {"crop_start": tuple(int(c) for c in prepared["crop_start"]),
                         "crop_end": tuple(int(c) for c in prepared["crop_end"]),
                         "crop": "CropForegroundd(k_divisible=roi): box in the (reoriented) source array; "
                                 "may extend outside it (zero padded)"}
        return {"features": feats, "canonical": "stage4", "meta": meta, "derived": ["stage4_gap"]}


build = BrainSegFounder
