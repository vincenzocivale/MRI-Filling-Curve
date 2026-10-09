"""Trained nnU-Net v2 model folder (MIC-DKFZ/nnUNet) as a frozen encoder, run by nnU-Net's own predictor.

Paths are relative to the nnUNet clone (`cfg["repo"]`); `TS/` = the TotalSegmentator clone (`args.totalseg_repo`).

- Model: `nnUNetPredictor.initialize_from_trained_model_folder` (nnunetv2/inference/predict_from_raw_data.py:68-131)
  reads trainer, configuration and mirroring axes from the checkpoint, builds the net with the trainer's
  `build_network_architecture` (old-format plans converted by plans_handling/plans_handler.py:36-97) and loads it.
  `cfg["checkpoint"]` is the model folder; one fold (feature spaces of different folds are not comparable).
- Optional TotalSegmentator pre-steps (`args.totalseg_resample` = TotalSegmentator's `resample` of the task), as in
  TS/totalsegmentator/nnunet.py:497-561: first volume of 4D input, float copy, `as_closest_canonical`,
  `change_spacing(order=resampling_order, dtype=int32)` (TS/totalsegmentator/resampling.py:194-294), saved as the
  `*_0000.nii.gz` that nnU-Net reads. These lines are glue around TS's own functions (`nnUNet_predict_image` also
  predicts, so it cannot be called for the input alone); equality with TS's file is checked in tests/fm/nnunet_parity.py.
- Preprocessing: the plans' preprocessor `run_case` (preprocessing/preprocessors/default_preprocessor.py:115-143:
  plans reader, transpose_forward, crop to nonzero, per-channel normalisation, resampling), converted exactly like
  inference/data_iterators.py:29-41.
- Inference: `predict_logits_from_preprocessed_data` (predict_from_raw_data.py:501-535) untouched: padding to the
  patch, sliding window, Gaussian, mirroring per checkpoint, CUDA autocast. A forward hook on `network.encoder`
  records the per-stage maps of every network call (tile x mirror) in call order.
- Stitching: each recorded stage map is replayed, nearest-upsampled by the stage stride to the patch grid, through
  nnU-Net's own `predict_sliding_window_return_logits` (:667-713), i.e. the same tiles, mirror flips, Gaussian
  weights, fp16 accumulation and padding removal as the logits. The resulting voxel field (on the preprocessed
  grid) is stored block-averaged over the stage stride (grid origin = preprocessed voxel 0).

Features: `stage{k}` stitched map [C, ceil(shape/stride)], `tiles_stage{k}` raw per-call maps [n_calls, C, patch/stride],
`logits` [classes, *preprocessed shape] (fp16, nnU-Net's buffer dtype). Canonical: the stitched bottleneck.
Derived (ours): `fg_mean_stage{k}`, the mean of the voxel-level stitched field over nnU-Net's nonzero mask.
"""
from __future__ import annotations

import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Wrapper, add_to_path

TS_TRIPLE_SPLIT_VOXELS = 512 * 512 * 900  # TS/totalsegmentator/nnunet.py:569


def block_mean(x: torch.Tensor, stride: list[int]) -> torch.Tensor:
    """[C, *S] -> mean over stride blocks, the last partial block averaged over its voxels only (= avg_pool ceil_mode,
    which refuses axes shorter than the stride, e.g. thin TotalSeg volumes)."""
    pad = [p for s, n in zip(reversed(stride), reversed(x.shape[1:])) for p in (0, -n % s)]
    pool = F.avg_pool3d if len(stride) == 3 else F.avg_pool2d
    ones = torch.ones(1, *x.shape[1:], device=x.device, dtype=x.dtype)
    return pool(F.pad(x, pad)[None], stride, stride)[0] / pool(F.pad(ones, pad)[None], stride, stride)[0]


class _Replay(nn.Module):
    """Stands in for the network inside nnU-Net's sliding window: returns the encoder map recorded at the same
    call of the real run, repeated `stride` times per axis so it covers the patch like the logits do."""

    def __init__(self, maps: list[torch.Tensor], stride: list[int]):
        super().__init__()
        self.maps, self.stride, self.calls = maps, stride, 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calls >= len(self.maps):
            raise RuntimeError("nnU-Net made more network calls than were recorded (OOM fallback?).")
        m = self.maps[self.calls].to(x.device)
        self.calls += 1
        for d, s in enumerate(self.stride):
            m = m.repeat_interleave(s, dim=2 + d)
        return m


def _plain(x: Any) -> Any:
    """numpy / tuples / slices -> python lists, so the result loads with torch.load(weights_only=True)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, slice):
        return [_plain(x.start), _plain(x.stop)]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


class NNUNet(Wrapper):
    """args: fold (0), checkpoint_name (checkpoint_final.pth), tile_step_size (0.5), use_mirroring (True; nnU-Net
    then mirrors only if the checkpoint allows it), stages (null = all encoder stages), return_logits (True),
    totalseg_repo / totalseg_resample (null = plain nnU-Net) / totalseg_resampling_order (1) /
    totalseg_multimodel (False; TS splits huge images only for multi-model tasks)."""

    name = "nnunet"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        add_to_path(cfg["repo"], *([self.args["totalseg_repo"]] if self.args.get("totalseg_repo") else []))
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        a = self.args
        self.predictor = nnUNetPredictor(tile_step_size=a.get("tile_step_size", 0.5), use_gaussian=True,
                                         use_mirroring=a.get("use_mirroring", True), perform_everything_on_device=True,
                                         device=torch.device(device), verbose=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(cfg["checkpoint"], use_folds=(a.get("fold", 0),),
                                                            checkpoint_name=a.get("checkpoint_name",
                                                                                  "checkpoint_final.pth"))
        strides = self.predictor.configuration_manager.network_arch_init_kwargs["strides"]
        dim = len(self.predictor.configuration_manager.patch_size)
        total, self.strides = [1] * dim, []
        for st in strides:
            total = [t * s for t, s in zip(total, [st] * dim if isinstance(st, int) else st)]
            self.strides.append(total)
        pm = self.predictor.plans_manager  # get_configuration is lru-cached: identity gives the checkpoint's name
        self.configuration = next(c for c in pm.plans["configurations"]
                                  if pm.get_configuration(c) is self.predictor.configuration_manager)
        n = len(self.strides)
        self.stages = sorted({s % n for s in (a.get("stages") or range(n))})

    def totalseg_input(self, image: str, out_dir: Path) -> tuple[Path, dict]:
        """TotalSegmentator's steps before nnU-Net (TS/totalsegmentator/nnunet.py:497-561), with TS's functions."""
        import nibabel as nib
        from totalsegmentator.alignment import as_closest_canonical
        from totalsegmentator.resampling import change_spacing

        resample = self.args["totalseg_resample"]
        resample = [resample] * 3 if isinstance(resample, (int, float)) else resample
        img_in_orig = nib.load(image)
        if len(img_in_orig.shape) > 3:
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:, :, :, 0], img_in_orig.affine)
        img_in = as_closest_canonical(nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine))
        img_rsp = change_spacing(img_in, resample, order=self.args.get("totalseg_resampling_order", 1),
                                 dtype=np.int32, nr_cpus=1, use_gpu=self.device.startswith("cuda"))
        if (self.args.get("totalseg_multimodel") and np.prod(img_rsp.shape) > TS_TRIPLE_SPLIT_VOXELS
                and img_rsp.shape[2] > 200):
            raise NotImplementedError("TotalSegmentator would split this image into 3 parts (nnunet.py:573).")
        path = out_dir / "s01_0000.nii.gz"
        nib.save(img_rsp, path)
        geo = {"original_affine": img_in_orig.affine, "original_shape": img_in_orig.shape[:3],
               "canonical_affine": img_in.affine, "canonical_shape": img_in.shape,
               "resampled_affine": img_rsp.affine, "resampled_shape": img_rsp.shape, "resample": resample}
        return path, geo

    def preprocess(self, image: str) -> dict:
        p = self.predictor
        with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp:
            path, geo = Path(image), {}
            if self.args.get("totalseg_resample") is not None:
                path, geo = self.totalseg_input(image, Path(tmp))
            pre = p.configuration_manager.preprocessor_class(verbose=False)
            data, seg, props = pre.run_case([str(path)], None, p.plans_manager, p.configuration_manager,
                                            p.dataset_json)
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        return {"data": data, "seg": seg, "properties": props, "totalseg": geo}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        """nnU-Net's preprocessed volume (after TotalSegmentator's canonical + resampling steps)."""
        from ..bench_geom import source_to_input
        meta = {"totalseg": prepared["totalseg"], "nnunet_properties": prepared["properties"],
                "transpose_forward": self.predictor.plans_manager.transpose_forward,
                "preprocessed_shape": list(prepared["data"].shape[1:])}
        return prepared["data"], source_to_input("nnunet", meta)

    def seg_net(self, n_out: int, pretrained: bool = True):
        """The checkpoint's nnU-Net architecture with n_out outputs: encoder from the trained network, decoder from
        scratch (as nnU-Net's pretrained fine-tuning, pretrainedTrainer.py:167)."""
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        cm = self.predictor.configuration_manager
        net = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                     cm.network_arch_init_kwargs_req_import, 1, n_out, allow_init=True,
                                     deep_supervision=False)
        if pretrained:
            params = self.predictor.list_of_parameters[0]
            net.encoder.load_state_dict({k[len("encoder."):]: v for k, v in params.items() if k.startswith("encoder.")})
        return net, ["encoder"], tuple(cm.patch_size)

    def _stitch(self, maps: list[torch.Tensor], stride: list[int], data: torch.Tensor) -> torch.Tensor:
        """nnU-Net's own sliding-window aggregation applied to the recorded maps -> [C, *data.shape[1:]]."""
        p = self.predictor
        net, lm = p.network, p.label_manager
        p.network = _Replay(maps, stride)
        p.label_manager = SimpleNamespace(num_segmentation_heads=maps[0].shape[1])
        try:
            return p.predict_sliding_window_return_logits(data)
        finally:
            p.network, p.label_manager = net, lm

    @torch.inference_mode()
    def features(self, prepared: dict) -> dict:
        p, data = self.predictor, prepared["data"]
        records: list[list[torch.Tensor]] = []
        hook = p.network.encoder.register_forward_hook(
            lambda _m, _i, out: records.append([out[k].detach().cpu() for k in self.stages]))
        try:
            logits = p.predict_logits_from_preprocessed_data(data)
        finally:
            hook.remove()

        mask = torch.from_numpy(prepared["seg"][0] >= 0)
        feats: dict[str, torch.Tensor] = {}
        for j, k in enumerate(self.stages):
            maps, stride = [r[j] for r in records], self.strides[k]
            field = self._stitch(maps, stride, data)  # [C, *shape] on the results device, fp16
            m = mask.to(field.device)
            feats[f"fg_mean_stage{k}"] = (field * m).flatten(1).float().sum(1).cpu() / m.sum().clamp_min(1).cpu()
            feats[f"stage{k}"] = block_mean(field.float(), stride).cpu()
            feats[f"tiles_stage{k}"] = torch.cat(maps)
            del field
        if self.args.get("return_logits", True):
            feats["logits"] = logits

        cm, patch = p.configuration_manager, p.configuration_manager.patch_size
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        padded, revert = pad_nd_image(data, patch, "constant", {"value": 0}, True, None)
        mirror = p.allowed_mirroring_axes if p.use_mirroring else None
        combos = [c for i in range(len(mirror)) for c in itertools.combinations(mirror, i + 1)] if mirror else []
        bottleneck = len(self.strides) - 1
        meta = {
            "totalseg": prepared["totalseg"],
            "nnunet_properties": prepared["properties"],  # spacing (zyx), affines, crop bbox, shapes
            "transpose_forward": p.plans_manager.transpose_forward,
            "configuration": self.configuration, "trainer": p.trainer_name, "fold": self.args.get("fold", 0),
            "spacing": cm.spacing, "preprocessed_shape": list(data.shape[1:]),
            "nonzero_mask": mask,
            "patch_size": patch, "tile_step_size": p.tile_step_size, "padded_shape": list(padded.shape[1:]),
            "padding_revert": revert[1:],
            "tile_slicers": [s[1:] for s in p._internal_get_sliding_window_slicers(padded.shape[1:])],
            "mirror_axes": mirror, "calls_per_tile": 1 + len(combos), "mirror_combos": [[]] + [list(c) for c in combos],
            "stage_strides": {k: self.strides[k] for k in self.stages},
            "stitched_map": "nnU-Net Gaussian stitching on the preprocessed voxel grid, block mean over stride",
            "autocast": "cuda fp16" if p.device.type == "cuda" else None,
        }
        canonical = f"stage{bottleneck}" if bottleneck in self.stages else f"stage{self.stages[-1]}"
        return {"features": feats, "canonical": canonical, "meta": _plain(meta),
                "derived": [f"fg_mean_stage{k}" for k in self.stages]}


def build(cfg: dict, device: str) -> NNUNet:
    return NNUNet(cfg, device)
