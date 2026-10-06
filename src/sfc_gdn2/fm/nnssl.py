"""OpenMind (ResEnc-L / Primus-M x MAE, MG, S3D, SimCLR, SimMIM, SwinUNETR, VF, VoCo) and nnFoundation (CNN = ResEnc-L,
ViT = Primus-X) checkpoints of MIC-DKFZ/nnssl @ 94fe12d (branch nnFoundation), run through the official nnssl
preprocessing and the official nnU-Net downstream-adaptation code (MIC-DKFZ/nnUNet @ 2edf346).

Every step is the original code (`cfg["repo"]` = nnssl clone, `args.nnunet_repo`, `args.ssl3d_repo`):

- preprocess: the checkpoint's own embedded plan (`nnssl_adaptation_plan.pretrain_plan`, `Plan.from_dict`,
  experiment_planning/experiment_planners/plan.py:139) -> its reader (SimpleITKIO, no reorientation,
  imageio/simpleitk_reader_writer.py:33-46) -> `preprocess_case` (OpenMind `onemmiso`: crop to nonzero, z-score of
  the whole crop with use_mask_for_norm=False, cubic resampling to 1 mm; preprocessing/preprocessors/
  default_preprocessor.py:40-108) or `no_resample_preprocess_case` (nnFoundation `noresample`,
  no_resampling_preprocessor.py:23-71); dispatch as default_preprocessor.py:223-229. We pass `masks=[]` instead of
  None: crop_to_nonzero then returns its own nonzero mask resampled with the plan's mask resampler
  (cropping/cropping.py:41-47, default_preprocessor.py:95-100); the image is bit-identical (asserted in the parity
  script) and the mask is only used to centre the SSL3D crop and returned in `meta`.
- network: the official downstream builders at the plan's `recommended_downstream_patchsize` (160^3 OpenMind,
  192^3 nnFoundation): `PretrainedTrainer.build_network_architecture` for ResEnc-L, `PretrainedTrainer_Primusx`'s for
  Primus (nnunetv2/training/nnUNetTrainer/pretraining/pretrainedTrainer.py:344-397, thrp_primusx_finetuning.py:
  131-173), with `num_output_channels=1` (only the decoder head, which we never use, depends on it).
- weights: `PretrainedTrainer.load_pretrained_weights` (pretrainedTrainer.py:151-342) with the plan's keys, exactly
  as `nnUNetv2_preprocess_like_nnssl` + `nnUNetv2_train_pretrained` pass them (like_nnssl.py:131-145, pretrainedTrainer
  .py:106-117). It handles the `encoder.`-prefixed Primus-M SimCLR/VoCo/SwinUNETR checkpoints and trilinearly
  interpolates the 64^3-trained SimCLR/VoCo pos-embed to 20^3 tokens (l.229-233, 304-313). Two fixes, both forced:
  (a) it calls `torch.load(path, weights_only=True)` without map_location (l.204) and the OpenMind files hold CUDA
  tensors, so `torch.load` gets map_location="cpu" during the call (network moved to `device` after);
  (b) the plan EMBEDDED in PrimusM-SimCLR / PrimusM-VoCo states the pre-training patch [192,192,64] / [256,256,64]
  (the trainer's sampling patch) while their pos-embed has 512 = (64/8)^3 tokens; like_nnssl.py:139 forwards that
  value and the resize then raises (load_weights_utils.py:212; SSL3D eva_mae_openneuro.py:249-264 would too). The
  `adaptation_plan.json` published next to each checkpoint (HF) states [64,64,64]; the pre-training patch is taken
  from it when present (the two plans differ in nothing else for all 16 checkpoints, checked in the parity script).
- inference: nnU-Net's sliding window (`nnUNetPredictor` with tile_step_size=0.5, Gaussian, `pad_nd_image` with 0,
  `_internal_get_sliding_window_slicers`, fp16 autocast on CUDA / fp32 on CPU; inference/predict_from_raw_data.py:
  537-571, 666-711). No mirroring TTA: flipped feature maps are not channel-equivariant, the flip-averaging is only
  defined for logits. Encoder outputs: ResEnc-L `network.encoder(x)` = all 6 stages (strides 1..32); Primus the
  `eva` output (final LayerNorm'ed tokens, captured with a forward hook during the original `Primus.forward`,
  dynamic_network_architectures primus.py:163-185) reshaped `b (w h d) c -> b c w h d` as at primus.py:185.
- volume embedding: the OpenMind linear-probe definition of constantinulrich/SSL3D_classification @ 848e22e (the
  classification framework linked from nnssl documentation/openmind.md): frozen encoder, ONE crop of the
  recommended size centred on the mask (`get_mask_center`, `crop_center_with_padding_np`, datasets/
  preprocess_3D_data/crop_to_mask.py:18-59), mean over space of the last stage (models/resenc.py:62) / over all
  tokens (models/classification_head.py:37; its `x[:, 1:]` drops a token only because primus.yaml sets
  cls_token_available although EvaEncoder has none). SSL3D's own HD-BET/1 mm/brain-mask z-score preprocessing
  (template_brain_preprocessing.py:45-107) is NOT used: the input is nnssl's preprocessing above.

Outputs (`features`):
  crop_last [C,h,w,d], crop_gap_last [C]       -> SSL3D linear-probe input; `canonical` = crop_gap_last for every
                                                  checkpoint. SSL3D defines it for OpenMind; nnFoundation has no
                                                  official volume-level probe yet (model card: "Classification: TBA"),
                                                  the same definition is applied at its recommended 192^3 patch.
  tiles_<lvl> [T,C,h,w,d] (args.keep_tiles)    -> raw sliding-window encoder outputs (nnU-Net procedure).
  stitched_<lvl> [C,H,W,D], gap_last [C]       -> derived (ours): Gaussian-weighted stitching of tile maps on the
                                                  padded volume's feature grid (ceil(size/stride)), tile voxel
                                                  offsets mapped linearly onto [0, grid - tile] and rounded (so the
                                                  last tile ends on the last cell); mean over tiles of the per-tile GAP.
  <lvl> = stage{i} (ResEnc-L, args.stages, default [3, 4, 5] = strides 8/16/32) or tokens (Primus, stride 8).
"""
from __future__ import annotations

import contextlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from .base import Image, Wrapper, add_to_path


@contextlib.contextmanager
def _torch_load_on_cpu():
    """pretrainedTrainer.py:204 loads without map_location; the OpenMind checkpoints store CUDA tensors."""
    orig = torch.load

    def load(*a, **kw):
        kw.setdefault("map_location", "cpu")
        return orig(*a, **kw)

    torch.load = load
    try:
        yield
    finally:
        torch.load = orig


def _plain(x):
    """numpy scalars / arrays -> python types, so the result loads with torch.load(weights_only=True)."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_plain(v) for v in x)
    return x.tolist() if isinstance(x, (np.generic, np.ndarray)) else x


class NnsslWrapper(Wrapper):
    """args: nnunet_repo, ssl3d_repo, stages (ResEnc levels kept, default [3,4,5]), keep_tiles (default true)."""

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        self.name = cfg["model"]
        add_to_path(Path(cfg["repo"]) / "src", self.args["nnunet_repo"], self.args["ssl3d_repo"])
        from nnssl.experiment_planning.experiment_planners.plan import Plan
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer
        from nnunetv2.training.nnUNetTrainer.pretraining.thrp_primusx_finetuning import (
            PretrainedTrainer_Primusx,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

        with _torch_load_on_cpu():
            ckpt = torch.load(cfg["checkpoint"], weights_only=True)
        ap = ckpt["nnssl_adaptation_plan"]
        self.plan = Plan.from_dict(deepcopy(ap["pretrain_plan"]))
        (self.config_name, self.config_plan), = self.plan.configurations.items()
        self.patch = list(ap["recommended_downstream_patchsize"])
        shipped = Path(cfg["checkpoint"]).with_name("adaptation_plan.json")
        self.pt_patch_source = str(shipped) if shipped.exists() else "checkpoint['nnssl_adaptation_plan']"
        src_plan = json.loads(shipped.read_text())["pretrain_plan"] if shipped.exists() else ap["pretrain_plan"]
        self.pt_patch = list(next(iter(src_plan["configurations"].values()))["patch_size"])
        arch = ap["architecture_plans"]
        self.arch = arch["arch_class_name"]
        self.is_primus = self.arch.startswith("Primus")
        trainer = PretrainedTrainer_Primusx if self.is_primus else PretrainedTrainer
        # like_nnssl.py:124-126,169-172 -> ConfigurationManager.network_arch_* -> pretrainedTrainer.py:88-96
        self.arch_details = {"network_class_name": self.arch, "arch_kwargs": arch["arch_kwargs"],
                             "_kw_requires_import": arch["arch_kwargs_requiring_import"]}
        net = trainer.build_network_architecture(
            architecture_class_name=self.arch, arch_init_kwargs=deepcopy(arch["arch_kwargs"]),
            arch_init_kwargs_req_import=arch["arch_kwargs_requiring_import"], input_patch_size=self.patch,
            num_input_channels=1, num_output_channels=1, enable_deep_supervision=False)
        with _torch_load_on_cpu():
            net, _ = trainer.load_pretrained_weights(
                net, pretrained_weights_path=cfg["checkpoint"], pt_input_channels=ap["pretrain_num_input_channels"],
                downstream_input_channels=1, pt_input_patchsize=self.pt_patch,
                downstream_input_patchsize=self.patch, pt_key_to_encoder=ap["key_to_encoder"],
                pt_key_to_stem=ap["key_to_stem"], pt_keys_to_in_proj=tuple(ap["keys_to_in_proj"]),
                pt_key_to_lpe=ap["key_to_lpe"])
        self.network = net.to(device).eval()
        self.adaptation_plan, self.trainer_name = ap, ckpt.get("trainer_name")
        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                         device=torch.device(device), allow_tqdm=False)
        self.predictor.configuration_manager = ConfigurationManager(
            {"patch_size": self.patch, "architecture": self.arch_details})
        self.stages = [int(s) for s in self.args.get("stages", [3, 4, 5])]
        self.keep_tiles = bool(self.args.get("keep_tiles", True))
        if self.is_primus:
            self._tokens = None
            self.network.eva.register_forward_hook(lambda m, i, o: setattr(self, "_tokens", o[0]))

    # ------------------------------------------------------------------ preprocessing (nnssl)
    def preprocess(self, image: Image) -> dict:
        from nnssl.preprocessing.preprocessors.default_preprocessor import preprocess_case
        from nnssl.preprocessing.preprocessors.no_resampling_preprocessor import no_resample_preprocess_case

        path = image if isinstance(image, str) else image["image"]
        data, props = self.plan.image_reader_writer_class()().read_images([path])
        if self.config_plan.spacing_style == "noresample":
            fn = no_resample_preprocess_case
        elif self.config_plan.spacing_style in ("onemmiso", "median"):
            fn = preprocess_case
        else:
            raise NotImplementedError(self.config_plan.spacing_style)
        data, masks = fn(data, [], props, self.plan, self.config_plan, False)
        return {"data": data, "nonzero": masks[0][0] >= 0, "props": props}

    # ------------------------------------------------------------------ network
    def _encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Original forward on one [1,1,*patch] input -> {level: [C,h,w,d]}."""
        if self.is_primus:
            self.network(x)
            w, h, d = (s // p for s, p in zip(x.shape[2:], self.network.down_projection.proj.kernel_size))
            return {"tokens": rearrange(self._tokens, "b (w h d) c -> b c w h d", w=w, h=h, d=d)[0]}
        skips = self.network.encoder(x)
        return {f"stage{i}": skips[i][0] for i in sorted({*self.stages, len(skips) - 1})}

    @torch.inference_mode()
    def features(self, prepared: dict) -> dict:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from datasets.preprocess_3D_data.crop_to_mask import crop_center_with_padding_np, get_mask_center
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.utilities.helpers import dummy_context

        dev = torch.device(self.device)
        amp = torch.autocast(dev.type, enabled=True) if dev.type == "cuda" else dummy_context()
        vol = torch.from_numpy(prepared["data"])
        # predict_from_raw_data.py:690-694
        padded, revert = pad_nd_image(vol, self.patch, "constant", {"value": 0}, True, None)
        slicers = self.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        tiles: dict[str, list[torch.Tensor]] = {}
        with amp:
            for sl in slicers:
                for k, v in self._encode(padded[sl][None].to(dev)).items():
                    tiles.setdefault(k, []).append(v.float().cpu())
            center = get_mask_center(prepared["nonzero"].astype(np.uint8))
            crop = crop_center_with_padding_np(prepared["data"][0], center, tuple(self.patch))
            crop_out = self._encode(torch.from_numpy(crop)[None, None].to(dev))
        last = "tokens" if self.is_primus else f"stage{max(int(k[5:]) for k in tiles)}"

        feats, strides = {}, {}
        for k, ts in tiles.items():
            t = torch.stack(ts)                                         # [T,C,h,w,d]
            stride = [p // f for p, f in zip(self.patch, t.shape[2:])]
            strides[k] = stride
            grid = [-(-s // st) for s, st in zip(padded.shape[1:], stride)]
            acc, wsum = torch.zeros(t.shape[1], *grid), torch.zeros(grid)
            g = compute_gaussian(tuple(t.shape[2:]), sigma_scale=1. / 8, value_scaling_factor=10,
                                 dtype=torch.float32, device=torch.device("cpu"))
            for ti, sl in zip(t, slicers):
                o = [round(s.start * (n - f) / (v - p)) if v > p else 0
                     for s, n, f, v, p in zip(sl[1:], grid, t.shape[2:], padded.shape[1:], self.patch)]
                win = tuple(slice(a, a + f) for a, f in zip(o, t.shape[2:]))
                acc[(slice(None), *win)] += ti * g
                wsum[win] += g
            feats[f"stitched_{k}"] = acc / wsum
            if self.keep_tiles or k == last:
                feats[f"tiles_{k}"] = t
        feats["gap_last"] = feats[f"tiles_{last}"].mean((2, 3, 4)).mean(0)
        feats["crop_last"] = crop_out[last].float().cpu()
        feats["crop_gap_last"] = feats["crop_last"].mean((1, 2, 3))

        derived = [k for k in feats if k.startswith("stitched_")] + ["gap_last"]
        canonical = "crop_gap_last"
        props = prepared["props"]
        meta = {
            "arch": self.arch, "trainer": self.trainer_name,
            "plan_config": self.config_name, "spacing_style": self.config_plan.spacing_style,
            "target_spacing": self.config_plan.spacing, "transpose_forward": self.plan.transpose_forward,
            "normalization": self.config_plan.normalization_schemes,
            "use_mask_for_norm": self.config_plan.use_mask_for_norm,
            # SimpleITK array axes (z, y, x); spacing / origin / direction in that order as nnssl stores them
            "sitk_stuff": props["sitk_stuff"], "spacing": props["spacing"],
            "shape_before_cropping": tuple(props["shape_before_cropping"]),
            "bbox_used_for_cropping": props["bbox_used_for_cropping"],
            "shape_after_cropping_and_before_resampling": tuple(props["shape_after_cropping_and_before_resampling"]),
            "preprocessed_shape": tuple(prepared["data"].shape[1:]),
            "pretrain_patch_size": self.pt_patch, "pretrain_patch_source": self.pt_patch_source,
            "patch_size": self.patch, "tile_step_size": self.predictor.tile_step_size,
            "padded_shape": tuple(padded.shape[1:]), "revert_padding": [(s.start, s.stop) for s in revert[1:]],
            "tile_starts": [tuple(s.start for s in sl[1:]) for sl in slicers], "strides": strides,
            "crop_center": tuple(int(c) for c in center),
            "precision": "fp16-autocast" if dev.type == "cuda" else "fp32",
            "canonical_definition": "SSL3D_classification linear probe: GAP of the last encoder stage / all tokens "
                                    "on one recommended-size crop centred on the nonzero mask"
                                    + ("" if self.name == "openmind" else " (applied by analogy: no official "
                                       "nnFoundation probe)"),
        }
        return {"features": feats, "canonical": canonical, "meta": _plain(meta), "derived": derived}


def build(cfg: dict, device: str) -> NnsslWrapper:
    return NnsslWrapper(cfg, device)
