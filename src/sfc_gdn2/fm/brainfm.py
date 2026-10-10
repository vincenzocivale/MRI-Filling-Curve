"""BrainFM (jhuldr/BrainFM): modality-agnostic multi-task 3D U-Net, run with the repo's own code.

All references are to the original repo (`cfg["repo"]`, verified at f1731a1):
- preprocessing: `utils/test_utils.py:235` `prepare_image` with the arguments of `scripts/demo_test.py:46`
  (win_size=None, zero_crop_first=True, spacing=None, add_bf=False): nibabel read + nan_to_num (:238-240),
  CT clamp to [0, 80] only if `args.is_ct` (:246), min-max to [0, 1] (:249-251), `utils/misc.py:1117`
  `torch_resize` to 1 mm isotropic (Gaussian anti-aliasing when downsampling), `misc.py:1207`
  `align_volume_to_ref` to the identity (RAS) orientation, then `zero_crop` (:60) to the bounding box of
  voxels > 0. We call prepare_image with zero_crop_first=False and apply the repo's `zero_crop` with the
  same bbox ourselves -- `center_crop(win_size=None, zero_crop_first=True)` (:155-165) does exactly this --
  only to record the bbox: the repo does not shift the returned affine by the crop offset (bug at
  :155-165 / :258), and the affine it returns is the RAS-aligned one while the image is NOT reoriented (:272-276
  align `final` with the already aligned affine, a no-op): `meta["affine"]` is the true one
  (`bench_geom.brainfm_input_affine`: 1 mm resize in source axis order, then the crop; checked against the shape). The tensor is unchanged (see the parity script).
- network: `Trainer/models/__init__.py:404` `build_model` from `cfgs/generator/default.yaml` +
  `cfgs/generator/test/demo_test.yaml` and `cfgs/trainer/default_train.yaml` + `default_val.yaml` +
  `cfgs/trainer/test/demo_test.yaml` (UNet3D, f_maps 64, 6 levels, 'gcl', unit_feat). The repo's own
  default paths are broken (`test_utils.py:28-37` -> `cfg/defaults/*`, and `utils/process_cfg.py:60`
  rejects relative paths with cfg_dir=''), so the same files are passed with absolute paths.
- checkpoint: `utils/checkpoint.py:409` `load_checkpoint` (strict=False with suffix matching); we then
  assert every checkpoint tensor landed in the model. The only model tensors not in the checkpoint are
  `head.final_conv_high_res_residual.*`: the demo config enables super_resolution but the released model
  was trained without it, so that head is random and `high_res*` outputs are never returned.
- inference: the body of `test_utils.py:290` `evaluate_image` (forward, processors, postprocessor; fp32,
  no_grad, whole volume, no TTA), split so the model is built once instead of per image.
- features: `outputs['feat']` = `Trainer/models/unet3d/model.py:195` `get_feature`, the decoder pyramid
  [bottleneck 2048 ch @ /32, 1024 @ /16, 512 @ /8, 256 @ /4, 128 @ /2, 64 @ 1 mm]; the last one is
  L2-normalised over channels (:207) and is what `scripts/demo_get_feature.py:31` returns and what every
  task head (1x1 convs, `Trainer/models/head.py:40`) consumes -> canonical `feat_last`.
  The repo defines no volume-level embedding (its age head was never released), so
  `feat_last_fgmean` (mean of feat_last over input > 0, the repo's mask convention `demo_test.py:53`)
  is ours and listed in `derived`.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..bench_geom import brainfm_input_affine
from .base import Image, Wrapper, add_to_path

GEN_CFGS = ("cfgs/generator/default.yaml", "cfgs/generator/test/demo_test.yaml")
TRAIN_CFGS = ("cfgs/trainer/default_train.yaml", "cfgs/trainer/default_val.yaml", "cfgs/trainer/test/demo_test.yaml")
UNTRAINED = ("head.final_conv_high_res_residual.",)
UNTRAINED_OUTPUTS = ("high_res", "high_res_residual")


class BrainFM(Wrapper):
    """args: is_ct (bool, default False: CT clamp in prepare_image), levels (bool, default True: also
    return feat_0..feat_4), task_outputs (bool, default False: also return the trained task heads'
    outputs -- synthesised T1/T2/FLAIR/CT, segmentation probabilities + label, distance maps, bias
    field, atlas coordinates regx/y/z -- all at the feat_last grid)."""

    name = "brainfm"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        self.repo = Path(cfg["repo"]).resolve()
        add_to_path(self.repo)
        with contextlib.chdir(self.repo):  # test_utils reads files/gca.mgz relative to the cwd at import (:40)
            import utils.test_utils as tu
        from Trainer.models import build_model
        from utils import misc
        from utils.checkpoint import load_checkpoint

        self.tu = tu
        self.gen_args = misc.preprocess_cfg([str(self.repo / p) for p in GEN_CFGS])
        self.train_args = misc.preprocess_cfg([str(self.repo / p) for p in TRAIN_CFGS])
        self.gen_args, self.train_args, self.model, self.processors, _, self.postprocessor = build_model(
            self.gen_args, self.train_args, device)
        load_checkpoint(cfg["checkpoint"], [self.model], model_keys=["model"], to_print=False)
        self._check_loaded(cfg["checkpoint"])
        self.is_ct = bool(self.args.get("is_ct", False))
        self.levels = bool(self.args.get("levels", True))
        self.task_outputs = bool(self.args.get("task_outputs", False))

    def _check_loaded(self, path: str) -> None:
        """load_checkpoint is strict=False: make sure every model tensor (but the untrained SR head) is the
        checkpoint's."""
        sd = torch.load(path, map_location="cpu")["model"]
        model_sd = self.model.state_dict()
        missing = [k for k in model_sd if k not in sd and not k.startswith(UNTRAINED)]
        differ = [k for k, v in sd.items() if k not in model_sd or not torch.equal(model_sd[k].cpu(), v)]
        if missing or differ:
            raise RuntimeError(f"brainfm: checkpoint not fully loaded: missing {missing[:4]}, differing {differ[:4]}")

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("brainfm is single-modality: pass a NIfTI path")
        final, _, _, _, aff, _, _ = self.tu.prepare_image(image, win_size=None, zero_crop_first=False, spacing=None,
                                                          add_bf=False, is_CT=self.is_ct, device=self.device)
        vol = final[0, 0]
        coords = torch.argwhere(vol > 0)  # zero_crop's own bbox rule (tol=0), test_utils.py:69-77
        lo, hi = coords.min(0)[0].tolist(), (coords.max(0)[0] + 1).tolist()
        x = self.tu.zero_crop(vol, crop_range_lst=[lo, hi])[None, None]
        src = nib.load(image)
        corrected = brainfm_input_affine(src.affine, src.shape, lo)
        size = np.round(np.asarray(src.shape[:3]) * np.sqrt((src.affine[:3, :3] ** 2).sum(0))).astype(int)
        if tuple(size) != tuple(vol.shape):
            raise RuntimeError(f"brainfm: network input {tuple(vol.shape)} is not the 1 mm resize {tuple(size)} "
                               "in source axis order; the geometry below would be wrong")
        meta = {
            "input_path": image,
            "source_shape": tuple(int(s) for s in src.shape),
            "source_affine": torch.as_tensor(src.affine, dtype=torch.float64),
            "aligned_shape": tuple(vol.shape),               # 1 mm RAS grid before the zero crop
            "aligned_affine": torch.as_tensor(np.asarray(aff, dtype=np.float64)),
            "bbox": [lo, hi],                                # zero crop in aligned voxels, [start, stop)
            "input_shape": tuple(x.shape[2:]),
            "affine": torch.as_tensor(corrected),            # input / feat_last voxel -> world (mm)
            "spacing": (1.0, 1.0, 1.0),
            "is_ct": self.is_ct,
        }
        return {"input": x, "meta": meta}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    seg_bf16 = False  # the repo infers in fp32; its torch 2.0 has no bf16 upsample_nearest3d

    def seg_input(self, prepared: dict):
        """The 1 mm zero-cropped volume the network sees, in the SOURCE axis order (bench_geom.brainfm_input_affine)."""
        from ..bench_geom import source_to_input
        return prepared["input"][0].float().cpu(), source_to_input("brainfm", prepared["meta"])

    def seg_net(self, n_out: int, pretrained: bool = True):
        """The repo's task head on the frozen backbone: `TaskHead` with task_f_maps [64] (cfgs/trainer/default_train.yaml:26)
        = one 1x1 conv on feat_last (head.py:40), here with n_out classes; training crop 128^3 (cfgs/generator/
        default.yaml:63). random: the backbone re-initialised with each layer's reset_parameters.
        The whole UNet is frozen, so the head is a full-resolution linear probe: a BatchNorm without affine goes before it
        (MAE's linear-probe convention; same function class). Without it feat_last's scale (spatial std ~0.05/channel)
        stalls AdamW at lr 1e-3: tumour Dice 0.014, below the random backbone's 0.197."""
        import copy

        head = torch.nn.Sequential(torch.nn.BatchNorm3d(64, affine=False),  # before the backbone reset: same head
                                   torch.nn.Conv3d(64, n_out, 1))           # init in both builds
        backbone = copy.deepcopy(self.model.backbone)
        if not pretrained:
            for mod in backbone.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()
        return _FeatLastHead(backbone, head), ["backbone"], (128, 128, 128)

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        x = prepared["input"]
        samples = [{"input": x}]
        outputs, _ = self.model(samples)                     # evaluate_image, test_utils.py:302-307
        for processor in self.processors:
            outputs = processor(outputs, samples)
        outputs, _, _ = self.postprocessor(self.gen_args, self.train_args, outputs, samples, target=None,
                                           feats=None, tasks=self.gen_args.tasks)
        out = outputs[0]
        feat = out["feat"]
        feats = {"feat_last": feat[-1][0]}
        if self.levels:
            feats |= {f"feat_{i}": f[0] for i, f in enumerate(feat[:-1])}
        if self.task_outputs:
            feats |= {f"task_{k}": v[0] for k, v in out.items()
                      if k != "feat" and k not in UNTRAINED_OUTPUTS and isinstance(v, torch.Tensor)}
        fg = (x[0, 0] > 0).to(feats["feat_last"].dtype)
        feats["feat_last_fgmean"] = (feats["feat_last"] * fg).sum((1, 2, 3)) / fg.sum()
        meta = prepared["meta"] | {
            "levels": {f"feat_{i}": tuple(f.shape[1:]) for i, f in enumerate(feat)},
            "level_note": "feat_i (i<5) is at stride 2**(5-i) of the input grid via floor MaxPool3d(2); "
                          "feat_last == feat_5 is on the input grid (meta['affine'])",
        }
        return {"features": feats, "canonical": "feat_last", "meta": meta, "derived": ["feat_last_fgmean"]}


class _FeatLastHead(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, x):
        return self.head(self.backbone.get_feature(x)[-1])  # joiner.py:180 -> head.py:40


def build(cfg: dict, device: str) -> BrainFM:
    return BrainFM(cfg, device)
