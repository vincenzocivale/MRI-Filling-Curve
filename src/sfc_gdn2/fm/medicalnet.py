"""MedicalNet / Med3D (Tencent/MedicalNet): 3D ResNet-{10..200} encoder, run with the repo's own code.

All references are to the original repo (`cfg["repo"]`, verified at 20f76aa):
- preprocessing: `datasets/brains18.py` `BrainS18Dataset.__getitem__` in phase "test" (:61-76), called as is on a
  one-line image list: `nibabel.load(...).get_data()` (:193; nibabel 4 still has it), nearest-neighbour
  `ndimage.interpolation.zoom(order=0)` of the whole array to (input_D, input_H, input_W) (:150-157), then a
  z-score over voxels > 0 with the zero voxels replaced by N(0, 1) noise (:133-148), -> [1, D, H, W] float32
  (:27-32). No reorientation / resampling / skull stripping: the stored NIfTI array axes are used.
  `input_size` defaults to `setting.py:51-65` (56, 448, 448), the repo's (demo) defaults; the repo defines no
  other size. The background noise is drawn from numpy's global RNG, which the repo never seeds; `seed`
  (default 1 = `setting.py:110` manual_seed) seeds it right before each image so features are reproducible.
- network: `models/resnet.py:217-263` `resnet{depth}` (what `model.py:8-71` `generate_model` calls) with the
  shortcut of the released weights (README "parameters settings", confirmed from the checkpoint keys):
  B for resnet_10 and resnet_50..200, A for resnet_18/34. `no_cuda` follows the device (the type-A shortcut
  creates its zero padding on the CPU when `no_cuda`, `resnet.py:26-37`).
- checkpoint: `{"state_dict": {"module.*"}}` (backbone only, no `conv_seg`). `model.py:88-115` copies the keys that
  match the DataParallel-wrapped net's own names; it only wraps in DataParallel when CUDA is used (:72-85) and
  then also rewrites CUDA_VISIBLE_DEVICES (:80), so the same copy is done here on the unwrapped net by stripping
  `module.`, and checked: every backbone key loaded, only `conv_seg.*` left at its (seeded) random init.
- inference / output: `test.py:48-57` (eval, no_grad, fp32, batch 1, one full-volume forward). The repo's only
  head, `conv_seg`, consumes `layer4` (`resnet.py:204-213`); `layer4` ([512 | 2048, D/8, H/8, W/8], layers 3-4
  dilated, stride 8) is grabbed with a forward hook during the repo's own forward and is the canonical
  feature. The repo has no global embedding: `layer4_gap` (spatial mean) is ours -> `derived`.
"""
from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .base import Image, Wrapper, add_to_path

SHORTCUT = {10: "B", 18: "A", 34: "A", 50: "B", 101: "B", 152: "B", 200: "B"}
STRIDE = 8


class MedicalNet(Wrapper):
    """args: depth (10 | 18 | 34 | 50 | 101 | 152 | 200), input_size ([D, H, W], default [56, 448, 448]),
    seed (int, default 1; null = leave numpy's RNG unseeded, as the repo does)."""

    name = "medicalnet"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        add_to_path(cfg["repo"])
        from models import resnet
        self.depth = int(self.args["depth"])
        self.input_size = tuple(int(s) for s in self.args.get("input_size", (56, 448, 448)))
        self.seed = self.args.get("seed", 1)
        torch.manual_seed(1)  # conv_seg init only (setting.py:110 manual_seed); conv_seg never reaches the features
        self.model = getattr(resnet, f"resnet{self.depth}")(
            sample_input_D=self.input_size[0], sample_input_H=self.input_size[1], sample_input_W=self.input_size[2],
            shortcut_type=SHORTCUT[self.depth], no_cuda=device == "cpu", num_seg_classes=2)
        sd = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        res = self.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=False)
        missing = [k for k in res.missing_keys if not k.startswith("conv_seg.")]
        if missing or res.unexpected_keys:
            raise RuntimeError(f"medicalnet: checkpoint does not match resnet{self.depth}/{SHORTCUT[self.depth]}: "
                               f"missing {missing[:4]}, unexpected {res.unexpected_keys[:4]}")
        self.model.eval().to(device)

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("medicalnet is single-modality: pass a NIfTI path")
        from datasets.brains18 import BrainS18Dataset
        sets = Namespace(input_D=self.input_size[0], input_H=self.input_size[1], input_W=self.input_size[2],
                         phase="test")
        with tempfile.TemporaryDirectory() as tmp:
            img_list = Path(tmp) / "test.txt"
            img_list.write_text(str(Path(image).resolve()) + "\n")  # root_dir is ignored for absolute paths
            dataset = BrainS18Dataset(tmp, str(img_list), sets)
            if self.seed is not None:
                np.random.seed(int(self.seed))
            x = dataset[0]  # np.float32 [1, D, H, W]
        img = nib.load(image)
        return {"image": torch.from_numpy(x), "path": image, "source_shape": tuple(int(s) for s in img.shape[:3]),
                "source_affine": torch.as_tensor(img.affine, dtype=torch.float64)}

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        volume = prepared["image"][None].to(self.device)  # DataLoader(batch_size=1) collation
        grabbed: dict = {}
        hook = self.model.layer4.register_forward_hook(lambda m, i, o: grabbed.__setitem__("layer4", o))
        try:
            self.model(volume)  # ResNet.forward (layer4 -> conv_seg)
        finally:
            hook.remove()
        layer4 = grabbed["layer4"][0].float()
        src = prepared["source_shape"]
        meta = {"input_path": prepared["path"], "source_shape": src, "source_affine": prepared["source_affine"],
                "array_axes": "NIfTI array axes as stored (no reorientation)",
                "input_shape": self.input_size,
                "zoom": tuple(i / s for i, s in zip(self.input_size, src)),
                "resize": "scipy zoom(order=0) source_shape -> input_shape, independently per axis",
                "stride": STRIDE, "grid": tuple(layer4.shape[1:]),
                "background_noise_seed": self.seed,
                "shortcut": SHORTCUT[self.depth]}
        return {"features": {"layer4": layer4, "layer4_gap": layer4.mean((1, 2, 3))}, "canonical": "layer4",
                "meta": meta, "derived": ["layer4_gap"]}


def build(cfg: dict, device: str) -> MedicalNet:
    return MedicalNet(cfg, device)
