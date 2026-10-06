"""BrainFM / Brain-ID (jhuldr/BrainFM, Apache-2.0) as a frozen feature extractor.

The released model is a 3D U-Net backbone (`Trainer/models/unet3d/model.py::UNet3D`, DoubleConv
blocks with GroupNorm + conv + LeakyReLU, nearest upsampling, concatenation skips) followed by task
heads. The repo is not pip-installable and its imports pull in training-only dependencies (h5py,
visdom, iopath, ...), so the backbone is re-implemented below in pure torch with the *same module
names*, hence the same state_dict keys.

Checkpoint (`ckp/brainfm_pretrained.pth`, from the project's OneDrive): `torch.save({"model": joiner.state_dict(),
"epoch": ..., ...})` (scripts/train.py:207); the repo reads the first key containing "model"
(utils/checkpoint.py:430-452). Backbone weights sit under `backbone.`, task heads under `head.` /
`head_dict.` (ignored here). The backbone is loaded strictly.

Repo conventions reproduced (scripts/demo_get_feature.py, utils/test_utils.py::evaluate_image,
cfgs/trainer/default_train.yaml, cfgs/trainer/test/demo_test.yaml): 1 input channel min-max scaled
to [0, 1] (= our cube, so no normalisation), `f_maps=64`, `num_groups=8`, `layer_order='gcl'`,
`num_levels=6` for the released model, `unit_feat=True`. The repo's feature is the last decoder
output (64 channels, full resolution, L2-normalised over channels); `feature_level` selects another
element of `backbone.get_feature` (0 = bottleneck ... -1 = last decoder). The input side must be a
multiple of 2**(num_levels-1) = 32; the default 128 matches our cube.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, select_prefix, strip_prefix


class SingleConv(nn.Sequential):
    """order 'gcl' = GroupNorm(in) -> Conv3d(no bias) -> LeakyReLU (buildingblocks.py::create_conv)."""

    def __init__(self, cin: int, cout: int, num_groups: int):
        super().__init__()
        self.add_module("groupnorm", nn.GroupNorm(1 if cin < num_groups else num_groups, cin))
        self.add_module("conv", nn.Conv3d(cin, cout, 3, padding=1, bias=False))
        self.add_module("LeakyReLU", nn.LeakyReLU(inplace=True))


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int, encoder: bool, num_groups: int):
        super().__init__()
        mid = max(cout // 2, cin) if encoder else cout
        self.add_module("SingleConv1", SingleConv(cin, mid, num_groups))
        self.add_module("SingleConv2", SingleConv(mid, cout, num_groups))


class Encoder(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool, num_groups: int):
        super().__init__()
        self.pooling = nn.MaxPool3d(2) if pool else None
        self.basic_module = DoubleConv(cin, cout, True, num_groups)

    def forward(self, x):
        return self.basic_module(x if self.pooling is None else self.pooling(x))


class Decoder(nn.Module):
    def __init__(self, cin: int, cout: int, num_groups: int):
        super().__init__()
        self.basic_module = DoubleConv(cin, cout, False, num_groups)

    def forward(self, skip, x):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        return self.basic_module(torch.cat((skip, x), dim=1))


class UNet3D(nn.Module):
    """`UNet3D(in_channels, f_maps, 'gcl', num_groups, num_levels, is_unit_vector)`."""

    def __init__(self, in_channels: int = 1, f_maps: int = 64, num_groups: int = 8, num_levels: int = 6,
                 is_unit_vector: bool = True):
        super().__init__()
        fm = [f_maps * 2 ** k for k in range(num_levels)]
        self.encoders = nn.ModuleList(Encoder(in_channels if i == 0 else fm[i - 1], c, i > 0, num_groups)
                                      for i, c in enumerate(fm))
        rev = fm[::-1]
        self.decoders = nn.ModuleList(Decoder(rev[i] + rev[i + 1], rev[i + 1], num_groups)
                                      for i in range(len(rev) - 1))
        self.is_unit_vector = is_unit_vector

    def get_feature(self, x: torch.Tensor) -> list[torch.Tensor]:
        """AbstractUNet.get_feature: [bottleneck, decoder_1, ..., decoder_last]."""
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.insert(0, x)
        feats = [x]
        for dec, skip in zip(self.decoders, skips[1:]):
            x = dec(skip, x)
            feats.append(x)
        if self.is_unit_vector:
            feats[-1] = F.normalize(feats[-1], dim=1)
        return feats


class BrainFMEncoder(FoundationEncoder):
    name = "brainfm"
    input_size = 128

    def __init__(self, f_maps: int = 64, num_groups: int = 8, num_levels: int = 6, unit_feat: bool = True,
                 feature_level: int = -1, pool_to: int | None = None, input_size: int = 128):
        super().__init__()
        if input_size % 2 ** (num_levels - 1):
            raise ValueError(f"input_size must be a multiple of {2 ** (num_levels - 1)}.")
        self.input_size, self.feature_level, self.pool_to = input_size, feature_level, pool_to
        self.backbone = UNet3D(1, f_maps, num_groups, num_levels, unit_feat).requires_grad_(False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x  # the repo min-max scales to [0, 1], which is already our cube

    def load_checkpoint(self, path: str | Path) -> dict:
        ckpt = read_checkpoint(path)
        model_keys = [k for k in ckpt if "model" in k]  # utils/checkpoint.py::find_model_key
        sd = strip_prefix(pick(ckpt, model_keys[:1]), "module.")
        return load_checked(self.backbone, select_prefix(sd, "backbone."), what="BrainFM checkpoint")

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self.backbone.get_feature(x)[self.feature_level]
        if self.pool_to:  # full-resolution 64-ch maps are large; pooling to the patch grid loses nothing downstream
            fmap = F.adaptive_avg_pool3d(fmap, self.pool_to)
        return {"map": fmap}


def build(name: str, model_args: dict) -> FoundationEncoder:
    return BrainFMEncoder(**model_args)
