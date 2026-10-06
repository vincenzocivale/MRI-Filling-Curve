"""MedicalNet (Tencent/MedicalNet) 3D ResNet encoder, pure torch with the original state_dict keys.

Checkpoint format (model.py:112, train.py:89): `{"state_dict": ..., ...}` saved from an
`nn.DataParallel` model, so keys start with `module.`; `conv_seg.*` is the segmentation head and is
ignored. Intensity: z-score over nonzero voxels (datasets/brains18.py), the base default.
Layers 3/4 are dilated (stride 1), so the feature map is input_size / 8.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, strip_prefix

DEPTHS = {10: ("basic", [1, 1, 1, 1]), 18: ("basic", [2, 2, 2, 2]), 34: ("basic", [3, 4, 6, 3]),
          50: ("bottleneck", [3, 4, 6, 3]), 101: ("bottleneck", [3, 4, 23, 3]),
          152: ("bottleneck", [3, 8, 36, 3]), 200: ("bottleneck", [3, 24, 36, 3])}


def conv3(i, o, stride=1, dilation=1):
    return nn.Conv3d(i, o, 3, stride=stride, dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv3(inplanes, planes, stride, dilation), nn.BatchNorm3d(planes)
        self.conv2, self.bn2 = conv3(planes, planes, dilation=dilation), nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv3d(inplanes, planes, 1, bias=False), nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3, self.bn3 = nn.Conv3d(planes, planes * 4, 1, bias=False), nn.BatchNorm3d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class ShortcutA(nn.Module):
    """Parameter-free shortcut (`downsample_basic_block`): strided identity, zero-padded channels."""

    def __init__(self, planes, stride):
        super().__init__()
        self.planes, self.stride = planes, stride

    def forward(self, x):
        out = F.avg_pool3d(x, 1, self.stride)
        return torch.cat([out, out.new_zeros(out.shape[0], self.planes - out.shape[1], *out.shape[2:])], 1)


class ResNet3D(nn.Module):
    """MedicalNet `ResNet` without `conv_seg`; same parameter names."""

    def __init__(self, depth: int, shortcut_type: str):
        super().__init__()
        kind, layers = DEPTHS[depth]
        block = BasicBlock if kind == "basic" else Bottleneck
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, dilation=4)
        self.out_channels = 512 * block.expansion

    def _make_layer(self, block, planes, n, shortcut_type, stride=1, dilation=1):
        down = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down = ShortcutA(planes * block.expansion, stride) if shortcut_type == "A" else nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, dilation, down)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, dilation=dilation) for _ in range(1, n)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class MedicalNet(FoundationEncoder):
    name = "medicalnet"

    def __init__(self, depth: int = 50, shortcut_type: str | None = None, input_size: int = 128):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError(f"MedicalNet depth must be one of {sorted(DEPTHS)}, got {depth}")
        # the repo's released weights: resnet_10/18/34 use shortcut A, resnet_50+ use B
        self.net = ResNet3D(depth, shortcut_type or ("A" if depth <= 34 else "B"))
        self.input_size = input_size

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"map": self.net(x)}

    def load_checkpoint(self, path: str | Path) -> dict:
        sd = strip_prefix(pick(read_checkpoint(path), ("state_dict",)), "module.")
        return load_checked(self.net, sd, ignore_unexpected=("conv_seg.",), what="MedicalNet checkpoint")


def build(name: str, model_args: dict) -> FoundationEncoder:
    return MedicalNet(**model_args)
