"""MoME / MoME+ (ZhangxinruBIT/MoME): mixture of modality experts for brain-lesion segmentation,
built as an nnU-Net v2 fork. Both are re-implemented here for feature extraction only, with pure-torch
nnU-Net encoder blocks whose state_dict keys equal the fork's (`encoder.stages.S.0.convs.K.{conv,norm,
all_modules.*}`), so the released nnU-Net checkpoints load strictly and nothing from the repo (nor
nnunetv2 / dynamic_network_architectures, which the fork patches and would clash with a stock install)
needs to be installed.

Pipeline reproduced (MoME_foundation/.../nnUNetTrainer.py:1074-1081 and the inference twin; MoME_plus
.../predict_from_raw_data.py:600-650), keeping only what feeds the encoder representation:
  1. every expert (a 1-channel nnU-Net) runs its first encoder stage on the image -> 32 ch at full res
     (`Feature_i[0]`, the first skip);
  2. the aggregator ("gating") nnU-Net takes cat(image, expert features) and its encoder is run;
  3. we return its deepest stage `[B, C, L/32, ...]` as the feature map.
The decoders / the expert-weighted segmentation head are not used.
- MoME: 5 experts (T1, T1ce, T2, FLAIR, DWI), aggregator input 1 + 5*32 channels; the single image is
  fed to every expert (that is how the trainer calls them).
- MoME+: 4 experts (T1, T1ce, T2, FLAIR), `ClsDispatchNet` fills the missing modalities from the
  available ones (`--MultiMod`); aggregator input 4 + 4*32. Our single T1w is placed in `modality`
  (default slot 0 = T1, MultiMod = [1,0,0,0]); the other three channels are synthesised by the dispatch net.

Checkpoints (nnUNetTrainer.save_checkpoint, MoME_foundation :1282-1380; MoME_plus :1410-1506): one
nnU-Net dict `{"network_weights": ..., "init_args", "trainer_name", ...}` per network, in the fold dir:
  checkpoint_best.pth            aggregator          (`checkpoint:` points at this file)
  checkpoint_best{1..5|1..4}.pth experts             (repo rule: filename.replace('best', 'best<i>'))
  checkpoint_best_dispatch.pth   dispatch net, MoME+ only
(`latest` / `final` are handled the same way). Experts can instead be given explicitly with
`model_args.expert_checkpoints` (e.g. the HF "Pretrained_Experts", plain nnU-Net checkpoints).

Architecture is not stored in the checkpoint but in nnU-Net's plans: give `plans_json` (+ `configuration`)
or `arch` (explicit kwargs); defaults are the nnU-Net v2 3d_fullres brain defaults (6 stages).
Intensity: nnU-Net's per-volume z-score inside the foreground (`FoundationEncoder.normalize`), the MoME
preprocessing being skull-stripped + cropped data. Input 128^3 (any multiple of 2^(n_stages-1)).
"""
from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import torch
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, strip_prefix

DEFAULT_ARCH = {"features_per_stage": [32, 64, 128, 256, 320, 320], "kernel_sizes": [3] * 6,
                "strides": [1, 2, 2, 2, 2, 2], "n_conv_per_stage": [2] * 6}
EXPERTS = {"mome": ("t1", "t1ce", "t2", "flair", "dwi"), "mome_plus": ("t1", "t1ce", "t2", "flair")}


# ------------------------------------------------------------------ nnU-Net blocks (key-compatible)
class ConvDropoutNormReLU(nn.Module):
    """conv -> InstanceNorm -> LeakyReLU. Like the original, `conv`/`norm`/`nonlin` are also registered
    individually, so the state_dict has both `conv.weight` and `all_modules.0.weight`."""

    def __init__(self, cin: int, cout: int, kernel, stride):
        super().__init__()
        k = [kernel] * 3 if isinstance(kernel, int) else list(kernel)
        self.conv = nn.Conv3d(cin, cout, k, stride, padding=[(i - 1) // 2 for i in k], bias=True)
        self.norm = nn.InstanceNorm3d(cout, eps=1e-5, affine=True)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.all_modules = nn.Sequential(self.conv, self.norm, self.nonlin)

    def forward(self, x):
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(self, n: int, cin: int, cout: int, kernel, stride):
        super().__init__()
        self.convs = nn.Sequential(ConvDropoutNormReLU(cin, cout, kernel, stride),
                                   *[ConvDropoutNormReLU(cout, cout, kernel, 1) for _ in range(1, n)])

    def forward(self, x):
        return self.convs(x)


class PlainConvEncoder(nn.Module):
    def __init__(self, cin: int, features_per_stage, kernel_sizes, strides, n_conv_per_stage):
        super().__init__()
        stages = []
        for f, k, s, n in zip(features_per_stage, kernel_sizes, strides, n_conv_per_stage):
            stages.append(nn.Sequential(StackedConvBlocks(n, cin, f, k, s)))
            cin = f
        self.stages = nn.Sequential(*stages)

    def forward(self, x, first_only: bool = False):
        if first_only:
            return self.stages[0](x)
        for stage in self.stages:
            x = stage(x)
        return x


class UNetEncoder(nn.Module):
    """Holds `encoder.*` only, i.e. the key layout of a PlainConvUNet checkpoint minus its decoder."""

    def __init__(self, cin: int, **arch):
        super().__init__()
        self.encoder = PlainConvEncoder(cin, arch["features_per_stage"], arch["kernel_sizes"], arch["strides"],
                                        arch["n_conv_per_stage"])
        for m in self.modules():  # InitWeights_He(1e-2)
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                nn.init.zeros_(m.bias)

    def forward(self, x, first_only: bool = False):
        return self.encoder(x, first_only)


class ClsDispatchNet(nn.Module):
    """MoME+ dispatch network (MoME_plus/.../Dispatch_network.py): availability mask -> [B, M, M]."""

    def __init__(self, input_dim: int = 4, hidden_dims=(4, 8, 16)):
        super().__init__()
        dims, layers = [input_dim, *hidden_dims], []
        for a, b in pairwise(dims):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-1], input_dim * input_dim))
        self.mlp = nn.Sequential(*layers)
        self.num_class = input_dim

    def forward(self, x):
        return self.mlp(x).view(-1, self.num_class, self.num_class)


def arch_from_plans(path: str | Path, configuration: str = "3d_fullres") -> dict:
    """nnU-Net plans.json -> encoder kwargs. Handles the fork's plans (conv_kernel_sizes, pool_op_kernel_sizes,
    UNet_base_num_features, ...) and the newer `architecture.arch_kwargs` layout."""
    plans = json.loads(Path(path).read_text())
    confs = plans["configurations"]
    cfg = dict(confs[configuration])
    while "inherits_from" in cfg:
        cfg = {**confs[cfg.pop("inherits_from")], **cfg}
    if "architecture" in cfg:
        a = cfg["architecture"]["arch_kwargs"]
        n = a["n_stages"]
        return {"features_per_stage": a["features_per_stage"], "kernel_sizes": a["kernel_sizes"],
                "strides": a["strides"], "n_conv_per_stage": a.get("n_conv_per_stage", a.get("n_blocks_per_stage"))
                or [2] * n}
    n = len(cfg["conv_kernel_sizes"])
    return {"features_per_stage": [min(cfg["UNet_base_num_features"] * 2 ** i, cfg["unet_max_num_features"])
                                   for i in range(n)],
            "kernel_sizes": [k[0] if len(set(k)) == 1 else k for k in cfg["conv_kernel_sizes"]],
            "strides": [s[0] if len(set(s)) == 1 else s for s in cfg["pool_op_kernel_sizes"]],
            "n_conv_per_stage": cfg["n_conv_per_stage_encoder"]}


def sibling(path: Path, suffix: str) -> Path:
    """The repo derives expert / dispatch files with filename.replace('best', 'best<suffix>') (same for
    latest / final)."""
    for tag in ("latest", "best", "final"):
        if tag in path.name:
            return path.with_name(path.name.replace(tag, tag + suffix))
    raise ValueError(f"Cannot derive sibling checkpoints from {path.name!r}: expected latest/best/final in the name.")


def network_weights(path: Path) -> dict:
    return strip_prefix(pick(read_checkpoint(path), ("network_weights", "state_dict")), "module.", "_orig_mod.")


class MoME(FoundationEncoder):
    name = "mome"

    def __init__(self, plus: bool = False, arch: dict | None = None, plans_json: str | None = None,
                 configuration: str = "3d_fullres", input_size: int = 128, modality: str = "t1",
                 expert_checkpoints: list[str] | None = None):
        super().__init__()
        self.plus, self.input_size, self.in_channels = plus, input_size, 1
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        if plans_json:
            self.arch = arch_from_plans(plans_json, configuration)
        names = EXPERTS["mome_plus" if plus else "mome"]
        if plus and modality not in names:
            raise ValueError(f"modality must be one of {names}, got {modality!r}")
        self.slot = names.index(modality) if plus else None
        self.expert_checkpoints = expert_checkpoints
        n, f0 = len(names), self.arch["features_per_stage"][0]
        self.experts = nn.ModuleList(UNetEncoder(1, **self.arch) for _ in range(n))
        self.aggregator = UNetEncoder((n if plus else 1) + n * f0, **self.arch)
        self.dispatch = ClsDispatchNet(n) if plus else None

    def features(self, x):
        if self.plus:
            mask = x.new_zeros(1, len(self.experts))
            mask[0, self.slot] = 1  # MultiMod = one-hot at our modality
            raw = self.dispatch(mask)
            raw = (raw - raw.min(2, keepdim=True).values + 1e-4) / (
                raw.max(2, keepdim=True).values - raw.min(2, keepdim=True).values + 1e-4)
            sparse = raw * mask[..., None].repeat(1, 1, len(self.experts)).permute(0, 2, 1)  # M(b,4,4)
            keep = mask == 1
            sparse[keep, :] = torch.eye(len(self.experts), device=x.device)[None].expand_as(sparse)[keep, :]
            data = x.new_zeros(x.shape[0], len(self.experts), *x.shape[2:])
            data[:, self.slot] = x[:, 0]
            x = torch.einsum("bij,bj...->bi...", sparse.expand(x.shape[0], -1, -1), data)
            feats = [e(x[:, i:i + 1], first_only=True) for i, e in enumerate(self.experts)]
        else:
            feats = [e(x, first_only=True) for e in self.experts]
        fmap = self.aggregator(torch.cat([x, *feats], dim=1))
        return {"map": fmap, "global": fmap.mean((2, 3, 4))}

    def load_checkpoint(self, path):
        path, report = Path(path), {}
        agg = network_weights(path)
        report["aggregator"] = load_checked(self.aggregator, agg, ignore_unexpected=("decoder.",),
                                            what=f"MoME aggregator {path.name}")
        paths = self.expert_checkpoints or [sibling(path, str(i + 1)) for i in range(len(self.experts))]
        for i, (expert, p) in enumerate(zip(self.experts, paths)):
            report[f"expert{i + 1}"] = load_checked(expert, network_weights(Path(p)), ignore_unexpected=("decoder.",),
                                                    what=f"MoME expert {i + 1} {Path(p).name}")
        if self.plus:
            report["dispatch"] = load_checked(self.dispatch, network_weights(sibling(path, "_dispatch")),
                                              what="MoME+ dispatch network")
        return report


def build(name: str, model_args: dict) -> MoME:
    model = MoME(plus=name == "mome_plus", **model_args)
    model.name = name
    return model
