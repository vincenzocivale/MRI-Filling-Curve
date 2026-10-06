"""nnU-Net (MIC-DKFZ/nnUNet, Apache-2.0) as a frozen encoder.

Works on any trained nnU-Net v2 model folder, i.e. also on models pretrained with nnU-Net
(e.g. self-supervised / foundation checkpoints exported as a model folder):

    <model_dir>/plans.json  dataset.json  fold_<f>/checkpoint_final.pth

Architecture: exactly what nnU-Net does in `get_network_from_plans` (class + kwargs from
`plans['configurations'][configuration]['architecture']`, built by `dynamic_network_architectures`),
re-implemented here in ~15 lines so that the heavy `nnunetv2` dependency tree (batchgenerators,
acvl_utils, ...) is not needed -- only `pip install dynamic-network-architectures`.

Checkpoint: `torch.save` dict written by `nnUNetTrainer.save_checkpoint`
(nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:1242-1250); weights under `network_weights`,
possibly with `module.` (DataParallel) or `_orig_mod.` (torch.compile) key prefixes, which
`nnUNetTrainer.load_checkpoint` (same file, :1259-1289) strips the same way. The *whole* network
is loaded strictly (decoder included) so a plans/checkpoint mismatch cannot pass silently; only the
encoder is then used.

Features: `network.encoder(x)` returns one map per stage; `stage` (default -1 = bottleneck) picks one.
"""
from __future__ import annotations

import json
import math
import pydoc
from pathlib import Path

import torch

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, require, strip_prefix


def _locate(s):
    obj = pydoc.locate(s)
    if obj is None:
        raise ImportError(f"nnU-Net plans reference {s!r}, which cannot be imported.")
    return obj


def resolve_configuration(plans: dict, name: str) -> dict:
    """`PlansManager.get_configuration` inheritance (plans_handler.py:237): child keys override parent."""
    cfg = dict(plans["configurations"][name])
    if parent := cfg.pop("inherits_from", None):
        return {**resolve_configuration(plans, parent), **cfg}
    return cfg


def network_from_plans(arch: dict, in_channels: int, n_classes: int) -> torch.nn.Module:
    """nnunetv2/utilities/get_network_from_plans.py:9-60 without the nnunetv2 import."""
    require("dynamic_network_architectures", "dynamic-network-architectures", "nnU-Net networks")
    kwargs = dict(arch["arch_kwargs"])
    for k in arch.get("_kw_requires_import", ()):
        v = kwargs.get(k)
        if v is not None:
            kwargs[k] = [_locate(i) for i in v] if isinstance(v, (list, tuple)) else _locate(v)
    net = _locate(arch["network_class_name"])(input_channels=in_channels, num_classes=n_classes, **kwargs)
    if hasattr(net, "initialize"):
        net.apply(net.initialize)
    return net


def num_heads(dataset: dict) -> int:
    """LabelManager.num_segmentation_heads (label_handling.py:259): regions or classes (no ignore label)."""
    labels = dataset["labels"]
    if any(isinstance(v, (list, tuple)) for v in labels.values()):
        return len([k for k, v in labels.items() if k != "background" and v is not None])
    return len([v for k, v in labels.items() if k != "ignore"])


class NNUNetEncoder(FoundationEncoder):
    name = "nnunet"

    def __init__(self, model_dir: str, configuration: str = "3d_fullres", fold: int | str = 0,
                 checkpoint_name: str = "checkpoint_final.pth", stage: int = -1,
                 input_size: int | None = None):
        super().__init__()
        self.model_dir, self.fold, self.checkpoint_name, self.stage = Path(model_dir), fold, checkpoint_name, stage
        plans = json.loads((self.model_dir / "plans.json").read_text())
        dataset = json.loads((self.model_dir / "dataset.json").read_text())
        cfg = resolve_configuration(plans, configuration)
        if "architecture" not in cfg:
            raise ValueError("plans.json has no 'architecture' entry (nnU-Net < 2.4 plans are not supported).")
        channels = dataset.get("channel_names") or dataset["modality"]
        self.in_channels = len(channels)
        self.net = network_from_plans(cfg["architecture"], self.in_channels, num_heads(dataset))
        self.net.requires_grad_(False)
        # cubic input: the plans' patch edge (smallest, if anisotropic), rounded up to the total stride
        total = [1, 1, 1]
        for st in cfg["architecture"]["arch_kwargs"]["strides"]:
            total = [t * k for t, k in zip(total, (st,) * 3 if isinstance(st, int) else st)]
        total = max(total)
        base = int(input_size or min(cfg["patch_size"]))
        self.input_size = math.ceil(base / total) * total

    def checkpoint_path(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else self.model_dir
        return p / f"fold_{self.fold}" / self.checkpoint_name if p.is_dir() else p

    def load_checkpoint(self, path: str | Path | None = None) -> dict:
        ckpt = read_checkpoint(self.checkpoint_path(path))
        sd = pick(ckpt, ["network_weights"])
        sd = strip_prefix(strip_prefix(sd, "module."), "_orig_mod.")  # DDP, then torch.compile
        return load_checked(self.net, sd, what="nnU-Net checkpoint")

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"map": self.net.encoder(x)[self.stage]}


def build(name: str, model_args: dict) -> FoundationEncoder:
    return NNUNetEncoder(**model_args)
