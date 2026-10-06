"""Common interface of the external foundation models (FMs) compared against our encoder.

An adapter wraps one FM's *encoder* so that the probe sees it like our own: it takes the canonical
[0,1] cube the data pipeline already produces and returns a spatial feature map. Everything after
that (pooling to volume / patch level, linear probes, selection, bootstrap CIs) is shared and
lives in `extract.py` and `probe.py`, so every model is evaluated by the same code on the same
splits.

Adapter contract (see `build` in each `fm/<family>.py`):
- architecture comes from the original library / repo and is built from `model_args`;
- `load_checkpoint(path)` accepts the checkpoint exactly as the original repo ships/saves it and
  verifies every key through `load_checked` (a silent partial load would make a "pretrained" probe
  indistinguishable from random init);
- `preprocess` maps our [0,1] cube to the model's input convention (size, intensity, channels).
"""
from __future__ import annotations

import pickle
import warnings
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class MissingDependency(RuntimeError):
    """The external library an adapter needs is not installed (they are optional, never vendored)."""


def require(module: str, pip: str, why: str):
    """Lazy import with an actionable error, e.g. require('monai', 'monai', 'BrainIAC ViT')."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingDependency(f"{why} needs `{module}` (pip install {pip}); it is not vendored here.") from e


def read_checkpoint(path: str | Path) -> dict:
    """torch.load on CPU. Tries the safe `weights_only=True` unpickler first; some repos (nnU-Net)
    pickle numpy / python objects into the file, so it falls back to a full unpickle -- only point
    configs at checkpoints you trust."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError):
        warnings.warn(f"{path}: not loadable with weights_only=True, falling back to full unpickling.")
        return torch.load(path, map_location="cpu", weights_only=False)


def pick(ckpt: dict, keys: Iterable[str]) -> dict:
    """First present key of a checkpoint container (`state_dict`, `model`, `network_weights`, ...);
    the checkpoint itself if none of them is (a bare state_dict)."""
    for k in keys:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def strip_prefix(sd: dict, *prefixes: str) -> dict:
    """Drop the first matching prefix of each key (DataParallel `module.`, Lightning `model.`, ...)."""
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        out[k] = v
    return out


def select_prefix(sd: dict, prefix: str) -> dict:
    """Keys under `prefix`, with it removed (e.g. the encoder inside a full segmentation network)."""
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def load_checked(module: nn.Module, sd: dict, *, ignore_unexpected: Iterable[str] = (),
                 ignore_missing: Iterable[str] = (), what: str = "checkpoint") -> dict:
    """load_state_dict(strict=False) that raises unless every module key was filled and every
    checkpoint key consumed, apart from the explicitly ignored prefixes (decoder, heads, ...).
    Returns {"loaded": n_tensors, "ignored": [...]} for logging."""
    res = module.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not any(k.startswith(p) for p in ignore_missing)]
    ignored = [k for k in res.unexpected_keys if any(k.startswith(p) for p in ignore_unexpected)]
    bad_unexpected = [k for k in res.unexpected_keys if k not in ignored]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"{what} does not match the architecture: "
                           f"{len(bad_missing)} missing (e.g. {bad_missing[:4]}), "
                           f"{len(bad_unexpected)} unexpected (e.g. {bad_unexpected[:4]}).")
    return {"loaded": len(sd) - len(ignored), "ignored": ignored}


class FoundationEncoder(nn.Module):
    """Wraps one FM encoder. Subclasses set `input_size` / `in_channels` and implement `features`,
    `load_checkpoint`, and (if the default [0,1] -> z-score is wrong) `normalize`."""

    name = "fm"
    input_size = 96           # cubic side the network is fed (our cube is resampled to it)
    in_channels = 1           # >1: the single MRI channel is broadcast (see `to_channels`)
    uses_background = False   # True: zero background must stay 0 after normalize (else z-score all)

    def preprocess(self, cube: torch.Tensor) -> torch.Tensor:
        """[B,S,S,S] canonical cube in [0,1] -> [B,in_channels,L,L,L] network input."""
        x = cube[:, None].float()
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size,) * 3, mode="trilinear", align_corners=False)
        return self.to_channels(self.normalize(x))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Default: per-volume z-score over foreground voxels (x > 0), background left at 0."""
        fg = x > 0
        n = fg.flatten(1).sum(1).clamp_min(1).view(-1, 1, 1, 1, 1)
        mean = (x * fg).flatten(1).sum(1).view(-1, 1, 1, 1, 1) / n
        var = (((x - mean) * fg).square().flatten(1).sum(1).view(-1, 1, 1, 1, 1)) / n
        return torch.where(fg, (x - mean) / var.sqrt().clamp_min(1e-6), torch.zeros_like(x))

    def to_channels(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.in_channels == 1 else x.expand(-1, self.in_channels, -1, -1, -1).contiguous()

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Preprocessed [B,C_in,L,L,L] -> {"map": [B,C,d,h,w] (required), "global": [B,C] (optional,
        e.g. CLS token / the model's own pooled embedding)}."""
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, cube: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.features(self.preprocess(cube))
