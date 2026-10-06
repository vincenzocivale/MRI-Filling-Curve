"""Turn a `FoundationEncoder` into a probe `Extractor` (patches [B,N,V] -> pooled features)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.volume import unpatchify
from .base import FoundationEncoder

FG_THRESHOLD = 0.05  # same foreground definition as probe.encoder_extractor


def to_grid(fmap: torch.Tensor, grid: int) -> torch.Tensor:
    """[B,C,d,h,w] -> [B,grid^3,C] in canonical raster order (adaptive average when the map is finer
    than the patch grid, trilinear upsampling when it is coarser)."""
    if fmap.shape[-1] >= grid:
        fmap = F.adaptive_avg_pool3d(fmap, grid)
    else:
        fmap = F.interpolate(fmap, size=(grid,) * 3, mode="trilinear", align_corners=False)
    return fmap.flatten(2).transpose(1, 2)


def fm_extractor(model: FoundationEncoder, level: str):
    """volume level: `mean` (global average of the map), `fg_mean` (average over foreground voxels
    only), plus `global` when the model exposes its own pooled embedding (CLS / projection).
    patch level: `token` = the feature map pooled onto the patch grid, canonical order."""
    def extract(x: torch.Tensor) -> dict[str, torch.Tensor]:
        grid = round(x.shape[1] ** (1 / 3))
        cube = unpatchify(x, grid)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(cube)
        fmap = out["map"].float()
        if level == "patch":
            return {"token": to_grid(fmap, grid)}
        w = F.interpolate((cube[:, None] > FG_THRESHOLD).float(), size=fmap.shape[2:], mode="nearest")
        feats = {"mean": fmap.mean((2, 3, 4)),
                 "fg_mean": (fmap * w).sum((2, 3, 4)) / w.sum((2, 3, 4)).clamp_min(1)}
        if "global" in out:
            feats["global"] = out["global"].float()
        return feats
    return extract
