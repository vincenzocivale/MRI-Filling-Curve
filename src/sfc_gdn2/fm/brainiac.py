"""BrainIAC (AIM-KannLab/BrainIAC): MONAI ViT-B/16 SimCLR backbone on 96^3 brain MRI.

Checkpoint format (src/model.py `ViTBackboneNet`): a Lightning ckpt whose `state_dict` holds the
backbone under `backbone.` (plus SimCLR projection-head keys, unused). Input convention
(src/dataset.py): trilinear resize to 96^3, z-score over nonzero voxels (the base default).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, require, select_prefix


class BrainIAC(FoundationEncoder):
    name = "brainiac"

    def __init__(self, img_size: int = 96, patch_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072,
                 num_layers: int = 12, num_heads: int = 12):
        super().__init__()
        vit = require("monai.networks.nets", "monai", "BrainIAC ViT").ViT
        self.input_size, self.patch = img_size, patch_size
        self.backbone = vit(in_channels=1, img_size=(img_size,) * 3, patch_size=(patch_size,) * 3,
                            hidden_size=hidden_size, mlp_dim=mlp_dim, num_layers=num_layers,
                            num_heads=num_heads)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # MONAI's ViT only has a CLS token with classification=True; the repo builds it without, so
        # its `features[0][:, 0]` "CLS" is really patch 0. All g^3 tokens are patches -> spatial map.
        tokens = self.backbone(x)[0]                       # [B, g^3, C], raster order
        g = self.input_size // self.patch
        return {"map": tokens.transpose(1, 2).reshape(len(x), -1, g, g, g)}

    def load_checkpoint(self, path: str | Path) -> dict:
        sd = select_prefix(pick(read_checkpoint(path), ("state_dict",)), "backbone.")
        if not sd:
            raise RuntimeError(f"{path}: no `backbone.*` keys; not a BrainIAC checkpoint.")
        return load_checked(self.backbone, sd, what="BrainIAC checkpoint")


def build(name: str, model_args: dict) -> FoundationEncoder:
    return BrainIAC(**model_args)
