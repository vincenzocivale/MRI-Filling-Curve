"""BrainSegFounder (lab-smile/BrainSegFounder): MONAI Swin-ViT encoder pretrained with SwinUNETR's
SSL recipe on UK Biobank (stage 1) and, optionally, further on BraTS / ATLAS (stage 2).

Only the Swin-ViT is used (monai.networks.nets.swin_unetr.SwinTransformer, the very class the repo's
`SSLHead` builds); the feature map is its deepest stage `[B, 16*feature_size, L/32, L/32, L/32]`.

Checkpoint (downstream/BraTS/ssl/main_T1T2.py:486-490, ATLAS/finetune.py:124-135,
BraTS/finetuning/*:180-194): a dict `{"state_dict": ..., "epoch", "optimizer"}`.
- stage-1 / stage-2 SSL: `state_dict` of `SSLHead` (`swinViT.*`, `rotation_head.*`, `contrastive_head.*`,
  `conv.*`), usually behind DDP's `module.`;
- the original MONAI SwinUNETR SSL weights use `module.<swin key>` (the repo's "module." -> "swinViT."
  rename only makes sense for those);
- fine-tuned SwinUNETR: `swinViT.*` + `encoder1..10`, `decoder1..5`, `out.*`.
All are accepted: wrapper prefixes are stripped, heads / decoder are ignored, and every Swin key must match.

Intensity: the repo scales to [0, 1] (`ScaleIntensityRanged`, b_min=0, b_max=1), which is what our
cube already is, so `normalize` is the identity. Input 96^3 as in the repo (roi_x/y/z=96).
"""
from __future__ import annotations

import torch

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, require

WRAPPERS = ("module.", "backbone.", "swinViT.", "model.")
# SSLHead heads and SwinUNETR decoder-side modules; none of them is part of the Swin-ViT.
NOT_ENCODER = ("rotation_head.", "rotation_pre.", "contrastive_head.", "contrastive_pre.", "conv.",
               "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder10.", "decoder", "out.")


def strip_wrappers(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        while (p := next((p for p in WRAPPERS if k.startswith(p)), None)) is not None:
            k = k[len(p):]
        out[k] = v
    return out


class BrainSegFounder(FoundationEncoder):
    name = "brainsegfounder"

    def __init__(self, feature_size: int = 48, in_channels: int = 1, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), input_size: int = 96, window_size: int = 7):
        super().__init__()
        swin = require("monai.networks.nets.swin_unetr", "monai", "BrainSegFounder (Swin-ViT)")
        self.in_channels, self.input_size = in_channels, input_size
        self.swinViT = swin.SwinTransformer(
            in_chans=in_channels, embed_dim=feature_size, window_size=(window_size,) * 3,
            patch_size=(2, 2, 2), depths=tuple(depths), num_heads=tuple(num_heads), mlp_ratio=4.0,
            qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm, use_checkpoint=False, spatial_dims=3)

    def normalize(self, x):
        return x  # repo input is already scaled to [0, 1]

    def features(self, x):
        fmap = self.swinViT(x.contiguous())[4]  # deepest stage, as SSLHead.forward does
        return {"map": fmap, "global": fmap.mean((2, 3, 4))}

    def load_checkpoint(self, path):
        sd = strip_wrappers(pick(read_checkpoint(path), ("state_dict", "model", "network_weights")))
        # relative_position_index is a deterministic buffer some MONAI versions do not serialise.
        for k, v in self.swinViT.state_dict().items():
            if k.endswith("relative_position_index"):
                sd.setdefault(k, v)
        return load_checked(self.swinViT, sd, ignore_unexpected=NOT_ENCODER, what="BrainSegFounder checkpoint")


def build(name: str, model_args: dict) -> BrainSegFounder:
    return BrainSegFounder(**model_args)
