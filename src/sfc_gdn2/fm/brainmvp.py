"""BrainMVP (shaohao011/BrainMVP): UniFormer-small 3D encoder pretrained on 16k multi-parametric MRI.

The UniFormer is re-implemented here in pure torch with the state_dict keys of the repo
(`models/uniformer_blocks.py` / `Downstream/model/uniformer.py`), so the released checkpoint loads
strictly without the repo, timm or MONAI. Output: stage-4 map `[B, 512, L/16, L/16, L/16]`
(after the final BatchNorm, as `UniFormer.forward` returns it), in canonical (x, y, z) order.

Checkpoint (Downstream/train_script.py:67-80, main.py:141): `{"state_dict": ...}` of the pretraining
`RecModel`: `encoder.uniformer.<k>` (DDP: `module.` in front), plus `decoder.*` and `rep_template`
(the learned modality templates). The repo's own transfer code strips `module.` and `uniformer.`;
we do the same and keep `encoder.*` -> our keys. Decoder / templates are ignored.

Modalities: pretraining feeds ONE modality at a time to a 1-channel encoder (`--in_channels 1`,
main.py:23; each step reconstructs `x[:, index]`), so the released weights take a single channel and a
T1w volume goes in as-is: `in_channels` stays 1. (The 4-channel downstream model discards
`patch_embed1` and re-learns it, Downstream/train_script.py:73-76; there is no pretrained 4-ch stem.)

Intensity: repo uses `ScaleIntensityRangePercentiles(5, 95 -> 0, 1, clip=True)` per channel; done here
over the foreground voxels of each volume, background stays 0. Input 96^3 (roi_x/y/z default).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint


class CMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Conv3d(dim, hidden, 1), nn.GELU(), nn.Conv3d(hidden, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class CBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.norm2 = nn.BatchNorm3d(dim)
        self.mlp = CMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.conv2(self.attn(self.conv1(self.norm1(x))))
        return x + self.mlp(self.norm2(x))


class SABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.pos_embed(x)
        b, c, *spatial = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, *spatial)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        b, c, *spatial = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(b, c, *spatial)


class UniFormer(nn.Module):
    """uniformer_small: depth [3,4,8,3], dims [64,128,320,512], head_dim 64 (CBlock x2 stages, SABlock x2)."""

    def __init__(self, in_chans: int = 1, depth=(3, 4, 8, 3), embed_dim=(64, 128, 320, 512), head_dim: int = 64):
        super().__init__()
        dims = (in_chans, *embed_dim)
        for i in range(4):
            setattr(self, f"patch_embed{i + 1}", PatchEmbed(dims[i], dims[i + 1]))
        for i in range(4):
            heads = embed_dim[i] // head_dim
            blocks = [CBlock(embed_dim[i]) if i < 2 else SABlock(embed_dim[i], heads) for _ in range(depth[i])]
            setattr(self, f"blocks{i + 1}", nn.ModuleList(blocks))
        self.norm = nn.BatchNorm3d(embed_dim[-1])

    def forward(self, x):
        x = x.permute(0, 1, 4, 2, 3)  # repo: "change C*H*W*D to C*D*H*W" (kernels are not axis-symmetric)
        for i in range(1, 5):
            x = getattr(self, f"patch_embed{i}")(x)
            for blk in getattr(self, f"blocks{i}"):
                x = blk(x)
        return self.norm(x).permute(0, 1, 3, 4, 2)  # back to (x, y, z), as the repo's decoder does


class BrainMVP(FoundationEncoder):
    name = "brainmvp"

    def __init__(self, in_channels: int = 1, input_size: int = 96, pct: tuple[float, float] = (5.0, 95.0)):
        super().__init__()
        self.in_channels, self.input_size, self.pct = in_channels, input_size, pct
        self.uniformer = UniFormer(in_chans=in_channels)

    def normalize(self, x):
        """Per-volume 5-95 percentile of the foreground -> [0, 1], clipped; background 0."""
        out = torch.zeros_like(x)
        for i, v in enumerate(x):
            fg = v > 0
            if fg.any():
                lo, hi = torch.quantile(v[fg].flatten().float(), torch.tensor(self.pct, device=v.device) / 100)
                out[i] = torch.where(fg, ((v - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), v)
        return out

    def features(self, x):
        fmap = self.uniformer(x)
        return {"map": fmap, "global": fmap.mean((2, 3, 4))}

    def load_checkpoint(self, path):
        sd = pick(read_checkpoint(path), ("state_dict", "model"))
        out = {}
        for k, v in sd.items():
            k = k.removeprefix("module.")
            if k.startswith("encoder.uniformer."):
                out[k[len("encoder.uniformer."):]] = v
            elif k.startswith("encoder."):
                out[k[len("encoder."):]] = v
            else:
                out[k] = v
        return load_checked(self.uniformer, out, ignore_unexpected=("decoder.", "rep_template", "kl_loss.",
                            "recon_loss."), what="BrainMVP checkpoint")


def build(name: str, model_args: dict) -> BrainMVP:
    return BrainMVP(**model_args)
