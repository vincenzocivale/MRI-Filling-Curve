from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _gdn2():
    try:
        from lit_gpt.gdn2 import GatedDeltaNet2
    except Exception as e:
        raise RuntimeError("Official NVlabs/GatedDeltaNet-2 (and its FLA/Triton deps) is required; "
                           "see README 'Install'.") from e
    return GatedDeltaNet2


class GDN2Block(nn.Module):
    """Pre-norm causal GDN-2 token mixer + MLP."""

    def __init__(self, d: int, head_dim: int, num_heads: int, use_short_conv: bool, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mix = _gdn2()(hidden_size=d, head_dim=head_dim, num_heads=num_heads, mode="chunk",
                           use_short_conv=use_short_conv, layer_idx=layer_idx)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.mix(self.norm(x), cu_seqlens=cu_seqlens)[0]
        return x + self.ffn(self.ffn_norm(x))


class Encoder(nn.Module):
    """Causal GDN-2 over curve-ordered patch tokens.

    Tokens are built in canonical raster space (patch embedding, masked patches replaced by a
    learned mask token) and only then gathered into curve order. There is no coordinate
    embedding: the curve order is the only spatial signal, and it differs between views, so
    an embedding cannot be made view-invariant through absolute position.
    """

    def __init__(self, patch_voxels: int, grid: int, d_model: int, depth: int, head_dim: int,
                 num_heads: int, use_short_conv: bool = True, grad_checkpoint: bool = False):
        super().__init__()
        self.grid, self.dim, self.grad_checkpoint = grid, d_model, grad_checkpoint
        self.patch_embed = nn.Linear(patch_voxels, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.blocks = nn.ModuleList(GDN2Block(d_model, head_dim, num_heads, use_short_conv, i)
                                    for i in range(depth))
        self.norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: dict) -> Encoder:
        (side, *rest), patch = cfg["data"]["target_shape"], cfg["data"]["patch_size"]
        if any(s != side for s in rest) or side % patch:
            raise ValueError("target_shape must be cubic and divisible by patch_size.")
        m = cfg["model"]
        return cls(patch ** 3, side // patch, m["d_model"], m["depth"], m["head_dim"], m["num_heads"],
                   m.get("use_short_conv", True), m.get("grad_checkpoint", False))

    def tokens(self, patches: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """[B,N,V] canonical patches (+ [B,N] or [N] mask) -> [B,N,d] canonical tokens."""
        x = self.patch_embed(patches)
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        return x

    def run(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """[B,S,d] curve-ordered tokens -> [B,S,d] hidden states (pre final norm). With `cu_seqlens`
        (int32 [n+1] offsets), x is [1,T,d]: n sequences packed back to back, no padding."""
        for block in self.blocks:
            x = (checkpoint(block, x, cu_seqlens, use_reentrant=False) if self.grad_checkpoint and self.training
                 else block(x, cu_seqlens))
        return x

    def forward(self, patches: torch.Tensor, perm: torch.Tensor, mask: torch.Tensor | None = None):
        """Encode in the order `perm` ([N] sequence position -> canonical index). Returns [B,N,d]."""
        return self.norm(self.run(self.tokens(patches, mask)[:, perm]))
