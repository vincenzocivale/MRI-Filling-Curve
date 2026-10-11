from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .curves import grid_coords, keys


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


class KernelEmbed(nn.Module):
    """Patch embedding for native voxels of any spacing: a learned continuous kernel K: mm^3 -> R^d,
    integrated over each voxel's physical extent,

        token = b + (1 / patch_mm^3) * sum_voxels v * integral_{voxel} K,

    so every voxel enters with its own weight and nothing is resampled (CKConv / CCNN-style continuous
    kernels: Romero et al. 2022, arXiv 2102.02611; Knigge et al. 2023, arXiv 2301.10540). K is separable
    in a learned basis, K(x, y, z) = G . (phi_x(x) (x) phi_y(y) (x) phi_z(z)), phi_a: mm -> R^rank an MLP on
    Fourier features, so the voxel integrals reduce to 1D integrals per axis (midpoint rule on
    `quad_mm` sub-intervals: exact up to quad_mm, whatever the spacing, 0.15 to 28 mm in the pool).
    `weights` builds one [P, d] matrix per scan (spacing, voxels per patch axis k)."""

    def __init__(self, d: int, patch_mm: float = 16.0, rank: int = 16, freqs: int = 32, hidden: int = 128,
                 max_cycles_per_mm: float = 4.0, quad_mm: float = 0.02):
        super().__init__()
        self.patch_mm, self.rank, self.quad_mm = patch_mm, rank, quad_mm
        f = torch.logspace(math.log10(1 / (4 * patch_mm)), math.log10(max_cycles_per_mm), freqs)
        self.register_buffer("freqs", 2 * math.pi * f, persistent=False)
        self.phi = nn.Sequential(nn.Linear(2 * freqs, hidden), nn.GELU(), nn.Linear(hidden, 3 * rank))
        # init: a constant patch of 1s gives tokens of std ~0.5 at every spacing. The 1000 is a constant in
        # `weights`, not in G: AdamW steps are ~lr in absolute size, so G at std ~16 learned ~250x slower than the
        # other matrices (renamed from G: an old checkpoint fails to load instead of embedding 1000x off)
        self.G_unit = nn.Parameter(torch.randn(d, rank, rank, rank) / rank ** 1.5)
        self.bias = nn.Parameter(torch.zeros(d))

    def axis_integrals(self, spacing: torch.Tensor, k) -> list[torch.Tensor]:
        """Per axis a: [k_a, rank] = integral of phi_a over each voxel (mm), voxel 0 starting at 0 mm."""
        out = []
        for a, (s, n) in enumerate(zip(spacing.tolist(), k)):
            q = max(1, math.ceil(s / self.quad_mm))
            t = (torch.arange(int(n) * q, device=self.freqs.device, dtype=torch.float32) + 0.5) * (s / q)
            ft = t[:, None] * self.freqs
            phi = self.phi(torch.cat([ft.sin(), ft.cos()], -1))[:, a * self.rank:(a + 1) * self.rank]
            out.append(phi.view(int(n), q, self.rank).mean(1) * s)
        return out

    def weights(self, spacing: torch.Tensor, k) -> torch.Tensor:
        """spacing [3] mm, k [3] voxels per patch axis -> W [k0*k1*k2, d] (patchify's voxel order)."""
        px, py, pz = self.axis_integrals(spacing, k)
        w = torch.einsum("dabc,xa,yb,zc->xyzd", self.G_unit, px, py, pz)
        return w.reshape(-1, self.G_unit.shape[0]) * (1000 / self.patch_mm ** 3)

    def forward(self, patches: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """[..., P] native voxel values (fp32: the embedding sees them unrounded) -> [..., d]."""
        with torch.autocast(patches.device.type, enabled=False):
            return patches.float() @ w.float() + self.bias


class Encoder(nn.Module):
    """Causal GDN-2 over curve-ordered patch tokens.

    Tokens are built in raster space (continuous-kernel embedding of each patch's native voxels,
    `KernelEmbed`; masked patches replaced by a learned mask token) and only then gathered into curve order. There is no coordinate embedding: the curve
    order is the only spatial signal, and it differs between views, so an embedding cannot be made
    view-invariant through absolute position. Any grid size: sequences are packed, not padded.
    """

    def __init__(self, d_model: int, depth: int, head_dim: int, num_heads: int, use_short_conv: bool = True,
                 grad_checkpoint: bool = False, patch_mm: float = 16.0, **embed):
        super().__init__()
        self.dim, self.grad_checkpoint = d_model, grad_checkpoint
        self.patch_embed = KernelEmbed(d_model, patch_mm, **embed)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.blocks = nn.ModuleList(GDN2Block(d_model, head_dim, num_heads, use_short_conv, i)
                                    for i in range(depth))
        self.norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: dict) -> Encoder:
        m = cfg["model"]
        return cls(m["d_model"], m["depth"], m["head_dim"], m["num_heads"], m.get("use_short_conv", True),
                   m.get("grad_checkpoint", False), cfg["data"]["patch_mm"], **m.get("embed", {}))

    def tokens(self, patches: torch.Tensor, w: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """[...,N,P] native patches of one scan, its `patch_embed.weights` [P,d] (+ [...,N] mask) -> [...,N,d]."""
        return self.apply_mask(self.patch_embed(patches, w), mask)

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """[...,N,d] tokens, masked ones replaced by the learned mask token."""
        return x if mask is None else torch.where(mask[..., None], self.mask_token.to(x.dtype), x)

    def run(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None, ckpt: bool | None = None) -> torch.Tensor:
        """[B,S,d] curve-ordered tokens -> [B,S,d] hidden states (pre final norm). With `cu_seqlens`
        (int32 [n+1] offsets), x is [1,T,d]: n sequences packed back to back, no padding."""
        for block in self.blocks:
            ckpt = self.grad_checkpoint if ckpt is None else ckpt
            x = (checkpoint(block, x, cu_seqlens, use_reentrant=False) if ckpt and self.training
                 else block(x, cu_seqlens))
        return x

    def packed(self, x: torch.Tensor, valid: torch.Tensor, ckpt: bool | None = None) -> torch.Tensor:
        """[V,L,d] curve-ordered tokens, [V,L] valid (a prefix of each row) -> [V,L,d] normed hidden
        states, zero at padding. Only valid tokens are run, packed into one sequence (views differ ~10x
        in length: padding was ~3x the work). FLA's short conv re-tunes (~20 s) for every new
        ceil(T/1024): a dummy last sequence rounds T up to 8 lengths per octave (<= 12.5% extra)."""
        flat, total = x[valid], int(valid.sum())
        pad = -total % (1 << max(total.bit_length() - 3, 10))
        cu = nn.functional.pad(valid.sum(1).cumsum(0), (1, 0))
        cu = torch.cat([cu, cu[-1:] + pad] if pad else [cu]).int()
        out = self.norm(self.run(nn.functional.pad(flat, (0, 0, 0, pad))[None], cu, ckpt)[0, :total])
        return out.new_zeros(*valid.shape, out.shape[-1]).index_put((valid,), out)


# scans on the device (`dataset.to_device`) -> {name: [B,F] (volume level) or [B,N,F] (patch level, N = the
# batch's largest grid)}
Extractor = Callable[[dict], dict[str, torch.Tensor]]


def encoder_extractor(encoder: Encoder, curves: list[str], level: str, seed: int, fg_threshold: float) -> Extractor:
    """Each volume read along each of `curves` over its whole patch grid (untransformed view), forwards and
    backwards; `bi` = forward ++ backward output per patch, averaged over the curves (test-time augmentation).
    patch: `bi` per patch (raster order). volume: `mean`, bi averaged over the foreground patches (the top of the
    pretraining pyramid). both: the two (the benchmark wrapper, fm/sfc.py)."""
    def extract(vols):
        grid = vols["grid"]
        n = grid.prod(1)
        xyz = grid_coords(grid, int(n.max()))
        j = torch.arange(xyz.shape[1], device=grid.device)
        valid = j < n[:, None]
        rev = torch.where(valid, (n[:, None] - 1 - j).clamp_min(0), j)
        t = torch.zeros(len(grid), xyz.shape[1], encoder.dim, device=grid.device)
        for i, x in enumerate(vols["patches"]):                             # per scan: its own kernel weights
            t[i, : len(x)] = encoder.tokens(x, encoder.patch_embed.weights(vols["spacing"][i], vols["k"][i].tolist()))
        bi = 0
        for curve in curves:
            order = torch.stack([keys(curve, c.clamp_min(0), g, seed).masked_fill(c[:, 0] < 0, 1 << 62).argsort()
                                 for c, g in zip(xyz, grid)])              # per volume: no batch-dependent curve
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = t.gather(1, order[..., None].expand(-1, -1, encoder.dim))
                h = encoder.packed(torch.cat([s, s.gather(1, rev[..., None].expand_as(s))]), valid.repeat(2, 1)).float()
            fwd, bwd = h[: len(grid)], h[len(grid):].gather(1, rev[..., None].expand(-1, -1, h.shape[-1]))
            back = order.argsort(1)[..., None]                                 # curve position -> raster
            bi = bi + torch.cat([fwd, bwd], -1).gather(1, back.expand(-1, -1, 2 * h.shape[-1])) / len(curves)
        if level == "patch":
            return {"bi": bi}
        from .data.dataset import per_patch
        fg = valid & (per_patch(vols, xyz.shape[1], lambda x: x.mean(-1)) > fg_threshold)
        w = (fg | (valid & ~fg.any(1, keepdim=True))).float()
        mean = (w[..., None] * bi).sum(1) / w.sum(1, keepdim=True)
        return {"mean": mean} if level == "volume" else {"bi": bi, "mean": mean}
    return extract
