"""Dense control decoder (user, 28/10: is Dmix's cascade what loses the fine tokens' advantage?): no hard decision
anywhere. The frozen encoder's cached tokens at 16 and 8 mm (the Dmix token cache: last ++ the depth quartiles, the
curves' mean) are layer-normed, projected per level and spread trilinearly onto the tile at 1/4 of its voxels, where they
join a convolutional image path (UNETR / U-Net style: a full-resolution stem on the raw voxels, two strided stages);
the decoder climbs back to the full resolution with the image skips. Output: logits [B, k, *T] (segrun's generic path:
loss_fn, sliding, landmark_step). args.dense_dec=true in seg_net. ponytail: no 4 mm tokens (live reading of every tile
cost ~10x the step; add a 4 mm cache level if the control says the tokens matter)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .token_cache import TokenCache

LEVELS = (16.0, 8.0)
CH = (16, 32, 64)   # full, 1/2, 1/4 resolution
TOK = 64            # channels of each token level after its projection


def block(a: int, b: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(a, b, 3, stride, 1), nn.InstanceNorm3d(b, affine=True), nn.LeakyReLU(0.01, True),
                         nn.Conv3d(b, b, 3, 1, 1), nn.InstanceNorm3d(b, affine=True), nn.LeakyReLU(0.01, True))


class DenseDec(nn.Module):
    """`encoder` frozen and unused in forward (its tokens come from `cache`); ctx rows: spacing, scan shape, volume
    index, the tile's corner voxel (segrun appends it to the training tiles and, tile_corner, to sliding's)."""

    tile_corner = True

    def __init__(self, encoder, cache: str, n_out: int):
        super().__init__()
        self.encoder = encoder
        self.cache = TokenCache(cache)
        self.cache.levels = LEVELS
        c = encoder.dim * (1 + len(encoder.taps))
        self.tok = nn.ModuleList(nn.Sequential(nn.LayerNorm(c), nn.Linear(c, TOK)) for _ in LEVELS)
        self.e0, self.e1, self.e2 = block(1, CH[0]), block(CH[0], CH[1], 2), block(CH[1], CH[2], 2)
        self.mid = block(CH[2] + TOK * len(LEVELS), CH[2])
        self.u1, self.d1 = nn.ConvTranspose3d(CH[2], CH[1], 2, 2), block(2 * CH[1], CH[1])
        self.u0, self.d0 = nn.ConvTranspose3d(CH[1], CH[0], 2, 2), block(2 * CH[0], CH[0])
        self.out = nn.Conv3d(CH[0], n_out, 1)

    def fields(self, ctx: torch.Tensor, T, dev) -> torch.Tensor:
        """Each level's projected tokens over each tile, trilinear on the tile's 1/4 grid -> [B, TOK x levels, *T/4]."""
        out = []
        for r in ctx.tolist():
            i, corner = self.cache.ids[int(r[6])], [int(v) for v in r[7:10]]
            k4, lv = self.cache.meta(i)["k4"], []
            for j, mm in enumerate(LEVELS):
                k = [a * int(mm // 4) for a in k4]
                c0 = [c // kk for c, kk in zip(corner, k)]
                n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
                t = self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).to(dev).float().reshape(*n, -1)
                g = self.tok[j](t).permute(3, 0, 1, 2)[None]                       # [1, TOK, *n] on the cell grid
                # cell centres -> voxels: the cells' grid spans n*k voxels from c0*k; the tile is a crop of it
                full = [a * kk for a, kk in zip(n, k)]
                g = F.interpolate(g, size=[max(1, f // 4) for f in full], mode="trilinear", align_corners=False)
                o = [(c - a * kk) // 4 for c, a, kk in zip(corner, c0, k)]
                g = g[:, :, o[0]: o[0] + T[0] // 4, o[1]: o[1] + T[1] // 4, o[2]: o[2] + T[2] // 4]
                g = F.pad(g, [p for d, s in zip(reversed(g.shape[2:]), reversed([t // 4 for t in T])) for p in (0, s - d)])
                lv.append(g)
            out.append(torch.cat(lv, 1))
        return torch.cat(out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        T0 = tuple(x.shape[2:])
        x = F.pad(x, [p for t in reversed(T0) for p in (0, -t % 4)])   # 1/4 grid: multiples of 4
        T = tuple(x.shape[2:])
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        tk = self.fields(ctx, T, x.device)   # the projections train; the cached tokens are data
        h = self.mid(torch.cat([f2, tk.to(f2.dtype)], 1))
        h = self.d1(torch.cat([self.u1(h), f1], 1))
        h = self.d0(torch.cat([self.u0(h), f0], 1))
        return self.out(h)[:, :, : T0[0], : T0[1], : T0[2]]

    @staticmethod
    def tile_max(n_out: int) -> float:
        """segrun.TILE_MAX for this decoder: tiles of at most ~160^3 voxels (its activations are dense)."""
        return float(160 ** 3 * n_out)
