"""Adaptive-resolution segmentation on the frozen encoder: read the tile at 8 mm everywhere, let a small ranker pick the 8 mm
cells to read at 4 mm, decode the mixed 8 / 4 mm leaves (scripts/adaptive_train.py trains it, scripts/train_ranker.py studied
the ranker alone).

1. pass 1: the encoder reads the whole tile at 8 mm (cells of k0 / 2 native voxels, k0 = apt.base_dims(spacing, 16 mm, 3));
2. the ranker (3x3x3 depthwise ConvNeXt blocks over the token grid + 4 voxel statistics) scores every 8 mm cell;
3. the top `split_frac` of the cells are split into their 8 children (4 mm); the others stay 8 mm: the leaves;
4. pass 2: the encoder reads the leaves as one sequence ordered along the curves (apt.APT: a leaf of any size gets its token
   from its own native voxels, nothing resampled; weights untouched);
5. decoder: leaf token + a learned level vector -> PatchDecode (`cells`^3 cells) -> trilinear to the leaf's voxels -> logits.
Privileged information: the labels give each 8 mm cell a gain (class-weighted voxels that turn right when the cell is read
as its 8 children, as in cache_tokens.py). The ranker is trained on it with ListNet (aux loss); while training, with
probability `q` (decaying, BATS' oracle warm-up) the cells are chosen by the true gain instead of the ranker, so the
decoder first sees good hierarchies. At test time the labels are not used.
"""
from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .apt import APT, base_dims, entropy
from .data.volume import patch_grid, patchify
from .model import encoder_extractor

LEVELS = 3   # 16 / 8 / 4 mm cells of the octree; leaves live at 8 and 4 mm


def hist(y: torch.Tensor, k, B: int) -> torch.Tensor:
    """y [D,H,W] uint8 (255 = no label), cells of k voxels -> [N, B] int32 counts of the labels 0 .. B - 1."""
    p = patchify(y, k).long()
    p = torch.where(p == 255, torch.full_like(p, B), p).clamp_max(B)
    N = p.shape[0]
    idx = (p + torch.arange(N, device=y.device)[:, None] * (B + 1)).flatten()
    h = torch.zeros(N * (B + 1), dtype=torch.int32, device=y.device)
    h.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
    return h.view(N, B + 1)[:, :B]


def block_sum(t: torch.Tensor, g1) -> torch.Tensor:
    """[g2x*g2y*g2z, ...] (4 mm cells, raster) -> [N1, ...] sums over each 8 mm cell's 2x2x2 children."""
    return t.view(g1[0], 2, g1[1], 2, g1[2], 2, *t.shape[1:]).sum((1, 3, 5)).flatten(0, 2)


def cell_gain(y: torch.Tensor, k0, g1, B: int) -> torch.Tensor:
    """[N1] class-weighted voxels that stop being outside the majority class when an 8 mm cell is read as its children."""
    H2 = hist(y, tuple(q // 4 for q in k0), B).float()
    H1 = block_sum(H2, g1)
    n = H1.sum(0)
    wc = torch.where(n > 0, n.clamp_min(1).rsqrt(), torch.zeros_like(n))

    def err(H):
        Hw = H * wc
        return Hw.sum(1) - Hw.max(1).values
    return err(H1) - block_sum(err(H2)[:, None], g1)[:, 0]


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.dw, self.norm = nn.Conv3d(w, w, 3, padding=1, groups=w), nn.LayerNorm(w)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))

    def forward(self, h):   # [gx, gy, gz, w]
        t = self.dw(h.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0)
        return h + self.mlp(self.norm(t))


class Ranker(nn.Module):
    """Token [dim] + 4 voxel statistics per 8 mm cell -> score; 3x3x3 context from the depthwise blocks."""

    def __init__(self, dim=512, width=128, blocks=3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim + 4), nn.Linear(dim + 4, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, tok, hand, g1):
        h = self.inp(torch.cat([tok, hand], -1)).view(*g1, -1)
        for b in self.blocks:
            h = b(h)
        return self.out(h.flatten(0, 2))[:, 0]


class AdaptiveSegNet(nn.Module):
    """forward(x [B,1,*T], spacing [B,3] or [B,6] = spacing + the scan's shape, y=None) -> logits [B,n_out,*T];
    `aux` holds the ranker's loss after a forward with labels (training). split_frac / q are plain attributes."""

    def __init__(self, encoder, curves, seed, n_out, split_frac=0.3, cells=4):
        super().__init__()
        from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchDecode
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He

        self.encoder, self.n_out, self.dim, self.cells = encoder, n_out, encoder.dim, cells
        self.split_frac, self.q = split_frac, 0.0
        self.apt = APT(encoder, curves, seed, 0.05, 16.0, LEVELS, {})
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0)
        self.ranker = Ranker(encoder.dim)
        self.level_emb = nn.Parameter(torch.zeros(2, encoder.dim))
        self.decode = PatchDecode((cells,) * 3, encoder.dim, n_out, norm=LayerNormNd, activation=nn.GELU)
        self.decode.apply(InitWeights_He(1e-2))
        self.aux = torch.zeros(())
        self.stats = {"leaves": 0, "uni4": 0}   # tokens read vs. a uniform 4 mm grid, summed over the tiles seen

    def forward(self, x, spacing, y=None):
        T = tuple(x.shape[2:])
        ext = spacing[:, 3:].long().tolist() if spacing.shape[1] > 3 else [T] * len(x)
        sp = spacing[:, :3]
        outs, aux = [], x.new_zeros(())
        for b in range(len(x)):
            V = tuple(min(a, t) for a, t in zip(ext[b], T))
            o, a = self.one(x[b, 0, : V[0], : V[1], : V[2]].float(), sp[b],
                            None if y is None else y[b, : V[0], : V[1], : V[2]])
            outs.append(F.pad(o, [p for v, t in zip(reversed(o.shape[1:]), reversed(T)) for p in (0, t - v)]))
            aux = aux + a
        self.aux = aux / len(x)
        return torch.stack(outs)

    def one(self, xv, sp, yv):
        dev = xv.device
        k0 = base_dims(sp.tolist(), 16.0, LEVELS)
        pad = [p for n, q in zip(reversed(xv.shape), reversed(k0)) for p in (0, -n % q)]
        xp, ones = F.pad(xv, pad), F.pad(torch.ones_like(xv), pad)
        k1, k2 = tuple(q // 2 for q in k0), tuple(q // 4 for q in k0)
        g1 = [s // q for s, q in zip(xp.shape, k1)]
        g2 = [2 * g for g in g1]
        spc, pe = sp.float().cpu(), self.encoder.patch_embed
        mm0 = pe.patch_mm
        with torch.no_grad():   # pass 1: the whole tile at 8 mm
            pe.patch_mm = 8.0
            grid = patch_grid(tuple(xp.shape), k1).to(dev)
            bi = self.extract({"patches": [patchify(xp, k1)], "grid": grid[None], "k": torch.tensor([k1], device=dev),
                               "spacing": sp[None].float()})["bi"][0]
            pe.patch_mm = mm0
            tok8 = bi[: int(grid.prod()), : self.dim].float()
            px = patchify(xp, k1)
            mean = px.mean(1)
            hand = torch.stack([mean, (px - mean[:, None]).pow(2).mean(1).sqrt(), px.amax(1) - px.amin(1), entropy(px, 16)], -1)
            valid1 = patchify(ones, k1).sum(1) > 0
            valid2 = (patchify(ones, k2).sum(1) > 0).view(*g2)
        s = self.ranker(tok8, hand, g1)
        aux = s.new_zeros(())
        gain = None
        if yv is not None:
            gain = cell_gain(F.pad(yv, pad, value=255), k0, g1, self.n_out)
            g, sv = gain[valid1], s[valid1].float()
            if g.sum() > 0:   # ListNet against the gains normalised in the tile
                aux = -(g / g.sum() * F.log_softmax(sv, 0)).sum()
        nv = int(valid1.sum())
        oracle = self.training and gain is not None and random.random() < self.q
        score = (gain if oracle else s.detach()).float().masked_fill(~valid1, -float("inf"))
        sel = torch.zeros_like(valid1)
        sel[score.argsort(descending=True)[: round(self.split_frac * nv)]] = True
        m1 = sel.view(*g1)
        m2 = m1
        for a in range(3):
            m2 = m2.repeat_interleave(2, a)
        lv1, lv2 = (valid1.view(*g1) & ~m1).nonzero(), (valid2 & m2).nonzero()
        leaves = [lv1.new_zeros(0, 3), lv1, lv2]
        self.stats["leaves"] += len(lv1) + len(lv2)
        self.stats["uni4"] += int(valid2.sum())
        with torch.no_grad():   # pass 2: the leaves as one sequence
            tok, _, corner, level = self.apt.embed(xp, spc, k0, leaves)
            bi2 = self.apt.encode(tok, corner, torch.tensor(g2, device=dev))
        h = bi2[:, : self.dim].float() + self.level_emb[level - 1]
        logits = xp.new_zeros(self.n_out, *xp.shape)
        off = 0
        for lev, kk, gg in ((1, k1, g1), (2, k2, g2)):
            c = leaves[lev]
            if not len(c):
                continue
            L6 = logits.view(self.n_out, gg[0], kk[0], gg[1], kk[1], gg[2], kk[2])
            step = max(1, (1 << 24) // (self.n_out * max(kk[0] * kk[1] * kk[2], self.cells ** 3)))
            for i in range(0, len(c), step):
                def run(z, kk=kk):
                    return F.interpolate(self.decode(z[:, :, None, None, None]), size=kk, mode="trilinear",
                                         align_corners=False)
                j = min(i + step, len(c))   # the slice must stop at the end of this level's leaves
                p = checkpoint(run, h[off + i: off + j], use_reentrant=False)
                ci = c[i: j]
                L6[:, ci[:, 0], :, ci[:, 1], :, ci[:, 2], :] = p.float()
            off += len(c)
        return logits[:, : xv.shape[0], : xv.shape[1], : xv.shape[2]], aux
