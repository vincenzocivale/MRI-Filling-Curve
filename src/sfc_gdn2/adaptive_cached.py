"""Our adaptive-resolution segmentation (adaptive.py: ranker + decoder on 8 / 4 mm leaves, the labels as privileged
information) on the TOKEN CACHE of a frozen encoder (fm/token_cache.py: the encoder's tokens of every scan at 16 / 8 / 4 mm
on nested grids, here the 512-channel PCA versions of the 8 and 4 mm levels). For checkpoints the online reading of
adaptive.py cannot serve (p13: attention blocks, 260M parameters) and to make an iteration cost the decoder only.

A tile is a box of whole 16 mm cells of a scan. Its 8 mm tokens come from the cache, the ranker scores every 8 mm cell
(3x3x3 depthwise blocks over the token grid + 4 voxel statistics), the top `split_frac` cells are read at 4 mm (their 8
children's cached tokens), the others stay 8 mm; each leaf token goes through a shared PatchDecode (+ a learned level
vector) to `cells`^3 cells, resized to the leaf's native voxels. Training: ListNet loss of the ranker against the gain of
splitting each cell (class-weighted voxels that turn right, from the labels), and with probability `q` (BATS' oracle
warm-up) the cells are chosen by the true gain instead of the ranker.

One difference from adaptive.py, by construction: the 4 mm tokens of the cache come from a uniform 4 mm reading of the
scan, not from one sequence mixing 8 and 4 mm leaves.
"""
from __future__ import annotations

import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adaptive import Ranker, block_sum, hist
from .apt import entropy
from .data.volume import patchify
from .fm.token_cache import TokenCache, fine_ok


def cell_gain(y: torch.Tensor, k4, g8, B: int) -> torch.Tensor:
    """[N8] class-weighted voxels that stop being outside the majority class when an 8 mm cell is read as its 8 children
    (4 mm cells of k4 voxels; y [D,H,W] uint8 padded with 255 to g8 * 2 * k4)."""
    H2 = hist(y, k4, B).float()
    H1 = block_sum(H2, g8)
    n = H1.sum(0)
    wc = torch.where(n > 0, n.clamp_min(1).rsqrt(), torch.zeros_like(n))

    def err(H):
        Hw = H * wc
        return Hw.sum(1) - Hw.max(1).values
    return err(H1) - block_sum(err(H2)[:, None], g8)[:, 0]


class CachedAdaptiveNet(nn.Module):
    def __init__(self, cache_root, n_out, split_frac=0.3, cells=4, dim=512, pca=512):
        super().__init__()
        from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchDecode
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He

        self.cache, self.n_out, self.dim, self.cells = TokenCache(cache_root), n_out, dim, cells
        self.cache.pca = pca
        self.split_frac, self.q = split_frac, 0.0
        self.ranker = Ranker(dim)
        self.norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(2))   # per level: the PCA tokens' scale differs
        self.level_emb = nn.Parameter(torch.zeros(2, dim))
        self.decode = PatchDecode((cells,) * 3, dim, n_out, norm=LayerNormNd, activation=nn.GELU)
        self.decode.apply(InitWeights_He(1e-2))
        self.aux = torch.zeros(())
        self.stats = {"leaves": 0, "uni4": 0}
        self.diag = {}

    @staticmethod
    def cell_dims(k4):
        return tuple(2 * a for a in k4)

    def prefetch(self, i, corner, V):
        """Start reading the tokens a tile will ask for (the cache's thread pool), while the previous tile trains."""
        k4 = self.cache.meta(i)["k4"]
        k8 = self.cell_dims(k4)
        c = self.cache
        for mm, k in ((8.0, k8), (4.0, tuple(k4))):
            if mm < 8 and not fine_ok(k4):
                continue
            n = tuple(-(-v // a) for v, a in zip(V, k)) if mm == 8 else tuple(2 * -(-v // a) for v, a in zip(V, k8))
            key = (i, mm, tuple(int(a) for a in corner), n)
            if key not in c._ahead:
                c._ahead[key] = (c._gen, c._io.submit(c._read, i, mm, [int(a) for a in corner], n, k))

    def forward(self, i, corner, xv, yv=None):
        """Tile of scan `i` at voxel `corner` (a multiple of the 16 mm cell), voxels xv [V] float (+ labels yv [V] uint8)
        -> logits [n_out, *V]; `aux` the ranker's loss."""
        dev, V = xv.device, tuple(xv.shape)
        meta = self.cache.meta(i)
        k4 = tuple(meta["k4"])
        k8 = self.cell_dims(k4)
        g8 = [-(-v // a) for v, a in zip(V, k8)]
        pad = [p for n, q in zip(reversed(V), reversed(k8)) for p in (0, -n % q)]
        xp = F.pad(xv, pad)
        tok8 = self.cache.tokens(i, 8.0, corner, g8, k8).to(dev).float()
        px = patchify(xp, k8)
        mean = px.mean(1)
        hand = torch.stack([mean, (px - mean[:, None]).pow(2).mean(1).sqrt(), px.amax(1) - px.amin(1), entropy(px, 16)], -1)
        s = self.ranker(tok8, hand, g8)   # the ranker normalises its own input (not shared with the decoder's)
        valid = patchify(F.pad(torch.ones_like(xv), pad), k8).sum(1) > 0
        nv, aux, gain = int(valid.sum()), s.new_zeros(()), None
        if yv is not None:
            gain = cell_gain(F.pad(yv, pad, value=255), k4, g8, self.n_out)
            g, sv = gain[valid], s[valid].float()
            if g.sum() > 0:
                aux = -(g / g.sum() * F.log_softmax(sv, 0)).sum()
                with torch.no_grad():   # progress that does not depend on the tile: share of the gain the ranker's top cells hold
                    ps = g / g.sum()
                    top = sv.argsort(descending=True)[: round(self.split_frac * nv)]
                    self.diag = {"captured": float(ps[top].sum()), "kl": float(aux.detach() + (ps * ps.clamp_min(1e-12).log()).sum())}
        four = fine_ok(k4)
        sel = torch.zeros_like(valid)
        if four:
            oracle = self.training and gain is not None and random.random() < self.q
            score = (gain if oracle else s.detach()).float().masked_fill(~valid, -float("inf"))
            sel[score.argsort(descending=True)[: round(self.split_frac * nv)]] = True
        g4 = [2 * g for g in g8]
        m1 = sel.view(*g8)
        m2 = m1
        for a in range(3):
            m2 = m2.repeat_interleave(2, a)
        c8 = (valid.view(*g8) & ~m1).nonzero()
        c4 = m2.nonzero()
        self.stats["leaves"] += len(c8) + len(c4)
        self.stats["uni4"] += nv * 8 if four else nv
        logits = xp.new_zeros(self.n_out, *xp.shape, dtype=torch.bfloat16)   # the decoder's own precision under autocast
        levels = [(0, c8, self.norm[0](tok8).view(*g8, -1), k8, g8)]
        if len(c4):
            tok4 = self.cache.tokens(i, 4.0, corner, g4, k4).to(dev).float()
            levels.append((1, c4, self.norm[1](tok4).view(*g4, -1), k4, g4))
        for lev, c, tok, kk, gg in levels:
            if not len(c):
                continue
            h = tok[c[:, 0], c[:, 1], c[:, 2]] + self.level_emb[lev]
            L6 = logits.view(self.n_out, g8[0], k8[0], g8[1], k8[1], g8[2], k8[2]) if lev == 0 else \
                logits.view(self.n_out, g4[0], k4[0], g4[1], k4[1], g4[2], k4[2])
            step = max(1, (1 << 24) // (self.n_out * max(kk[0] * kk[1] * kk[2], self.cells ** 3)))
            for a in range(0, len(c), step):
                b = min(a + step, len(c))

                def run(z, kk=kk):
                    return F.interpolate(self.decode(z[:, :, None, None, None]), size=kk, mode="trilinear",
                                         align_corners=False)
                p = checkpoint(run, h[a:b], use_reentrant=False)
                ci = c[a:b]
                L6[:, ci[:, 0], :, ci[:, 1], :, ci[:, 2], :] = p.to(logits.dtype)
        self.aux = aux
        return logits[:, : V[0], : V[1], : V[2]]
