"""APT at inference (Choudhury et al., arXiv 2510.18091, "Accelerating Vision Transformers with Adaptive Patch
Sizes"): the scan is cut into an octree of patches, coarse where the content is simple and fine where it is not, and
the encoder reads one token per leaf, in one sequence along a space-filling curve.

Differences from the paper, forced by this repo (and what they buy):
- no patch-embedding surgery: `KernelEmbed` is a continuous kernel in mm (B-spline basis: one family for every cell
  size), so a leaf of any size gets its token straight from its native voxels (nothing resampled), the encoder's
  weights are untouched and nothing is trained;
- an octree, not a quadtree: a cell of k voxels per axis splits into 8 cells of k/2. The base cell is
  k0 = 2^(levels-1) * round(base_mm / (2^(levels-1) * spacing)) voxels per axis, so every split is exact (the
  pretraining's round(patch_mm / spacing) can differ from k0 by up to 2^(levels-2) voxels);
- ordering: leaves are sorted by the curve key (raster / morton / hilbert) of their minimum corner on the finest grid.
  An aligned cube is a contiguous range of the Morton and Hilbert keys, so a leaf's voxels stay consecutive.

A reading is an *arm*: {score: entropy | random, tau: [bits per level] | frac: [fraction per level] | match: <arm>}.
`entropy` is APT's criterion, the Shannon entropy (bits) of the cell's normalised-intensity histogram; a cell splits
when it is above `tau` (or, with `frac`, among the highest fraction of the cells still alive). `random` ignores the
content: with `match: <arm>` it splits, at every level, as many cells as that arm did on the same scan, the control
that isolates what the content criterion buys at equal token count.
"""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from .curves import keys
from .data.volume import patchify


def base_dims(spacing, base_mm: float, levels: int) -> tuple[int, int, int]:
    """Voxels per axis of the base cell: a multiple of 2^(levels-1) closest to base_mm (>= 2^(levels-1))."""
    f = 2 ** (levels - 1)
    return tuple(f * max(1, round(base_mm / (f * float(s)))) for s in spacing[:3])


def entropy(p: torch.Tensor, bins: int) -> torch.Tensor:
    """[N, P] normalised values (~[0, 1]; clamped) -> [N] Shannon entropy (bits) of each row's histogram."""
    idx = (p.clamp(0, 1) * bins).long().clamp_max(bins - 1)
    h = torch.zeros(len(p), bins, device=p.device).scatter_add_(1, idx, torch.ones_like(p))
    q = h / h.sum(1, keepdim=True)
    return -(q * torch.log2(q.clamp_min(1e-12))).sum(1)


@torch.no_grad()
def scores(vol: torch.Tensor, k0, levels: int, bins: int) -> list[torch.Tensor]:
    """vol [D,H,W] (zero-padded to multiples of k0) -> per level l < levels-1 the entropy of every cell of that
    level's grid [g0 * 2^l]."""
    out = []
    for l in range(levels - 1):
        k = tuple(x // 2 ** l for x in k0)
        g = [s // q for s, q in zip(vol.shape, k)]
        out.append(entropy(patchify(vol, k), bins).view(g))
    return out


def _split(score: torch.Tensor, alive: torch.Tensor, arm: dict, l: int, n: int | None) -> torch.Tensor:
    """Cells of this level that split: alive, and above arm["tau"][l] / in the top arm["frac"][l] / the top n."""
    if n is None and arm.get("frac") is not None:
        n = round(float(arm["frac"][l]) * int(alive.sum()))
    if n is None:
        return alive & (score > float(arm["tau"][l]))
    n = min(int(n), int(alive.sum()))
    if n <= 0:
        return torch.zeros_like(alive)
    s = torch.where(alive, score, score.new_full((), -float("inf"))).flatten()
    keep = torch.zeros_like(s, dtype=torch.bool)
    keep[s.topk(n).indices] = True
    return keep.view_as(alive) & alive


@torch.no_grad()
def partition(sc: list[torch.Tensor], g0, levels: int, arm: dict, counts: list[int] | None = None,
              gen: torch.Generator | None = None) -> tuple[list[torch.Tensor], list[int]]:
    """Entropy scores -> the leaves of the arm's octree: per level the coordinates [n, 3] (that level's grid units) of
    its leaves, and the number of cells split at each level (what a `match`ing arm copies)."""
    dev = sc[0].device if sc else torch.device("cpu")
    alive = torch.ones(tuple(int(g) for g in g0), dtype=torch.bool, device=dev)
    leaves, split = [], []
    for l in range(levels):
        if l < levels - 1:
            s = torch.rand(alive.shape, device=dev, generator=gen) if arm.get("score") == "random" else sc[l]
            ref = _split(s, alive, arm, l, None if counts is None else counts[l])
        else:
            ref = torch.zeros_like(alive)
        leaves.append((alive & ~ref).nonzero())
        split.append(int(ref.sum()))
        for a in range(3):   # a split cell's 8 children are alive on the next level
            ref = ref.repeat_interleave(2, a)
        alive = ref
    return leaves, split


def block(v: torch.Tensor, j: int, d: int, n: int) -> torch.Tensor:
    """[..., 2dn] = fwd (last block, taps) ++ bwd (same) -> block j's fwd ++ bwd [..., 2d] (fm/sfc.py `features`)."""
    return torch.cat([v[..., j * d: (j + 1) * d], v[..., (n + j) * d: (n + j + 1) * d]], -1)


class APT:
    """An encoder read by octree leaves. `read(vol, spacing)` -> {arm: {stat: [2dn]}} for every arm."""

    def __init__(self, encoder, curves: list[str], seed: int, fg_threshold: float, base_mm: float, levels: int,
                 arms: dict, bins: int = 16):
        self.encoder, self.curves, self.seed, self.fg_threshold = encoder, list(curves), seed, fg_threshold
        self.base_mm, self.levels, self.arms, self.bins = float(base_mm), int(levels), arms, int(bins)
        for name, a in arms.items():   # a `match`ed arm needs the arm it copies computed before it
            assert a.get("match") is None or list(arms).index(a["match"]) < list(arms).index(name), name

    def embed(self, vol, spacing, k0, leaves):
        """Leaves per level -> tokens [n, d], their mean value [n], finest-grid corners [n, 3], level [n]."""
        pe, mm0, f = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm, self.levels - 1
        tok, mean, corner, level = [], [], [], []
        try:
            for l, c in enumerate(leaves):
                if not len(c):
                    continue
                k = tuple(x // 2 ** l for x in k0)
                g = [s // q for s, q in zip(vol.shape, k)]
                x = patchify(vol, k)[(c[:, 0] * g[1] + c[:, 1]) * g[2] + c[:, 2]]
                pe.patch_mm = self.base_mm / 2 ** l   # the normalisation the kernel was trained with, per scale
                tok.append(pe(x, pe.weights(spacing.float().cpu(), k)))
                mean.append(x.mean(-1))
                corner.append(c * 2 ** (f - l))
                level.append(torch.full((len(c),), l, device=vol.device))
        finally:
            pe.patch_mm = mm0
        return torch.cat(tok), torch.cat(mean), torch.cat(corner), torch.cat(level)

    def encode(self, tok: torch.Tensor, corner: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
        """Leaf tokens [n, d] -> bi [n, 2dn] averaged over the curves (leaf order of `tok`)."""
        enc, n = self.encoder, len(tok)
        valid, bi = torch.ones(1, n, dtype=torch.bool, device=tok.device), 0
        for curve in self.curves:
            order = keys(curve, corner, dims, self.seed).argsort()
            back = order.argsort()
            s = tok[order][None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if enc.bidir == "late":
                    rev = torch.arange(n - 1, -1, -1, device=tok.device)
                    h = enc.packed(torch.cat([s, s[:, rev]]), valid.repeat(2, 1)).float()
                    fb = torch.cat([h[0], h[1][rev]], -1)[None]
                else:                                                         # one stream: bi = h ++ h
                    fb = enc.packed(s, valid).float().repeat(1, 1, 2)
            bi = bi + fb[0][back] / len(self.curves)
        return bi

    @torch.no_grad()
    def read(self, vol: torch.Tensor, spacing: torch.Tensor) -> dict:
        """vol [D,H,W] normalised values, spacing [3] mm."""
        k0 = base_dims(spacing.tolist(), self.base_mm, self.levels)
        vol = F.pad(vol, [p for n, q in zip(reversed(vol.shape), reversed(k0)) for p in (0, -n % q)])
        g0 = [s // q for s, q in zip(vol.shape, k0)]
        dims = torch.tensor(g0, device=vol.device) * 2 ** (self.levels - 1)
        sc = scores(vol, k0, self.levels, self.bins)
        gen = torch.Generator(vol.device).manual_seed(self.seed + int(vol.numel()))
        out, split = {}, {}
        for name, arm in self.arms.items():
            leaves, split[name] = partition(sc, g0, self.levels, arm, None if arm.get("match") is None
                                            else split[arm["match"]], gen)
            tok, mean, corner, level = self.embed(vol, spacing, k0, leaves)
            if vol.is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            bi = self.encode(tok, corner, dims)
            if vol.is_cuda:
                torch.cuda.synchronize()
            fg = mean > self.fg_threshold
            fg = fg if fg.any() else torch.ones_like(fg)
            res = {"time": torch.tensor([time.time() - t0]),
                   "ntok": torch.bincount(level, minlength=self.levels).float().cpu()}
            for suffix, w in (("", fg.float() * 8.0 ** (self.levels - 1 - level)), ("_tok", fg.float())):
                # "" weights a leaf by its volume (a uniform finest grid's mean, whatever the leaf sizes); _tok: one each
                w = w[:, None]
                m = (w * bi).sum(0) / w.sum()
                res |= {"mean" + suffix: m, "std" + suffix: ((w * (bi - m) ** 2).sum(0) / w.sum()).sqrt()}
            out[name] = res
        return out
