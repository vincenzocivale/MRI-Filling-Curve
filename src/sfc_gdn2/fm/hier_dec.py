"""Dmix gerarchico + kernel continuo (user 29/10, args.hier_dec=true; the artifact "Decoder Dmix vs CubeDec", proposal).
One token per cell, coarse to fine: 16 mm every cell of the tile (cached tokens, last ++ the depth quartiles -> W); at
8 / 4 / 2 mm the children of the parent level's cells ranked most likely mixed (the top Q of the cells read there, in
training as at test: no label decides what is read), each child = parent + MLP(parent ++ encoder -> W ++ its voxels'
KernelEmbed -> W). A cell head per level (K classes pure | mixed) ranks the cells and is trained on the cell's labels.
The tokens become one field on the voxels: the 16 mm tokens plus each finer level's details (token - parent, zero where
nothing was read), each level trilinear (a linear B-spline over its cell centres). A continuous-kernel stem reads the
voxels; a continuous-kernel head on it, plus the field through a 1x1, decides each voxel. Both kernels: a 2 mm window
centred on the voxel, integrated over each voxel's extent (the KernelEmbed's idea at stride 1), so one set of weights for every spacing.
Generic segrun path (forward -> logits [B, K, *T], tile_corner, sel_y / aux as CubeDec's selector)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .token_cache import TokenCache, fine_ok

LV = (16.0, 8.0, 4.0, 2.0)
W, C, Q = 256, 32, 0.3   # token width, stem width, share of a level's read cells whose children are read
LIVE = "hilbert"         # the 4 / 2 mm cells read live along one curve (29/10: 3 curves were 85 % of the field's forward)
R_MM, RANK = 1.0, 4      # CK-conv: window radius (mm, centred: 2 mm across), B-splines per axis


def axis_taps(s: float, rank: int, dev, r: float = R_MM) -> torch.Tensor:
    """[2J + 1, rank]: per tap (voxels at offsets -J..J of spacing s mm, the window [-r, r] mm centred on the voxel) the
    integral of each B-spline (on the window) over the tap voxel's part of the window, in mm (midpoint rule, 0.02 mm)."""
    from ..model import bspline_basis
    J = math.ceil((r + s / 2) / s) - 1   # 1 mm voxels: 3 taps (the outer two half inside), 0.4 mm: 5, 3 mm: 1
    q = max(1, math.ceil(s / 0.02))
    t = (torch.arange((2 * J + 1) * q, device=dev, dtype=torch.float32) + 0.5) * (s / q) - (J + 0.5) * s
    phi = bspline_basis((t + r) / (2 * r), rank) * (t.abs() < r).float()[:, None]
    return phi.view(2 * J + 1, q, rank).mean(1) * s


class CKConv(nn.Module):
    """A convolution whose kernel is continuous in mm: K = G . (phi_x (x) phi_y (x) phi_z), separable B-splines over the
    window, each tap the kernel's integral over its voxel / the window's volume (a constant input gives the same output
    at every spacing)."""

    def __init__(self, cin: int, cout: int, rank: int = RANK):
        super().__init__()
        self.rank = rank
        self.G = nn.Parameter(torch.randn(cout, cin, rank, rank, rank) * rank ** 1.5 / cin ** 0.5)   # ~unit output
        self.bias = nn.Parameter(torch.zeros(cout))

    def forward(self, x: torch.Tensor, sp) -> torch.Tensor:
        P = [axis_taps(float(s), self.rank, x.device) for s in sp]
        with torch.autocast(x.device.type, enabled=False):
            k = torch.einsum("oiabc,xa,yb,zc->oixyz", self.G, *P) / (2 * R_MM) ** 3
        return F.conv3d(x, k.to(x.dtype), self.bias.to(x.dtype), padding=[(len(p) - 1) // 2 for p in P])


class HierDec(nn.Module):
    """`encoder` frozen: its cached 16 / 8 mm tokens (`cache`), and read live (no grad) on the 4 / 2 mm cells chosen."""

    tile_corner = True

    def __init__(self, encoder, cache: str, n_out: int, curves=(), seed: int = 17):
        super().__init__()
        from ..model import KernelEmbed
        self.encoder, self.curves, self.seed = encoder, list(curves), seed
        if str(cache).startswith("live:"):   # a task without a cache yet: written as segrun loads each scan
            from .token_cache import LiveCache
            self.cache, self.enc_curves = LiveCache(str(cache)[5:], self), list(curves)
        else:
            self.cache = TokenCache(cache)
        self.cache.levels = LV[:2]
        c = encoder.dim * (1 + len(encoder.taps))
        self.enc = nn.ModuleList(nn.Linear(c, W) for _ in LV)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(3 * W, W), nn.GELU(), nn.Linear(W, W)) for _ in LV[1:])
        self.vox = KernelEmbed(W, rank=8, basis="bspline")   # one for 8 / 4 / 2 mm: B-splines in the cell's own coordinate
        self.lhead = nn.ModuleList(nn.Linear(W, n_out + 1) for _ in LV)
        # the field enters the head through a 1x1 (user 29/10: smooth over >= 2 mm already, a 2 mm kernel on it added
        # little at 8x CubeDec's step), applied on the cells before the interpolation (both linear: the same field,
        # interpolated at C channels, not W); no bias: a cell not read stays exactly 0
        self.fproj = nn.Linear(W, C, bias=False)
        self.stem, self.head, self.out = CKConv(1, C), CKConv(C, C), nn.Conv3d(C, n_out, 1)
        self.sel_y, self.aux = None, 0.0
        # the CK-convs' shapes depend on the spacing (a few per task): cuDNN's autotuning pays (29/10: 7.5 -> 6.5 s per step)
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def tile_max(n_out: int) -> float:
        return float(160 ** 3 * n_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        T = tuple(x.shape[2:])
        ys = None if self.sel_y is None or not self.training else self.sel_y.to(x.device).reshape(len(x), *T)
        rows = ctx.tolist()
        sps = [tuple(r[:3]) for r in rows]
        field, aux = self.fields(x[:, 0], sps, [self.cache.ids[int(r[6])] for r in rows],
                                 [[int(v) for v in r[7:10]] for r in rows], ys)
        groups = {}   # the tiles of one spacing: one kernel, one convolution per layer
        for b, sp in enumerate(sps):
            groups.setdefault(sp, []).append(b)
        out, order = [], []
        for sp, idx in groups.items():
            # the group's tiles by slices: advanced indexing of the [B, C, 160^3] field (~1e9 values) failed on the GPU
            # (CUDA "invalid configuration argument", 29/10)
            pick = (lambda t: t) if len(idx) == len(x) else (lambda t, idx=idx: torch.cat([t[b:b + 1] for b in idx]))
            s = F.gelu(self.stem(pick(x), sp))
            out.append(self.out(F.gelu(self.head(s, sp) + pick(field).to(s.dtype))))
            order += idx
        if ys is not None:
            self.aux, self.sel_y = aux, None
        out = torch.cat(out)
        return out if order == sorted(order) else out[torch.tensor(order, device=x.device).argsort()]

    def fields(self, xs: torch.Tensor, sps, ids, corners, ys) -> tuple:
        """Tiles xs [B, *T] -> their fields [B, C, *T] and the cell heads' loss (ys: the tiles' labels, training), level
        by level over the whole batch: the tiles' cells concatenated, one call per layer and one live reading per level."""
        from ..data.volume import patch_dims, patchify
        B, T, dev = len(xs), tuple(xs.shape[1:]), xs.device
        ks = [[[4 * a for a in k4], [2 * a for a in k4], list(k4), list(patch_dims(sp, 2.0))]
              for k4, sp in ((self.cache.meta(i)["k4"], sp) for i, sp in zip(ids, sps))]
        spt = [torch.tensor(sp, device=dev, dtype=torch.float32) for sp in sps]
        field, loss, prev, live = [0.0] * B, 0.0, [None] * B, list(range(B))
        for j, mm in enumerate(LV):
            members, geo, par, pats, qs = [], [], [], [], []
            for b in live:
                k, corner = ks[b][j], corners[b]
                if j and (not fine_ok(k) or k == ks[b][j - 1]):   # a scan too coarse for this level (and the finer ones)
                    continue
                c0 = [c // kk for c, kk in zip(corner, k)]                       # the level's cells on the scan's grid
                n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
                front = [c - a * kk for c, a, kk in zip(corner, c0, k)]          # tile voxels before its first cell
                N = math.prod(n)
                if j == 0:
                    keep = torch.arange(N, device=dev)
                else:
                    g = torch.stack(torch.meshgrid(*[torch.arange(a, device=dev) for a in n], indexing="ij"), -1).view(-1, 3)
                    ctr = (g + torch.tensor(c0, device=dev)) * torch.tensor(k, device=dev) + torch.tensor(k, device=dev) // 2
                    pk, pc0, pn, prow, pt, pl = prev[b]
                    pq = (ctr // torch.tensor(pk, device=dev) - torch.tensor(pc0, device=dev)).clamp(min=0)
                    pq = torch.minimum(pq, torch.tensor(pn, device=dev) - 1)
                    pidx = (pq[:, 0] * pn[1] + pq[:, 1]) * pn[2] + pq[:, 2]        # each cell's parent on its grid
                    mix = pl.float().softmax(-1)[:, -1]                            # the parents read: P(mixed)
                    chosen = torch.zeros(len(prow), dtype=torch.bool, device=dev)
                    chosen[prow.nonzero()[:, 0][mix.topk(math.ceil(Q * len(mix))).indices]] = True
                    keep = chosen[pidx].nonzero()[:, 0]
                    if not len(keep):
                        continue
                    par.append(pt[(prow.cumsum(0) - 1)[pidx[keep]]])
                    # ponytail: a tile not starting on this level's grid (sliding at val / test) reads its first cells
                    # zero-padded; tiles from the scan instead of xs if the tile borders show
                    xp = F.pad(xs[b].float(), [p for f, a, kk, t_ in zip(reversed(front), reversed(n), reversed(k), reversed(T))
                                               for p in (f, a * kk - f - t_)])
                    pats.append(patchify(xp, k)[keep])
                    qs.append(g[keep])
                members.append(b)
                geo.append((k, c0, n, front, N, keep))
            live = members
            if not live:
                break
            if mm in self.cache.levels:   # the cached levels' tokens, else the frozen encoder now on the cells chosen
                enc = torch.cat([self._cached(ids[b], mm, c0, n, k, dev)[keep] for b, (k, c0, n, _, _, keep) in zip(live, geo)])
            else:
                enc = self._read_many(pats, qs, [g_[2] for g_ in geo], [spt[b] for b in live], mm, [g_[0] for g_ in geo])
            e = self.enc[j](enc)
            if j == 0:
                t = d = e
            else:
                self.vox.patch_mm = mm
                vox = torch.cat([self.vox(pat, self.vox.weights(spt[b], g_[0])) for b, pat, g_ in zip(live, pats, geo)])
                p_ = torch.cat(par)
                d = self.mlp[j - 1](torch.cat([p_, e, vox.to(p_.dtype)], -1))
                t = p_ + d
            lg, f = self.lhead[j](t), self.fproj(d)
            cnt = [len(g_[5]) for g_ in geo]
            for b, (k, c0, n, front, N, keep), tb, lb, fb in zip(live, geo, t.split(cnt), lg.split(cnt), f.split(cnt)):
                if ys is not None:
                    loss = loss + self._cell_loss(lb, ys[b], front, n, k, keep) / B
                row = torch.zeros(N, dtype=torch.bool, device=dev)
                row[keep] = True
                prev[b] = (k, c0, n, row, tb, lb.detach())
                grid = fb.new_zeros(N, C).index_copy(0, keep, fb) if j else fb   # a cell not read: detail 0
                up = F.interpolate(grid.T.reshape(1, C, *n), size=[a * kk for a, kk in zip(n, k)], mode="trilinear",
                                   align_corners=False)
                field[b] = field[b] + up[0, :, front[0]: front[0] + T[0], front[1]: front[1] + T[1], front[2]: front[2] + T[2]]
        return torch.stack(field), loss

    def _cached(self, i, mm, c0, n, k, dev) -> torch.Tensor:
        return self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).to(dev).float()

    @torch.no_grad()
    def _read_many(self, pats, qs, ns, sps, mm, ks) -> torch.Tensor:
        """The frozen encoder at level mm on several tiles' chosen cells (q [m, 3] on grid n, patches [m, P]) in one call:
        one sequence per tile along the curve LIVE (Dmix's live levels too) -> [sum m, last ++ taps], tiles in order.
        ponytail: one call whatever the count (Dmix cuts at 2^19 tokens: a Triton grid failed near 1e6); ~1e5 at most here."""
        from ..curves import keys
        pe, dev = self.encoder.patch_embed, pats[0].device
        mm0, pe.patch_mm = pe.patch_mm, mm
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                os_ = [keys(LIVE, q, torch.tensor(n, device=dev), self.seed).argsort() for q, n in zip(qs, ns)]
                xe = torch.cat([pe(p, pe.weights(s, k))[o] for p, s, k, o in zip(pats, sps, ks, os_)])
                cu = torch.tensor([0] + [len(q) for q in qs], device=dev).cumsum(0).to(torch.int32)
                h = self.encoder.flat(xe, cu, pos=torch.cat([q[o] for q, o in zip(qs, os_)])).float()
        finally:
            pe.patch_mm = mm0
        return torch.cat([hh[o.argsort()] for hh, o in zip(h.split([len(q) for q in qs]), os_)])

    @staticmethod
    def _cell_loss(lg, y, front, n, k, keep) -> torch.Tensor:
        """Cross-entropy of the cells keep's heads [m, K + 1] against their labels: the class if the cell holds one,
        mixed (K) if more, ignored if none labelled (255 = ignore, the padding too)."""
        from ..data.volume import patchify
        K = lg.shape[1] - 1
        yp = F.pad(y.float(), [p for f, a, kk, t_ in zip(reversed(front), reversed(n), reversed(k), reversed(y.shape))
                               for p in (f, a * kk - f - t_)], value=255)
        v = patchify(yp, k)[keep].long()
        ok = v != 255
        cnt = torch.zeros(len(v), K + 1, device=v.device).scatter_add_(1, torch.where(ok, v, K), ok.float())[:, :K]
        tgt = torch.where((cnt > 0).sum(1) > 1, K, cnt.argmax(1))
        tgt = torch.where(ok.any(1), tgt, -100)
        return F.cross_entropy(lg.float(), tgt, ignore_index=-100) if (tgt >= 0).any() else lg.sum() * 0
