"""Dmix gerarchico + kernel continuo (user 29-30/10, args.hier_dec=true; the artifact "Decoder Dmix vs CubeDec").
One token per cell, coarse to fine: 16 and 8 mm every cell (cached tokens, last ++ the depth quartiles, one projection
-> W for every level); 4 / 2 mm the children of the cells above with P(mixed) > TAU (training: no label decides what is
read), read live; each child = parent + MLP(parent ++ encoder -> W ++ its voxels' KernelEmbed -> W). P(mixed): one logit
per 8 / 4 mm cell, class-balanced BCE on detached tokens. The tokens become one field on the voxels: the 16 mm tokens plus
each finer level's details (token - parent, zero where nothing was read), each level trilinear (a linear B-spline over
its cell centres). Two continuous-kernel layers on the voxels (stem, head) plus the field through a 1x1, then a second
continuous-kernel layer, decide each voxel. The kernels: a 2 mm window centred on the voxel, integrated over each
voxel's extent (the KernelEmbed's idea at stride 1), one set of weights for every spacing. Eval: every cell read, the
threshold a mask; the test's chosen on val (segrun, one pass over TAUS). Generic segrun path (forward -> logits
[B, K, *T], tile_corner, sel_y / aux as CubeDec's selector)."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .token_cache import TokenCache, fine_ok

LV = (16.0, 8.0, 4.0, 2.0)
W, C = 256, 32           # token width, stem width
TAU, TAUS = 0.5, (0.5, 0.3, 0.1, 0.0)   # training: read the children of the cells with P(mixed) > TAU; the test's
                                        # threshold chosen on val among TAUS (segrun, one pass: every cell read, masks)
DROP = 0.1               # dropout on the encoder's 4096-d tokens (as CubeDec: small datasets)
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
    """`encoder` frozen: its cached 16 / 8 mm tokens (`cache`, every cell), and read live (no grad) at 4 / 2 mm under the
    cells of the level above with P(mixed) > tau (the validations during training too). `full` (segrun: the threshold's
    choice and the test): every cell read, each threshold of `read_taus` a mask (a cell counts if its ancestors at the live
    levels pass it), several -> their log-probabilities concatenated."""

    tile_corner = True
    pad_batch = True   # segrun: training tiles of different shapes padded into one batch (any tile shape reads)

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
        # one projection for every level (user 30/10: the encoder reads every grain in one space; four of them were 80 %
        # of the parameters and where HIE overfit, train Dice .585 vs val .176)
        self.enc = nn.Linear(c, W)
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(3 * W, W), nn.GELU(), nn.Linear(W, W)) for _ in LV[1:])
        self.vox = KernelEmbed(W, rank=8, basis="bspline")   # one for 8 / 4 / 2 mm: B-splines in the cell's own coordinate
        # P(mixed) of the 8 / 4 mm cells (what decides the 4 / 2 mm reads): one logit, class-balanced BCE on detached
        # tokens (only where to read; the tokens learn from the voxels' loss alone); bias -2: little read at the start
        self.lhead = nn.ModuleList(nn.Linear(W, 1) for _ in LV[1:3])
        for h in self.lhead:
            nn.init.constant_(h.bias, -2.0)
        # the field enters the head through a 1x1 (user 29/10: smooth over >= 2 mm already, a 2 mm kernel on it added
        # little at 8x CubeDec's step), applied on the cells before the interpolation (both linear: the same field,
        # interpolated at C channels, not W); no bias: a cell not read stays exactly 0
        self.fproj = nn.Linear(W, C, bias=False)
        # two CK-conv layers on the voxels (user 30/10: Willis' small arteries, errors inside the cells read)
        self.stem, self.head, self.head2 = CKConv(1, C), CKConv(C, C), CKConv(C, C)
        self.out = nn.Conv3d(C, n_out, 1)
        self.tau, self.read_taus, self.TAUS = TAU, [TAU], TAUS
        self.full = False   # eval reads every cell (thresholds as masks): segrun sets it for the threshold's choice and the test
        self.sel_y, self.aux, self._n = None, 0.0, 0
        self.off = set()    # levels left out of the field (diagnosis: what each one adds)
        self.trace = None   # a list: each tile's cells read per level are appended (diagnosis, bench_p28/tools/diag_hier.py)
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
                                 [[int(v) for v in r[7:10]] for r in rows], ys)   # [B, thresholds, C, *T]
        groups = {}   # the tiles of one spacing: one kernel, one convolution per layer
        for b, sp in enumerate(sps):
            groups.setdefault(sp, []).append(b)
        out, order = [], []
        for sp, idx in groups.items():
            # the group's tiles by slices: advanced indexing of the [B, C, 160^3] field (~1e9 values) failed on the GPU
            # (CUDA "invalid configuration argument", 29/10)
            pick = (lambda t: t) if len(idx) == len(x) else (lambda t, idx=idx: torch.cat([t[b:b + 1] for b in idx]))
            a, fl = self.head(F.gelu(self.stem(pick(x), sp)), sp), pick(field)
            o = [self.out(F.gelu(self.head2(F.gelu(a + fl[:, m].to(a.dtype)), sp))) for m in range(fl.shape[1])]
            # several thresholds: each one's log-probabilities (sliding's softmax over all of them then gives each
            # threshold's probabilities / their count: the same argmax per threshold)
            out.append(o[0] if len(o) == 1 else torch.cat([q.float().log_softmax(1) for q in o], 1))
            order += idx
        if ys is not None:
            self.aux, self.sel_y = aux, None
            self._n += 1
            if self._n % 625 == 0:
                print(f"[seg] hier_dec: cell heads' BCE {float(aux):.4f} (in the loss above)", flush=True)
        out = torch.cat(out)
        return out if order == sorted(order) else out[torch.tensor(order, device=x.device).argsort()]

    def fields(self, xs: torch.Tensor, sps, ids, corners, ys) -> tuple:
        """Tiles xs [B, *T] -> their fields [B, thresholds, C, *T] and the cell heads' loss (ys: the tiles' labels,
        training), level by level over the whole batch: the tiles' cells concatenated, one call per layer and one live
        reading per level."""
        from ..data.volume import patch_dims, patchify
        B, T, dev = len(xs), tuple(xs.shape[1:]), xs.device
        taus = torch.tensor(self.read_taus if self.full and not self.training else [self.tau], device=dev)
        nt = len(taus)
        ks = [[[4 * a for a in k4], [2 * a for a in k4], list(k4), list(patch_dims(sp, 2.0))]
              for k4, sp in ((self.cache.meta(i)["k4"], sp) for i, sp in zip(ids, sps))]
        spt = [torch.tensor(sp, device=dev, dtype=torch.float32) for sp in sps]
        field, loss, prev, live = [0.0] * B, 0.0, [None] * B, list(range(B))
        for j, mm in enumerate(LV):
            dense = mm in self.cache.levels   # cached: every cell; live: the children of the cells passing the threshold
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
                    keep, inc = torch.arange(N, device=dev), torch.ones(N, nt, dtype=torch.bool, device=dev)
                else:
                    g = torch.stack(torch.meshgrid(*[torch.arange(a, device=dev) for a in n], indexing="ij"), -1).view(-1, 3)
                    ctr = (g + torch.tensor(c0, device=dev)) * torch.tensor(k, device=dev) + torch.tensor(k, device=dev) // 2
                    pk, pc0, pn, prow, pt, pm, pinc = prev[b]
                    pq = (ctr // torch.tensor(pk, device=dev) - torch.tensor(pc0, device=dev)).clamp(min=0)
                    pq = torch.minimum(pq, torch.tensor(pn, device=dev) - 1)
                    pidx = (pq[:, 0] * pn[1] + pq[:, 1]) * pn[2] + pq[:, 2]        # each cell's parent on its grid
                    prr = (prow.cumsum(0) - 1).clamp(min=0)                        # parent cell -> its row (read ones)
                    ok = prow[pidx]
                    if not dense and not (self.full and not self.training):   # reads what passes; full: all, masks
                        ok = ok & (pm[prr[pidx]] > self.tau)
                    keep = ok.nonzero()[:, 0]
                    if not len(keep):
                        continue
                    r = prr[pidx[keep]]
                    par.append(pt[r])
                    inc = pinc[r] if dense else pinc[r] & (pm[r][:, None] > taus)
                    # ponytail: a tile not starting on this level's grid (sliding at val / test) reads its first cells
                    # zero-padded; tiles from the scan instead of xs if the tile borders show
                    xp = F.pad(xs[b].float(), [p for f, a, kk, t_ in zip(reversed(front), reversed(n), reversed(k), reversed(T))
                                               for p in (f, a * kk - f - t_)])
                    pats.append(patchify(xp, k)[keep])
                    qs.append(g[keep])
                members.append(b)
                geo.append((k, c0, n, front, N, keep, inc))
            live = members
            if not live:
                break
            if dense:   # the cached levels' tokens, else the frozen encoder now on the cells chosen
                enc = torch.cat([self._cached(ids[b], mm, g_[1], g_[2], g_[0], dev)[g_[5]] for b, g_ in zip(live, geo)])
            else:
                enc = self._read_many(pats, qs, [g_[2] for g_ in geo], [spt[b] for b in live], mm, [g_[0] for g_ in geo])
            e = self.enc(F.dropout(enc, DROP, self.training))
            if j == 0:
                t = d = e
            else:
                self.vox.patch_mm = mm
                vox = torch.cat([self.vox(pat, self.vox.weights(spt[b], g_[0])) for b, pat, g_ in zip(live, pats, geo)])
                p_ = torch.cat(par)
                d = self.mlp[j - 1](torch.cat([p_, e, vox.to(p_.dtype)], -1))
                t = p_ + d
            cnt = [len(g_[5]) for g_ in geo]
            pm = torch.zeros(len(t), device=dev)
            if j in (1, 2):   # P(mixed) of the 8 / 4 mm cells: what the next level reads
                lg = self.lhead[j - 1](t.detach()).float()[:, 0]
                pm = lg.sigmoid().detach()
                if ys is not None:
                    tg = torch.cat([self._mixed(ys[b], g_[3], g_[2], g_[0], g_[5]) for b, g_ in zip(live, geo)])
                    v = tg >= 0
                    if v.any():
                        pos = (tg[v] == 1).float()
                        w = (len(pos) - pos.sum()) / pos.sum().clamp(min=1)   # class-balanced: miste vs pure
                        loss = loss + F.binary_cross_entropy_with_logits(lg[v], pos, pos_weight=w)
            f = self.fproj(d)
            for b, (k, c0, n, front, N, keep, inc), tb, pb, fb in zip(live, geo, t.split(cnt), pm.split(cnt), f.split(cnt)):
                row = torch.zeros(N, dtype=torch.bool, device=dev)
                row[keep] = True
                prev[b] = (k, c0, n, row, tb, pb, inc)
                if self.trace is not None:
                    self.trace.append((corners[b], j, k, c0, n, front, keep, pb, fb.detach().float().norm(dim=1)))
                fb = fb * 0 if j in self.off else fb
                # per threshold the cells it keeps (a cell not kept: detail 0), all thresholds in one interpolation
                grid = fb.new_zeros(nt, N, C)
                grid[:, keep] = fb[None] * inc.T[:, :, None].to(fb.dtype)
                up = F.interpolate(grid.transpose(1, 2).reshape(nt, C, *n), size=[a * kk for a, kk in zip(n, k)],
                                   mode="trilinear", align_corners=False)
                field[b] = field[b] + up[:, :, front[0]: front[0] + T[0], front[1]: front[1] + T[1], front[2]: front[2] + T[2]]
        return torch.stack(field), loss

    def _cached(self, i, mm, c0, n, k, dev) -> torch.Tensor:
        return self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).to(dev).float()

    @torch.no_grad()
    def _read_many(self, pats, qs, ns, sps, mm, ks) -> torch.Tensor:
        """The frozen encoder at level mm on several tiles' chosen cells (q [m, 3] on grid n, patches [m, P]): one
        sequence per tile along the curve LIVE (Dmix's live levels too), tiles packed in calls of <= 2^19 tokens (Dmix's
        MIX_MAX: a Triton grid failed near 1e6) -> [sum m, last ++ taps], tiles in order."""
        from ..curves import keys
        pe, dev = self.encoder.patch_embed, pats[0].device
        mm0, pe.patch_mm = pe.patch_mm, mm
        out, grp, tot = [], [], 0
        try:
            for i in range(len(qs) + 1):
                if grp and (i == len(qs) or tot + len(qs[i]) > 1 << 19):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        os_ = [keys(LIVE, qs[a], torch.tensor(ns[a], device=dev), self.seed).argsort() for a in grp]
                        xe = torch.cat([pe(pats[a], pe.weights(sps[a], ks[a]))[o] for a, o in zip(grp, os_)])
                        cu = torch.tensor([0] + [len(qs[a]) for a in grp], device=dev).cumsum(0).to(torch.int32)
                        h = self.encoder.flat(xe, cu, pos=torch.cat([qs[a][o] for a, o in zip(grp, os_)])).float()
                    out += [hh[o.argsort()] for hh, o in zip(h.split([len(qs[a]) for a in grp]), os_)]
                    grp, tot = [], 0
                if i < len(qs):
                    grp.append(i)
                    tot += len(qs[i])
        finally:
            pe.patch_mm = mm0
        return torch.cat(out)

    @staticmethod
    def _mixed(y, front, n, k, keep) -> torch.Tensor:
        """The cells keep: 1 if they hold more than one label, 0 if one, -1 if none (255 = ignore, the padding too)."""
        from ..data.volume import patchify
        yp = F.pad(y.float(), [p for f, a, kk, t_ in zip(reversed(front), reversed(n), reversed(k), reversed(y.shape))
                               for p in (f, a * kk - f - t_)], value=255)
        v = patchify(yp, k)[keep]
        ok = v != 255
        lo, hi = torch.where(ok, v, 1e9).amin(1), torch.where(ok, v, -1.0).amax(1)
        return torch.where(ok.any(1), (lo != hi).long(), -1)
