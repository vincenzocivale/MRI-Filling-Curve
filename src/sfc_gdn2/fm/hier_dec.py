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

import functools
import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from .token_cache import TokenCache, fine_ok

LV = (16.0, 8.0, 4.0, 2.0)   # (30/10: a 1 mm level, hier5, cost +20-30 % per step; dropped by the user)
W, C = 256, 32           # token width, stem width
TAU, TAUS = 0.5, (0.5, 0.3, 0.1, 0.0)   # training: read the children of the cells with P(mixed) > TAU; the test's
                                        # threshold chosen on val among TAUS (segrun, one pass: every cell read, masks)
DROP = 0.1               # dropout on the encoder's 4096-d tokens (as CubeDec: small datasets)
LIVE_MAX = 1 << 17   # tokens per live encoder call (30/10: 2^19 ran out of memory on tumor_ucsf, a batch's earlier shape
                     # groups' graphs held until the one backward)
SNAKE_GAP = 16       # v10 (user 30/10, "il serpente"): live cells read as stretches of the curve; a gap of <= this many unread
                     # cells between open ones read too (context only), a longer one starts a new sequence
OPEN_MAX = 1 << 14   # v10: open cells per tile and live level at most (the highest parent P(mixed)): tumor_ucsf ran out of memory
SEQ_MAX = 4095       # v10: sequences per live call at most (GDN-2 kernels: a grid axis of sequences x 16 heads <= 65535)
LIVE = "hilbert"         # the 4 / 2 mm cells read live along one curve (29/10: 3 curves were 85 % of the field's forward)
R_MM, RANK = 1.0, 4      # CK-conv: window radius (mm, centred: 2 mm across), B-splines per axis
PDROP = float(__import__("os").environ.get("SFC_PDROP", "0.25"))   # HierUNet: path dropout per training tile (image off, else field off); v8 0.25, v9-v13 off,
                         # v14 on for every task (fetal test MRE 1.43 with it, 1.65 without; Willis unchanged)
ALPHAS = (1.0, 0.5, 0.0)   # HierUNet v9: the U-Net head's weight in the mix with the token head, chosen on val


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


def _patchify_b(vol: torch.Tensor, k) -> torch.Tensor:
    """data.volume.patchify on a batch: [G, D, H, W] (multiples of k) -> [G, N, k0*k1*k2], cells in raster order."""
    G, (a, b, c) = len(vol), k
    gx, gy, gz = (s // q for s, q in zip(vol.shape[1:], k))
    return vol.reshape(G, gx, a, gy, b, gz, c).permute(0, 1, 3, 5, 2, 4, 6).reshape(G, gx * gy * gz, a * b * c)


def _down_fg(y: torch.Tensor, s, k: int) -> torch.Tensor:
    """Labels [B, *T] -> the deep supervision's [B, *T / s] keeping small structures (31/10: a stride dropped 2 mm
    microbleeds at 1/4): a block takes its most frequent foreground class if it holds any, else background if it holds
    any, else ignore (255)."""
    best, cnt = None, None
    for c in range(1, k):
        f = F.avg_pool3d((y == c).float()[:, None], s, s)[:, 0]
        best = torch.where(f > cnt, c, best) if best is not None else torch.where(f > 0, c, 0)
        cnt = torch.maximum(cnt, f) if cnt is not None else f
    bg = F.avg_pool3d((y == 0).float()[:, None], s, s)[:, 0] > 0
    return torch.where(cnt > 0, best, torch.where(bg, 0, 255)).long()


class HierDec(nn.Module):
    """`encoder` frozen: its cached 16 / 8 mm tokens (`cache`, every cell), and read live (no grad) at 4 / 2 mm under the
    cells of the level above with P(mixed) > tau (the validations during training too). `full` (segrun: the threshold's
    choice and the test): every cell read, each threshold of `read_taus` a mask (a cell counts if its ancestors at the live
    levels pass it), several -> their log-probabilities concatenated."""

    tile_corner = True
    any_tile = True   # segrun: tiles no larger than their volume, a batch's tiles grouped by shape (group_step)

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
        # LayerNorm first (as CubeDec's ctok; 30/10: without it the field was ~250x the image branch, whose CK-convs never
        # left their init)
        self.enc = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, W))
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
        self.fnorm = None   # HierUNet: each cell's field layer-normed
        self.grid = None    # HierUNet: per level (output, stride per axis); None: one field at the voxels
        self.tau, self.read_taus, self.TAUS = TAU, [TAU], TAUS
        self.full = False   # eval reads every cell (thresholds as masks): segrun sets it for the threshold's choice and the test
        self.sel_y, self.aux, self._n = None, 0.0, 0
        self.off = set()    # levels left out of the field (diagnosis: what each one adds)
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
        training), level by level over the whole batch, vectorised (31/10, user: "fallo e basta", B 64 tiles): per level
        the tiles of one geometry (spacing, cell grid, offset to the parent grid: tiles start on the 16 mm grid, so a
        batch holds few) as [tiles, cells] tensors, one call per layer and one live reading per level."""
        from ..data.volume import patch_dims
        B, T, dev = len(xs), tuple(xs.shape[1:]), xs.device
        taus = torch.tensor(self.read_taus if self.full and not self.training else [self.tau], device=dev)
        nt = len(taus)
        ks = [[[4 * a for a in k4], [2 * a for a in k4], list(k4), list(patch_dims(sp, 2.0))]
              for k4, sp in ((self.cache.meta(i)["k4"], sp) for i, sp in zip(ids, sps))]
        grid_ = self.grid or [(0, (1, 1, 1))] * len(LV)
        sz = {o: [t // a for t, a in zip(T, st)] for o, st in grid_}
        # the field in the autocast dtype (30/10: tumor_ucsf's 160^3 tiles, B 8, ran out of memory with fp32 fields)
        fdt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else torch.float32
        field = {o: [xs.new_zeros(nt, C, *z, dtype=fdt) for _ in range(B)] for o, z in sz.items()}
        loss, prev, live = 0.0, [None] * B, list(range(B))
        for j, mm in enumerate(LV):
            dense = mm in self.cache.levels   # cached: every cell; live: the children of the cells passing the threshold
            groups = {}
            for b in live:
                k, corner = ks[b][j], corners[b]
                if j and (not fine_ok(k) or k == ks[b][j - 1]):   # a scan too coarse for this level (and the finer ones)
                    continue
                c0 = [c // kk for c, kk in zip(corner, k)]                       # the level's cells on the scan's grid
                n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
                front = [c - a * kk for c, a, kk in zip(corner, c0, k)]          # tile voxels before its first cell
                rel = () if not j else (tuple(prev[b][0]), tuple(prev[b][2]),
                                        tuple(a * kk - pa * pk for a, kk, pa, pk in zip(c0, k, prev[b][1], prev[b][0])))
                groups.setdefault((tuple(sps[b]), tuple(k), tuple(n), tuple(front), rel), []).append((b, c0))
            G_, ENC, PAR, VOX, TG, INC = [], [], [], [], [], []
            for (sp, k, n, front, rel), items in groups.items():
                bs = [b for b, _ in items]
                N, G = math.prod(n), len(bs)
                pads = [p for f, a, kk, t_ in zip(reversed(front), reversed(n), reversed(k), reversed(T)) for p in (f, a * kk - f - t_)]
                if j == 0:
                    ok = torch.ones(G, N, dtype=torch.bool, device=dev)
                else:
                    g = self._grid(n, dev)
                    pk, pn, off = rel
                    pq = ((g * torch.tensor(k, device=dev) + torch.tensor(k, device=dev) // 2 + torch.tensor(off, device=dev))
                          // torch.tensor(pk, device=dev)).clamp(min=0)
                    pq = torch.minimum(pq, torch.tensor(pn, device=dev) - 1)
                    pidx = (pq[:, 0] * pn[1] + pq[:, 1]) * pn[2] + pq[:, 2]   # each cell's parent on its grid
                    prow = torch.stack([prev[b][3] for b in bs])
                    pmf = torch.stack([prev[b][5] for b in bs])[:, pidx]
                    ok = prow[:, pidx]
                    if not dense and not (self.full and not self.training):   # reads what passes; full: all, masks
                        ok = ok & (pmf > self.tau)
                    if not dense and int(ok.sum(1).max()) > OPEN_MAX:   # the cap: the likeliest mixed parents
                        top = torch.where(ok, pmf, -1.0).topk(OPEN_MAX, 1).indices
                        ok = ok & torch.zeros_like(ok).scatter_(1, top, True)
                ti, ci = ok.nonzero(as_tuple=True)
                if not len(ti):
                    continue
                if j:
                    pr = pidx[ci]
                    PAR.append(torch.stack([prev[b][4] for b in bs])[ti, pr])
                    pinc = torch.stack([prev[b][6] for b in bs])[ti, pr]
                    INC.append(pinc if dense else pinc & (pmf[ti, ci][:, None] > taus))
                    allp = _patchify_b(F.pad(xs[bs].float(), pads), k)   # [G, N, P]
                    self.vox.patch_mm = mm
                    VOX.append(self.vox(allp[ti, ci], self.vox.weights(torch.tensor(sp, device=dev), list(k))))
                else:
                    INC.append(torch.ones(len(ti), nt, dtype=torch.bool, device=dev))
                if dense:   # the cached levels' tokens, else the frozen encoder now on the cells chosen
                    ENC.append(torch.stack([self._cached(ids[b], mm, c0, n, k, dev) for b, c0 in items])[ti, ci])
                else:
                    ENC.append(self._read_snake(allp, g, ok, n, torch.tensor(sp, device=dev), list(k), mm))
                if ys is not None and j in (1, 2):
                    yl = _patchify_b(F.pad(ys[bs].float(), pads, value=255), k)[ti, ci]
                    okl = yl != 255
                    lo, hi = torch.where(okl, yl, 1e9).amin(1), torch.where(okl, yl, -1.0).amax(1)
                    TG.append(torch.where(okl.any(1), (lo != hi).long(), -1))
                G_.append((bs, items, k, n, front, N, ti, ci))
            if not G_:
                break
            live = [b for gg in G_ for b in gg[0]]
            e = self.enc(F.dropout(torch.cat(ENC), DROP, self.training))
            inc = torch.cat(INC)
            if j == 0:
                t = d = e
            else:
                p_ = torch.cat(PAR)
                d = self.mlp[j - 1](torch.cat([p_, e, torch.cat(VOX).to(p_.dtype)], -1))
                t = p_ + d
            pm = torch.zeros(len(t), device=dev)
            if j in (1, 2):   # P(mixed) of the 8 / 4 mm cells: what the next level reads
                lg = self.lhead[j - 1](t.detach()).float()[:, 0]
                pm = lg.sigmoid().detach()
                if ys is not None:
                    tg = torch.cat(TG)
                    v = tg >= 0
                    if v.any():
                        pos = (tg[v] == 1).float()
                        # class-balanced as the mean of the two classes' mean BCE (31/10: the weight pure / mixed was
                        # ~1e3-1e4 on microbleeds and landmarks; through the global gradient clipping it starved the
                        # segmentation: cmbpts collapsed to the background, fetal's first loss ~2.5e4)
                        bce = F.binary_cross_entropy_with_logits(lg[v], pos, reduction="none")
                        loss = loss + 0.5 * ((bce * pos).sum() / pos.sum().clamp(min=1)
                                             + (bce * (1 - pos)).sum() / (1 - pos).sum().clamp(min=1))
            f = self.fproj(d) if self.fnorm is None else self.fnorm(self.fproj(d))
            f = f * 0 if j in self.off else f
            o, st = grid_[j]   # the cells' span on the output's grid (stride st), the tile cropped from it
            at = 0
            for bs, items, k, n, front, N, ti, ci in G_:
                M, G = len(ti), len(bs)
                fg, tg_, pg, ig = f[at: at + M], t[at: at + M], pm[at: at + M], inc[at: at + M]
                at += M
                if j < len(LV) - 1:   # the next level's parents: dense per tile
                    row = torch.zeros(G, N, dtype=torch.bool, device=dev)
                    row[ti, ci] = True
                    td = tg_.new_zeros(G, N, tg_.shape[1])
                    td[ti, ci] = tg_
                    pd = pg.new_zeros(G, N)
                    pd[ti, ci] = pg
                    idn = ig.new_zeros(G, N, nt)
                    idn[ti, ci] = ig
                    for q, (b, c0) in enumerate(items):
                        prev[b] = (k, c0, n, row[q], td[q], pd[q], idn[q])
                # per threshold the cells it keeps (a cell not kept: detail 0), all tiles and thresholds in one interpolation
                grid = fg.new_zeros(G, N, nt, C)
                grid[ti, ci] = fg[:, None, :] * ig[:, :, None].to(fg.dtype)
                up = F.interpolate(grid.permute(0, 2, 3, 1).reshape(G * nt, C, *n),
                                   size=[max(1, a * kk // s) for a, kk, s in zip(n, k, st)], mode="trilinear", align_corners=False)
                lo, z = [f_ // s for f_, s in zip(front, st)], sz[o]
                up = up[:, :, lo[0]: lo[0] + z[0], lo[1]: lo[1] + z[1], lo[2]: lo[2] + z[2]]
                up = up.reshape(G, nt, C, *up.shape[2:])   # (the cropped shape: 31/10, the uncropped one broke every run)
                for q, b in enumerate(bs):
                    field[o][b][:, :, : up.shape[3], : up.shape[4], : up.shape[5]] += up[q].to(fdt)
        out = [torch.stack(field[o]) for o in sorted(field)]
        return (out[0] if self.grid is None else out), loss

    @staticmethod
    @functools.cache
    def _grid_cpu(n: tuple) -> torch.Tensor:
        return torch.stack(torch.meshgrid(*[torch.arange(a) for a in n], indexing="ij"), -1).view(-1, 3)

    def _grid(self, n, dev) -> torch.Tensor:
        return self._grid_cpu(tuple(n)).to(dev)

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
                if grp and (i == len(qs) or tot + len(qs[i]) > LIVE_MAX):
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

    @torch.no_grad()
    def _read_snake(self, allp, g, ok, n, sp, k, mm) -> torch.Tensor:
        """The frozen encoder at level mm on a group's open cells (ok [G, N]; allp [G, N, P] all cells' patches, g [N, 3]
        their grid coordinates, one geometry), as stretches of the curve LIVE (v10): the open cells and the unread ones in
        gaps of <= SNAKE_GAP between them, one sequence per stretch, packed in calls of <= LIVE_MAX tokens and SEQ_MAX
        sequences -> the open cells' tokens in ok.nonzero() order."""
        from ..curves import keys
        pe, dev, N = self.encoder.patch_embed, g.device, len(g)
        order = keys(LIVE, g, torch.tensor(n, device=dev), self.seed).argsort()   # the cells along the curve
        om = ok[:, order]
        ar = torch.arange(N, device=dev)
        prv = torch.where(om, ar, -1).cummax(1).values                      # the last open place at or before
        nxt = torch.where(om, ar, N).flip(1).cummin(1).values.flip(1)       # the first open place at or after
        m = om | ((nxt - prv <= SNAKE_GAP + 1) & (prv >= 0) & (nxt < N))   # short gaps bridged
        gi, p = m.nonzero(as_tuple=True)                                      # places read, tile-major, along the curve
        brk = torch.ones(len(p), dtype=torch.bool, device=dev)
        brk[1:] = (gi[1:] != gi[:-1]) | (p[1:] - p[:-1] > 1)                 # a new stretch: next tile, or a long gap
        b0 = brk.nonzero()[:, 0].tolist() + [len(p)]
        lens = [b - a for a, b in itertools.pairwise(b0)]
        cells = order[p]
        mm0, pe.patch_mm = pe.patch_mm, mm
        hs, grp, tot, at = [], [], 0, 0
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = pe(allp[gi, cells], pe.weights(sp, k))
                q = g[cells]
                for i in range(len(lens) + 1):
                    if grp and (i == len(lens) or tot + lens[i] > LIVE_MAX or len(grp) >= SEQ_MAX):
                        cu = torch.tensor([0] + grp, device=dev).cumsum(0).to(torch.int32)
                        hs.append(self.encoder.flat(x[at: at + tot], cu, pos=q[at: at + tot]).float())
                        at, grp, tot = at + tot, [], 0
                    if i < len(lens):
                        grp.append(lens[i])
                        tot += lens[i]
        finally:
            pe.patch_mm = mm0
        h = torch.cat(hs)
        oe = om[gi, p]                                                        # the open places among those read
        return h[oe][(gi[oe] * N + cells[oe]).argsort()]


class HierUNet(HierDec):
    """args.hier_unet (user 30/10, "CubeDec ha fatto 76"): CubeDec's U-Net on the voxels (32 / 64 / 128 channels, nnU-Net
    strides per task, z-score per scan, deep supervision at 1/2 and 1/4) with HierDec's cascade as its token input in place
    of CubeDec's cubes / selector / gains: the 16 / 8 mm levels' field at the bottleneck, the 4 / 2 mm levels' (read under
    the mixed cells) at the 1/2 stage and at the voxels (30/10, v8: HierDec's field reached the voxels, best on HIE). Path
    dropout (v8): per training tile the image (e0's output) off with probability PDROP, else the field off with PDROP, so
    each path segments alone and the convs learn to use either. Token head (v9, user 30/10: "l'ensamble è la cosa più
    intelligente", HIE: every U-Net 0.13-0.29 on test, every token-only decoder 0.30-0.34): the field at the voxels (coarse
    levels upsampled + fine) -> GELU -> 1x1, its own loss; the prediction mixes the two heads' probabilities, the U-Net's
    weight alpha chosen on val among ALPHAS (segrun's threshold pass: TAUS / read_taus / full carry the alphas here; 0.5
    during training; landmarks: the U-Net head alone). The cascade reads by the training threshold (no full read)."""

    def __init__(self, encoder, cache: str, n_out: int, curves=(), seed: int = 17):
        super().__init__(encoder, cache, n_out, curves, seed)
        from .dense_dec import block
        del self.stem, self.head, self.head2, self.out   # no CK-convs
        self.TAUS, self.read_taus = ALPHAS, [0.5]   # segrun's threshold pass chooses the heads' mix (alpha) here
        # no per-cell LayerNorm on the field (30/10: every read cell at the same norm, small details amplified, slow start;
        # the blocks' InstanceNorm already sets the field's scale per tile)
        ch = (32, 64, 128)
        self.e0, self.e1, self.e2 = block(1, ch[0]), block(ch[0], ch[1], 2), block(ch[1], ch[2], 2)
        self.mid = block(ch[2] + C, ch[2])
        self.u1, self.d1 = nn.ConvTranspose3d(ch[2], ch[1], 2, 2), block(2 * ch[1] + C, ch[1])
        self.u0, self.d0 = nn.ConvTranspose3d(ch[1], ch[0], 2, 2), block(2 * ch[0] + C, ch[0])
        self.out = nn.Conv3d(ch[0], n_out, 1)
        self.ds = nn.ModuleList([nn.Conv3d(ch[1], n_out, 1), nn.Conv3d(ch[2], n_out, 1)])
        self.tok_out = nn.Conv3d(C, n_out, 1)   # the token head
        self.cum = {1: [2, 2, 2], 2: [4, 4, 4]}
        self.vol_stats = {}   # segrun: z-score per scan

    def plan(self, ids) -> None:
        """nnU-Net's strides (CubeDec.plan): an axis halves only while finer than twice the finest one, on the training
        scans' median spacing; the levels' field: 16 / 8 mm at the bottleneck, 4 / 2 mm at the 1/2 stage."""
        import numpy as np
        ids = [i for i in ids if (self.cache.root / f"{i}.json").exists()]
        if not ids:
            return
        cur, cum = np.median([self.cache.meta(i)["spacing"] for i in ids], 0).tolist(), [1, 1, 1]
        for s in (1, 2):
            st = [2 if c < 2 * min(cur) else 1 for c in cur]
            cur, cum = [c * a for c, a in zip(cur, st)], [c * a for c, a in zip(cum, st)]
            self.cum[s] = cum
        self.e1[0].stride, self.e2[0].stride = tuple(self.cum[1]), tuple(b // a for a, b in zip(self.cum[1], self.cum[2]))
        self.grid = [(0, tuple(self.cum[2]))] * 2 + [(1, (1, 1, 1))] * 2
        print(f"[seg] hier_unet plan: strides {self.cum[1]} / {self.cum[2]}", flush=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        from .dense_dec import CubeDec
        from .segrun import loss_fn
        if self.grid is None:   # no plan (no training scan in the cache yet): CubeDec's default strides
            self.grid = [(0, tuple(self.cum[2]))] * 2 + [(1, (1, 1, 1))] * 2
        rows = ctx.tolist()
        if self.vol_stats:
            ms = torch.tensor([self.vol_stats[int(r[6])] for r in rows], device=x.device, dtype=x.dtype)
            x = (x - ms[:, 0, None, None, None, None]) / ms[:, 1, None, None, None, None]
        T0 = tuple(x.shape[2:])
        x = F.pad(x, [p for t, m in zip(reversed(T0), reversed(self.cum[2])) for p in (0, -t % m)])
        ys = None
        if self.sel_y is not None and self.training:
            ys = F.pad(self.sel_y.to(x.device).reshape(len(x), *T0).long(),
                       [p for t, m in zip(reversed(T0), reversed(self.cum[2])) for p in (0, -t % m)], value=255)
        full, self.full = self.full, False   # (full: the alphas' pass; the cascade reads by the training threshold)
        try:
            (fc, ff), aux = self.fields(x[:, 0], [tuple(r[:3]) for r in rows], [self.cache.ids[int(r[6])] for r in rows],
                                        [[int(v) for v in r[7:10]] for r in rows], ys)
        finally:
            self.full = full
        T = tuple(x.shape[2:])
        lm = getattr(self, "landmark", False)
        f0 = self.e0(x)
        fc, ff = fc[:, 0].to(f0.dtype), ff[:, 0].to(f0.dtype)
        # the token head in the autocast dtype (30/10: in fp32 at the voxels, B 8 x 160^3, out of memory)
        tok = None if lm else self.tok_out(F.gelu(F.interpolate(fc, size=T, mode="trilinear", align_corners=False) + ff))
        if self.training and PDROP:   # path dropout: u < PDROP image off, PDROP <= u < 2 PDROP field off
            u = torch.rand(len(x), device=x.device)[:, None, None, None, None]
            f0 = f0 * (u >= PDROP).to(f0.dtype)
            on = ((u < PDROP) | (u >= 2 * PDROP)).to(f0.dtype)
            fc, ff = fc * on, ff * on
        f1 = self.e1(f0)
        hm = self.mid(torch.cat([self.e2(f1), fc], 1))
        st = [b // a for a, b in zip(self.cum[1], self.cum[2])]
        h1 = self.d1(torch.cat([CubeDec._up(self.u1, hm, st), f1, F.avg_pool3d(ff, self.cum[1], self.cum[1])], 1))
        if ys is not None:
            if not lm:   # deep supervision, nnU-Net's weights 0.5, 0.25; the token head's own loss
                for w, hd, head, s in ((0.5, h1, self.ds[0], self.cum[1]), (0.25, hm, self.ds[1], self.cum[2])):
                    aux = aux + w * loss_fn(head(hd).float(), _down_fg(ys, s, self.out.out_channels), self.out.out_channels)
                aux = aux + loss_fn(tok, ys, self.out.out_channels)
            self.aux, self.sel_y = aux, None
        h = self.d0(torch.cat([CubeDec._up(self.u0, h1, self.cum[1]), f0, ff], 1))
        o = self.out(h)
        if lm or self.training:   # training: segrun's loss on the U-Net head (the token head's is in aux)
            return o[:, :, : T0[0], : T0[1], : T0[2]]
        pu, pt = o.float().softmax(1), tok.float().softmax(1)
        al = self.read_taus if self.full else [0.5]
        out = [(a * pu + (1 - a) * pt).clamp(min=1e-8).log() for a in al]   # log-probabilities per alpha, side by side
        return torch.cat(out, 1)[:, :, : T0[0], : T0[1], : T0[2]]
