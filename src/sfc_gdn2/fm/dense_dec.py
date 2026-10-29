"""Dense control decoder (user, 28/10: is Dmix's cascade what loses the fine tokens' advantage?): no hard decision
anywhere. The frozen encoder's cached tokens at 16 and 8 mm (the Dmix token cache: last ++ the depth quartiles, the
curves' mean) are layer-normed, projected per level and spread trilinearly onto the tile at 1/4 of its voxels, where they
join a convolutional image path (UNETR / U-Net style: a full-resolution stem on the raw voxels, two strided stages);
the decoder climbs back to the full resolution with the image skips. Output: logits [B, k, *T] (segrun's generic path:
loss_fn, sliding, landmark_step). args.dense_dec=true in seg_net. ponytail: no 4 mm tokens (live reading of every tile
cost ~10x the step; add a 4 mm cache level if the control says the tokens matter).
Variant B (args.dense_mg=<folder>, user 28/10: "the decoder could use the tokens as the probe does"): + the probe's
multigrain 4 / 2 mm levels (fm/sfc.multigrain: cells under the most varied ones of the level above, label-free, a fixed
budget), precomputed once per volume (python -m sfc_gdn2.fm.dense_dec <task> <folder> <checkpoint>), each 1/4-grid voxel
taking its cell's projected token where that cell was read, zero elsewhere, with a read / not-read mask channel.
Sub-cell decoding (args.dense_sub=true, user 28/10: vs's small tumours under-segmented, trilinear treats a token as a
point sample at its cell's centre and smears it over the neighbours): each 8 mm token is decoded into its own 4^3 sub-cells
of 2 mm (a linear map, as a ViT's patch decoder / pixel shuffle), only then interpolated onto the 1/4 voxel grid.
No tokens (args.dense_notok=true, user 28/10: how much of the dense decoder is the U-Net alone?): the image path only."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .token_cache import TokenCache, fine_ok

LEVELS = (16.0, 8.0)
MG = (4.0, 2.0)     # variant B's sparse levels
SUB = 4             # dense_sub: sub-cells per axis of an 8 mm token (2 mm)
CH = (16, 32, 64)   # full, 1/2, 1/4 resolution
TOK = 64            # channels of each token level after its projection


def block(a: int, b: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(a, b, 3, stride, 1), nn.InstanceNorm3d(b, affine=True), nn.LeakyReLU(0.01, True),
                         nn.Conv3d(b, b, 3, 1, 1), nn.InstanceNorm3d(b, affine=True), nn.LeakyReLU(0.01, True))


class DenseDec(nn.Module):
    """`encoder` frozen and unused in forward (its tokens come from `cache`); ctx rows: spacing, scan shape, volume
    index, the tile's corner voxel (segrun appends it to the training tiles and, tile_corner, to sliding's)."""

    tile_corner = True

    def __init__(self, encoder, cache: str, n_out: int, mg: str | None = None, width: float = 1.0, dense4: bool = False,
                 sub: bool = False, notok: bool = False):
        super().__init__()
        global CH, TOK   # ponytail: module-level widths scaled per run (one decoder per process); args.dense_width
        CH, TOK = tuple(int(c * width) for c in (16, 32, 64)), int(64 * width)
        self.encoder = encoder
        self.cache = TokenCache(cache)
        self.bott = [4, 4, 4]   # voxels per bottleneck voxel and axis (CubeDec.plan: anisotropic)
        self.levels = () if notok else (*LEVELS, 4.0) if dense4 else LEVELS   # simpleC: + 4 mm dense (--dense4)
        self.cache.levels = self.levels
        c = encoder.dim * (1 + len(encoder.taps))
        self.tok = nn.ModuleList(nn.Sequential(nn.LayerNorm(c), nn.Linear(c, TOK)) for _ in self.levels)
        self.unpatch = nn.Linear(TOK, TOK * SUB ** 3) if sub else None   # the 8 mm level's sub-cells
        self.e0, self.e1, self.e2 = block(1, CH[0]), block(CH[0], CH[1], 2), block(CH[1], CH[2], 2)
        self.mg, self._mg = mg, {}
        if mg:
            self.tok_mg = nn.ModuleList(nn.Sequential(nn.LayerNorm(c), nn.Linear(c, TOK)) for _ in MG)
        self.mid = block(CH[2] + (TOK + 1) * len(MG) * bool(mg) + TOK * len(self.levels), CH[2])
        self.u1, self.d1 = nn.ConvTranspose3d(CH[2], CH[1], 2, 2), block(2 * CH[1], CH[1])
        self.u0, self.d0 = nn.ConvTranspose3d(CH[1], CH[0], 2, 2), block(2 * CH[0], CH[0])
        self.out = nn.Conv3d(CH[0], n_out, 1)

    def fields(self, ctx: torch.Tensor, T, dev) -> torch.Tensor:
        """Each level's projected tokens over each tile, trilinear on the tile's 1/4 grid -> [B, TOK x levels, *T/4]."""
        out, b = [], self.bott
        T4 = [t // a for t, a in zip(T, b)]
        for r in ctx.tolist():
            i, corner = self.cache.ids[int(r[6])], [int(v) for v in r[7:10]]
            k4, lv = self.cache.meta(i)["k4"], []
            for j, mm in enumerate(self.levels):
                k = [a * int(mm // 4) for a in k4]
                if mm < 8 and not fine_ok(k):   # a scan too coarse for 4 mm cells (no file): an empty level
                    lv.append(torch.zeros(1, TOK, *T4, device=dev))
                    continue
                c0 = [c // kk for c, kk in zip(corner, k)]
                n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
                t = self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).to(dev).float().reshape(*n, -1)
                g = self.tok[j](t)                                                 # [*n, TOK] on the cell grid
                if self.unpatch is not None and mm == 8.0:   # [*n, S, S, S, TOK] -> the sub-cell grid [*n * S, TOK]
                    g = self.unpatch(g).view(*n, SUB, SUB, SUB, TOK).permute(0, 3, 1, 4, 2, 5, 6)
                    g = g.reshape(*[a * SUB for a in n], TOK)
                g = g.permute(3, 0, 1, 2)[None]                                    # [1, TOK, *grid]
                # cell centres -> voxels: the cells' grid spans n*k voxels from c0*k; the tile is a crop of it
                full = [a * kk for a, kk in zip(n, k)]
                g = F.interpolate(g, size=[max(1, f // a) for f, a in zip(full, b)], mode="trilinear", align_corners=False)
                o = [(c - a * kk) // bb for c, a, kk, bb in zip(corner, c0, k, b)]
                g = g[:, :, o[0]: o[0] + T4[0], o[1]: o[1] + T4[1], o[2]: o[2] + T4[2]]
                g = F.pad(g, [p for d, s in zip(reversed(g.shape[2:]), reversed(T4)) for p in (0, s - d)])
                lv.append(g)
            if self.mg:
                lv += self.sparse(i, corner, T, dev)
            out.append(torch.cat(lv, 1))
        return torch.cat(out)

    def sparse(self, i, corner, T, dev) -> list:
        """Variant B: per multigrain level, each 1/4-grid voxel of the tile takes the projected token of the cell it lies
        in if that cell was read (zero otherwise) + the read mask -> [1, TOK + 1, *T/4] per level."""
        from pathlib import Path
        if i not in self._mg:
            self._mg = dict(list(self._mg.items())[-15:]) | {i: torch.load(Path(self.mg) / f"{i}.pt", weights_only=False)}
        lv, n4 = self._mg[i], [t // 4 for t in T]
        pos = [torch.arange(n, device=dev) * 4 + 2 + c for n, c in zip(n4, corner)]   # 1/4-grid voxel centres
        out = []
        for j, mm in enumerate(MG):
            if mm not in lv:   # a level the scan cannot hold (fm/sfc.multigrain): empty
                out.append(torch.zeros(1, TOK + 1, *n4, device=dev))
                continue
            q, h, k = lv[mm]
            q = q.to(dev).long()
            cell = [p // kk for p, kk in zip(pos, k)]                    # each axis' cell index
            lo = [int(c[0]) for c in cell]
            n = [int(c[-1]) - a + 1 for c, a in zip(cell, lo)]
            ok = ((q >= torch.tensor(lo, device=dev)) & (q < torch.tensor(lo, device=dev) + torch.tensor(n, device=dev))).all(1)
            idx = torch.full(n, -1, dtype=torch.long, device=dev)
            qi = q[ok] - torch.tensor(lo, device=dev)
            idx[qi[:, 0], qi[:, 1], qi[:, 2]] = torch.arange(int(ok.sum()), device=dev)
            g = idx[(cell[0] - lo[0])[:, None, None], (cell[1] - lo[1])[None, :, None], (cell[2] - lo[2])[None, None, :]]
            feat = self.tok_mg[j](h[ok.cpu()].to(dev).float()) if ok.any() else torch.zeros(0, TOK, device=dev)
            f = torch.cat([feat, feat.new_ones(len(feat), 1)], 1)       # + the read mask
            f = torch.cat([f.new_zeros(1, TOK + 1), f])[g + 1]           # -1 (not read) -> the zero row
            out.append(f.permute(3, 0, 1, 2)[None])
        return out

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        T0 = tuple(x.shape[2:])
        x = F.pad(x, [p for t in reversed(T0) for p in (0, -t % 4)])   # 1/4 grid: multiples of 4
        T = tuple(x.shape[2:])
        f0 = self.e0(x)
        f1 = self.e1(f0)
        f2 = self.e2(f1)
        if self.levels or self.mg:
            tk = self.fields(ctx, T, x.device)   # the projections train; the cached tokens are data
            f2 = torch.cat([f2, tk.to(f2.dtype)], 1)
        h = self.mid(f2)
        h = self.d1(torch.cat([self.u1(h), f1], 1))
        h = self.d0(torch.cat([self.u0(h), f0], 1))
        return self.out(h)[:, :, : T0[0], : T0[1], : T0[2]]

    @staticmethod
    def tile_max(n_out: int) -> float:
        """segrun.TILE_MAX for this decoder: tiles of at most ~160^3 voxels (its activations are dense)."""
        return float(160 ** 3 * n_out)


NCH = (16, 32, 64, 64, 64)                              # args.dense_native: stages at 1, 2, 4, 8, 16 voxels (~mm)
NTOK = {16.0: 32, 8.0: 32, 4.0: 16, 2.0: 8}             # channels per token level: 4 / 2 mm overcomplete, cut to image density
NSTAGE = {16.0: 4, 8.0: 3, 4.0: 2, 2.0: 1}
# args.dense_cubes: the fine levels read in cubes of the pretraining view (16^3 cells: 6.4 cm at 4 mm, 3.2 cm at 2 mm) where
# the decoder itself sees foreground; one rule for every task (user 28/10: nothing per dataset)
CUBE, CAP, TAU, DROP = 16, 8, 0.1, 0.3
CUBE_RAM_GB = float(__import__("os").environ.get("SFC_CUBE_RAM_GB", "60"))   # the cube cache in RAM (a workstation: ~4)   # cells per axis, cubes per volume and level, min foreground prob, cubes dropped in training


def gather(g: torch.Tensor, c0, k, corner, T, s) -> torch.Tensor:
    """g [*n, C] on the cell grid from cell c0 (cells of k voxels) -> [1, C, *T / stride]: each voxel of stage s (an int:
    stride 2^s on every axis; a list: the stride per axis) takes the cell its centre lies in (nearest, no interpolation:
    a token is a cell, not a point sample)."""
    sts, dev = [2 ** s] * 3 if isinstance(s, int) else list(s), g.device
    idx = [((torch.arange(t // st, device=dev) * st + st // 2 + c) // kk - a).clamp(0, n - 1)
           for t, c, kk, a, n, st in zip(T, corner, k, c0, g.shape[:3], sts)]
    return g[idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]].permute(3, 0, 1, 2)[None]


class NativeDec(nn.Module):
    """args.dense_native=true (user 28/10: "keep simpleA small, so it has to use the tokens"): a small 5-stage U-Net on the
    image (1, 2, 4, 8, 16 voxels, NCH channels), each token level concatenated, projected, at the stage of its own
    resolution (16 mm -> stage 4, 8 mm -> 3; with args.dense_mg the multigrain 4 / 2 mm, sparse + read mask, -> 2 / 1),
    so the skips carry them up; args.dense_notok: the same U-Net without tokens (its control).
    args.dense_cubes (user 28/10): the 4 / 2 mm levels in cubes placed by the decoder, in the budget of the others: up to the
    first validation 16 / 8 mm only; after it (on_val) the model predicts every training scan, cubes go greedily on its
    foreground probability (TAU, CAP) and the frozen encoder reads them as pretraining views; 4 mm from the first validation,
    2 mm from the second (`stage`, a buffer: the best state carries it). Val / test (`prepare`, from segrun.sliding): the
    same passes on the scan, coarse to fine, no label. Training drops DROP of the cubes (the model meets unread cells)."""

    tile_corner = True

    def __init__(self, encoder, cache: str, n_out: int, mg: str | None = None, notok: bool = False, cubes: bool = False,
                 curves=(), seed: int = 17):
        super().__init__()
        self.encoder, self.cache = encoder, TokenCache(cache)
        self.dense = () if notok else LEVELS
        self.cache.levels = self.dense or LEVELS
        self.mg, self._mg = (None if notok else mg), {}
        self.cube, self.curves, self.seed = cubes and not notok, list(curves), seed
        self.cubes, self.active, self.preparing = {}, None, False   # {volume index: {mm: (cells, tokens, k, cube)}}
        self.cube_dir = None   # segrun: <out>.cubes, the training scans' cubes per level (<mm>.pt), reloaded on resume
        self.register_buffer("stage", torch.zeros((), dtype=torch.long))
        lv = (*self.dense, *(MG if self.mg or self.cube else ()))
        c = encoder.dim * (1 + len(encoder.taps))
        self.tok = nn.ModuleDict({f"g{mm:g}": nn.Sequential(nn.LayerNorm(c), nn.Linear(c, NTOK[mm])) for mm in lv})
        extra = [0] * len(NCH)
        for mm in lv:
            extra[NSTAGE[mm]] += NTOK[mm] + (mm in MG)   # + the read mask
        conv = lambda a, b, st=1: nn.Sequential(nn.Conv3d(a, b, 3, st, 1), nn.InstanceNorm3d(b, affine=True),
                                                nn.LeakyReLU(0.01, True))
        self.down = nn.ModuleList(conv(1 if s == 0 else NCH[s - 1], NCH[s], 1 if s == 0 else 2) for s in range(len(NCH)))
        self.body = nn.ModuleList(conv(NCH[s] + extra[s], NCH[s]) for s in range(len(NCH)))
        self.up = nn.ModuleList(nn.ConvTranspose3d(NCH[s], NCH[s - 1], 2, 2) for s in range(1, len(NCH)))
        self.dec = nn.ModuleList(nn.Sequential(conv(2 * NCH[s - 1], NCH[s - 1]), conv(NCH[s - 1], NCH[s - 1]))
                                 for s in range(1, len(NCH)))
        self.out = nn.Conv3d(NCH[0], n_out, 1)

    def tokens(self, ctx: torch.Tensor, T, dev) -> dict:
        """{stage: [B, channels, *T / 2^stage]} of the tile's token levels."""
        per = {}
        for r in ctx.tolist():
            i, corner = self.cache.ids[int(r[6])], [int(v) for v in r[7:10]]
            k4 = self.cache.meta(i)["k4"]
            for mm in self.dense:
                k = [a * int(mm // 4) for a in k4]
                c0 = [c // kk for c, kk in zip(corner, k)]
                n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
                t = self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).to(dev).float().reshape(*n, -1)
                per.setdefault(NSTAGE[mm], []).append(gather(self.tok[f"g{mm:g}"](t), c0, k, corner, T, NSTAGE[mm]))
            src = self._mg_of(i) if self.mg else self._cubes_of(int(r[6])) if self.cube else None
            if src is not None:
                for mm, f in self.sparse(src, corner, T, dev).items():
                    per.setdefault(NSTAGE[mm], []).append(f)
        return {s: torch.cat(v) for s, v in per.items()}   # one level per stage: one entry per tile

    def _mg_of(self, i) -> dict:
        from pathlib import Path
        if i not in self._mg:
            self._mg = dict(list(self._mg.items())[-15:]) | {i: torch.load(Path(self.mg) / f"{i}.pt", weights_only=False)}
        return self._mg[i]

    def _cubes_of(self, vid: int) -> dict:
        """The volume's cube cells of the active levels (training: DROP of its cubes left out)."""
        a, out = int(self.stage) if self.active is None else self.active, {}
        for mm in MG[:a]:
            if mm in self.cubes.get(vid, {}):
                q, h, k, cid = self.cubes[vid][mm]
                if self.training:
                    keep = (torch.rand(int(cid.max()) + 1) >= DROP)[cid]
                    q, h = q[keep], h[keep]
                out[mm] = (q, h, k)
        return out

    def sparse(self, src: dict, corner, T, dev) -> dict:
        """The 4 / 2 mm levels at their stages (src {mm: (cells, tokens, k)}): the read cells' projected tokens, zero
        elsewhere, + the mask."""
        out = {}
        for mm in MG:
            s, w = self._stride(mm), NTOK[mm] + 1
            if mm not in src:
                out[mm] = torch.zeros(1, w, *[t // a for t, a in zip(T, s)], device=dev)
                continue
            q, h, k = src[mm]
            q = q.to(dev).long()
            c0 = [c // kk for c, kk in zip(corner, k)]
            n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
            lo, hi = torch.tensor(c0, device=dev), torch.tensor(c0, device=dev) + torch.tensor(n, device=dev)
            ok = ((q >= lo) & (q < hi)).all(1)
            g = torch.zeros(*n, w, device=dev)
            if ok.any():
                f = self._proj(mm)(h[ok.cpu()].to(dev).float())
                qi = q[ok] - lo
                g[qi[:, 0], qi[:, 1], qi[:, 2]] = torch.cat([f, f.new_ones(len(f), 1)], 1).float()
            out[mm] = gather(g, c0, k, corner, T, s)
        return out

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        T0, m = tuple(x.shape[2:]), 2 ** (len(NCH) - 1)
        x = F.pad(x, [p for t in reversed(T0) for p in (0, -t % m)])   # stage 4: multiples of 16
        T = tuple(x.shape[2:])
        tk = self.tokens(ctx, T, x.device) if self.dense or self.mg else {}
        skips, h = [], x
        for s in range(len(NCH)):
            h = self.down[s](h)
            h = self.body[s](torch.cat([h, tk[s].to(h.dtype)], 1) if s in tk else h)
            skips.append(h)
        for s in range(len(NCH) - 1, 0, -1):
            h = self.dec[s - 1](torch.cat([self.up[s - 1](h), skips[s - 1]], 1))
        return self.out(h)[:, :, : T0[0], : T0[1], : T0[2]]

    tile_max = staticmethod(DenseDec.tile_max)

    def _proj(self, mm: float) -> nn.Module:
        return self.tok[f"g{mm:g}"]

    @staticmethod
    def _stride(mm: float) -> list:
        return [2 ** NSTAGE[mm]] * 3

    @staticmethod
    def _cell(sp, mm: float) -> list:
        from ..data.volume import patch_dims
        return list(patch_dims(sp, mm))

    @torch.no_grad()
    def place(self, x: torch.Tensor, ctx: torch.Tensor, level: int, predict) -> None:
        """Level `level` (1: 4 mm, 2: 2 mm) of the scan x [1, *S] (ctx: spacing, shape, volume index): predict() with the
        levels above, cubes greedily on the foreground probability (each cube's cells excluded from the next centres),
        each cube read by the frozen encoder as one view (box-local curve keys, the curves averaged)."""
        from ..curves import grid_coords, keys
        from ..data.volume import patchify
        vid, mm = int(ctx[6]), MG[level - 1]
        self.active, self.preparing = level - 1, True
        try:
            p = predict()
        finally:
            self.active, self.preparing = None, False
        dev, sp = p.device, ctx[:3].float()
        k = self._cell(sp.tolist(), mm)
        self.cubes.setdefault(vid, {}).pop(mm, None)
        if not fine_ok(k) or mm in getattr(self, "off", ()):
            return
        sc = F.max_pool3d(p[1:].float().amax(0)[None, None], k, k, ceil_mode=True)[0, 0]   # per cell: max foreground
        g, boxes = sc.shape, []
        while len(boxes) < CAP and float(sc.max()) >= TAU:
            c = [int(a) for a in torch.unravel_index(sc.argmax(), g)]
            lo = [max(0, min(a - CUBE // 2, n - CUBE)) for a, n in zip(c, g)]
            hi = [min(n, a + CUBE) for a, n in zip(lo, g)]
            sc[lo[0]: hi[0], lo[1]: hi[1], lo[2]: hi[2]] = -1
            boxes.append((torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)))
        if not boxes:
            return
        pt = patchify(x[0].to(dev).float(), k)
        xyz = grid_coords(torch.tensor(g, device=dev)[None], len(pt))[0]
        seen, qs, hs, cs = torch.zeros(len(pt), dtype=torch.bool, device=dev), [], [], []
        pe = self.encoder.patch_embed
        mm0, pe.patch_mm = pe.patch_mm, mm
        try:
            wt = pe.weights(sp.to(dev), k)
            for j, (lo, hi) in enumerate(boxes):
                sel = ((xyz >= lo) & (xyz < hi)).all(1).nonzero()[:, 0]
                q, h = xyz[sel], 0
                cu = torch.tensor([0, len(sel)], dtype=torch.int32, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xe = pe(pt[sel], wt)
                    for c in self.curves:
                        o = keys(c, q - lo, hi - lo, self.seed).argsort()
                        h = h + self.encoder.flat(xe[o], cu, pos=q[o]).float()[o.argsort()]
                new = ~seen[sel]
                seen[sel] = True
                qs.append(q[new].short().cpu())
                hs.append((h[new] / len(self.curves)).half().cpu())
                cs.append(torch.full((int(new.sum()),), j))
        finally:
            pe.patch_mm = mm0
        self.cubes[vid][mm] = (torch.cat(qs), torch.cat(hs), list(k), torch.cat(cs))

    def prepare(self, x: torch.Tensor, ctx: torch.Tensor, predict) -> None:
        """Val / test (segrun.sliding, before its pass): the scan's cubes of every level the state has, coarse to fine."""
        for level in range(1, int(self.stage) + 1):
            self.place(x, ctx, level, predict)

    def on_val(self, it: int, every: int, scans, predict, resume: bool = False) -> None:
        """segrun, after the validation at `it`: the first two raise the stage and place that level on every training scan
        (scans: (x, ctx); predict(x, ctx) -> probabilities). resume: a resumed run (debug chunks) places again every level
        its state has, with the resumed model (the cubes live in RAM only)."""
        n = int(self.stage) if resume else it // every
        if not self.cube or n > len(MG) or (not resume and n <= int(self.stage)):
            return
        t0 = __import__("time").time()
        self.stage.fill_(n)
        for level in range(1 if resume else n, n + 1):
            mm = MG[level - 1]
            f = None if self.cube_dir is None else self.cube_dir / f"{mm:g}.pt"
            if resume and f is not None and f.exists():   # the cubes placed when the level started, as they were
                for vid, v in torch.load(f, weights_only=False).items():
                    self.cubes.setdefault(vid, {})[mm] = v
                how = "reloaded"
            else:
                for x, c in scans:
                    self.place(x, c, level, lambda x=x, c=c: predict(x, c))
                if f is not None:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    ids = {int(c[6]) for _, c in scans}
                    torch.save({vid: v[mm] for vid, v in self.cubes.items() if vid in ids and mm in v}, f.with_suffix(".tmp"))
                    f.with_suffix(".tmp").replace(f)
                how = "placed again on resume" if resume else f"placed from it {it}"
            cells = sum(len(v[mm][0]) for v in self.cubes.values() if mm in v)
            print(f"[seg] cubes: {mm:g} mm {how} on {len(scans)} training scans, {cells} cells "
                  f"({__import__('time').time() - t0:.0f} s)", flush=True)


@torch.no_grad()
def main() -> None:
    """Variant B's tokens: python -m sfc_gdn2.fm.dense_dec <task> <out folder> <checkpoint> -> <out>/<id>.pt = {mm: (cells
    [n, 3] int16, tokens [n, C] fp16, cell voxels)} for the 4 / 2 mm multigrain levels of every volume of the task (the
    segdec inputs: the decoder's own voxel grid), read as the probe's extraction reads them."""
    import sys
    from pathlib import Path

    import pandas as pd

    from ..bench.segdec import task_items
    from ..io import load_yaml
    from .api import load_config
    from .sfc import SFC, multigrain
    task, out, ck = sys.argv[1:4]
    dense4 = sys.argv[4:5] == ["--dense4"]   # simpleC: <out> = the Dmix cache's 16 / 8 mm (links) + 4 mm dense, written here
    MV = Path("/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2")
    SI = Path("/leonardo_scratch/fast/IscrC_SFMRI/fcorrent/segdec_inputs/sfc_gdn2")
    vols = pd.read_csv(MV / "volumes.csv", low_memory=False, dtype={"session": str})
    tasks = load_yaml("configs/bench/probe_tasks_v2.yaml")
    items = task_items({task: tasks[task]}, vols, pd.read_csv(MV / "splits.csv"), 0)[task]
    cfg = load_config("configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"] = ck
    w = SFC(cfg, "cuda")
    w.extractor()
    enc, seed, thr = w._extract_enc, int(w.run_cfg["seed"]), w.run_cfg["objective"].get("fg_threshold", 0.05)
    Path(out).mkdir(parents=True, exist_ok=True)
    if dense4:
        from .token_cache import write_volume
        src = Path("/leonardo_scratch/fast/IscrC_SFMRI/fcorrent/tok/bench_p28") / task
        enc.taps = (3, 7, 11)   # the Dmix cache's columns: last ++ the depth quartiles (args.taps 3)
        for g in src.iterdir():
            if not (Path(out) / g.name).exists():
                (Path(out) / g.name).symlink_to(g)
    for n, it in enumerate(items):
        f = Path(out) / f"{it['id']}.pt"
        if dense4:
            if (Path(out) / f"{it['id']}.4.mean.npy").exists() or not (SI / f"{it['id']}.pt").exists():
                continue
            r = torch.load(SI / f"{it['id']}.pt", map_location="cpu", weights_only=False)
            write_volume(enc, w.curves, seed, r["x"].float(), r["ctx"][:3], Path(out), it["id"], {}, 4.0)
            continue
        if f.exists() or not (SI / f"{it['id']}.pt").exists():
            continue
        r = torch.load(SI / f"{it['id']}.pt", map_location="cpu", weights_only=False)
        keep = {}
        multigrain(enc, w.curves, seed, r["x"][0].cuda().float(), r["ctx"][:3].float().cuda(), thr, {}, keep)
        torch.save({mm: v for mm, v in keep.items() if mm in MG}, f)
        if n % 50 == 0:
            print(f"[dense_mg] {task} {n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()


class CubeDec(DenseDec):
    """args.dense_cubes without dense_native (user 28/10: "keep simpleA, 3 stages"): simpleA (16 / 8 mm at the 1/4
    bottleneck) + NativeDec's cubes, the 2 mm ones at the 1/2 stage (8 channels + mask: into the encoder path and its skip),
    the 4 mm ones at the bottleneck (16 + mask). A cube's cell is aligned to its stage: per axis the stage's stride times
    round(grain / (stride x spacing)), at least one stride (fetal 0.6 mm: 2.4 and 4.8 mm cells, 2 stage voxels each)."""

    sparse = NativeDec.sparse
    tile_max = staticmethod(DenseDec.tile_max)

    def __init__(self, encoder, cache: str, n_out: int, curves=(), seed: int = 17):
        live = str(cache).startswith("live:")   # a task without a cache yet: written as segrun loads each scan
        super().__init__(encoder, str(cache).removeprefix("live:"), n_out, width=2.0)   # 32 / 64 / 128 channels (29/10)
        if live:
            from .token_cache import LiveCache
            self.cache, self.enc_curves = LiveCache(str(cache)[5:], self), list(curves)
            self.cache.levels = self.levels
        c = encoder.dim * (1 + len(encoder.taps))
        self.ctok = nn.ModuleDict({f"g{mm:g}": nn.Sequential(nn.LayerNorm(c), nn.Linear(c, NTOK[mm])) for mm in MG})
        w2, w4 = NTOK[2.0] + 1, NTOK[4.0] + 1
        self.e2 = block(CH[1] + w2, CH[2], 2)
        self.mid = block(CH[2] + TOK * len(self.levels) + w4, CH[2])
        self.d1 = block(2 * CH[1] + w2, CH[1])
        self.cube, self.curves, self.seed, self.mg = True, list(curves), seed, None
        self.cubes, self.active, self.preparing, self.cube_dir = {}, None, False, None
        self.register_buffer("stage", torch.full((), 2, dtype=torch.long))   # both fine levels from the start
        self.cum, self.off = {0: [1, 1, 1], 1: [2, 2, 2], 2: [4, 4, 4]}, set()   # plan(): per task
        # 29/10 (user): the fine inputs scaled by gains from 0 (the network opens them when they help: the schedule); the
        # cubes chosen by a selector = a linear probe on the 16 / 8 mm tokens (fit before training) + a correction on the
        # U-Net's bottleneck (from 0, detached), trained with BCE + Tversky on "the cell holds foreground"; the input
        # z-scored per scan (vol_stats, segrun.load), as every external model's preprocessing
        self.gain = nn.Parameter(torch.zeros(2))
        self.sel_ln = nn.LayerNorm(c, elementwise_affine=False)
        self.sel_probe = nn.Linear(2 * c, 1)
        self.sel_corr = nn.Conv3d(CH[2], 1, 1)
        self.ds = nn.ModuleList([nn.Conv3d(CH[1], n_out, 1), nn.Conv3d(CH[2], n_out, 1)])   # deep supervision, 1/2 and 1/4
        nn.init.zeros_(self.sel_corr.weight)
        nn.init.zeros_(self.sel_corr.bias)
        self.vol_stats, self.sel_y, self.aux, self.sel_mode = {}, None, 0.0, False
        self.sel_map, self.boxes = {}, {}   # training scans: the selector's map; (scan, level): the cubes chosen
        self.vols, self.ram, self.ram_bytes = {}, __import__("collections").OrderedDict(), 0   # scans; the cube cache

    def plan(self, ids) -> None:
        """The task's fingerprint (median spacing of the training scans, from the token cache) -> the 3 stages' strides per
        axis (nnU-Net's rule: an axis halves only while finer than twice the finest one) and the fine levels kept (off: a
        level whose cell on the median scan holds < 8 voxels, or the voxels of the level above). Isotropic scans: simpleA."""
        import numpy as np
        ids = [i for i in ids if (self.cache.root / f"{i}.json").exists()]
        if not ids:   # a live cache before its first scans: planned again once segrun has loaded them
            return
        sp = np.median([self.cache.meta(i)["spacing"] for i in ids], 0).tolist()
        cur, cum = list(sp), [1, 1, 1]
        for s in (1, 2):
            st = [2 if c < 2 * min(cur) else 1 for c in cur]
            cur, cum = [c * a for c, a in zip(cur, st)], [c * a for c, a in zip(cum, st)]
            self.cum[s] = cum
        self.e1[0].stride, self.e2[0].stride = tuple(self.cum[1]), tuple(b // a for a, b in zip(self.cum[1], self.cum[2]))
        self.bott = self.cum[2]
        cells = {mm: self._cell(sp, mm) for mm in MG}
        self.off = {mm for mm in MG if math.prod(cells[mm]) < 8} | ({2.0} if cells[2.0] == cells[4.0] else set())
        print(f"[seg] plan: median spacing {[round(z, 2) for z in sp]} mm, strides {self.cum[1]} / {self.cum[2]}, "
              f"cells {cells} voxels, levels off {sorted(self.off)}", flush=True)

    def _proj(self, mm: float) -> nn.Module:
        return self.ctok[f"g{mm:g}"]

    def _stride(self, mm: float) -> list:
        return self.cum[NSTAGE[mm]]

    def _cell(self, sp, mm: float) -> list:
        return [a * max(1, round(mm / (a * float(z)))) for a, z in zip(self._stride(mm), sp)]

    def _cells(self, i: str, mm: float, corner, T):
        """The cached tokens of level mm over the tile (corner, T voxels): [*n, C] from cell c0, cells of k voxels."""
        k = [a * int(mm // 4) for a in self.cache.meta(i)["k4"]]
        c0 = [c // kk for c, kk in zip(corner, k)]
        n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
        return self.cache.tokens(i, mm, [a * kk for a, kk in zip(c0, k)], n, k).float().reshape(*n, -1), c0, k

    def _probe_feats(self, i: str, corner, T):
        """Each 8 mm cell of the tile: its token ++ its 16 mm parent's, layer-normed -> [*n8, 2C], c0, k8."""
        t8, c0, k8 = self._cells(i, 8.0, corner, T)
        t16, d0, _ = self._cells(i, 16.0, corner, T)
        par = [((torch.arange(n) + a) // 2 - b).clamp(0, m - 1) for n, a, b, m in zip(t8.shape[:3], c0, d0, t16.shape[:3])]
        t16 = t16[par[0][:, None, None], par[1][None, :, None], par[2][None, None, :]]
        return torch.cat([self.sel_ln(t8), self.sel_ln(t16)], -1), c0, k8

    def _probe_data(self, scans, g, sub: bool = True):
        """(features, labels) of the 8 mm cells of scans (x, y, ctx); sub: every positive and up to 4x as many negatives."""
        X, Y = [], []
        for x, y, c in scans:
            i = self.cache.ids[int(c[6])]
            S = list(y.shape[-3:])
            f, _, k8 = self._probe_feats(i, [0, 0, 0], S)
            fg = ((y > 0) & (y < 255)).float().reshape(1, 1, *S)
            occ = F.max_pool3d(fg, k8, k8, ceil_mode=True)[0, 0][: f.shape[0], : f.shape[1], : f.shape[2]].flatten() > 0
            f = f.reshape(-1, f.shape[-1])[: len(occ)]
            neg = (~occ).nonzero()[:, 0]
            if sub:
                neg = neg[torch.randperm(len(neg), generator=g)[: max(4 * int(occ.sum()), 64)]]
            X.append(torch.cat([f[occ], f[neg]]).half())
            Y.append(torch.cat([torch.ones(int(occ.sum())), torch.zeros(len(neg))]))
        dev = self.sel_probe.weight.device if sub else "cpu"   # the val cells (all of them) stay on the CPU
        return torch.cat(X).to(dev), torch.cat(Y).to(dev)

    @torch.no_grad()
    def fit_selector(self, scans, val) -> None:
        """The selector's probe before training (user 29/10): logistic regression of "the 8 mm cell holds foreground" on
        the training scans (scans: (x, y, ctx); every positive cell, up to 4x as many negatives), L-BFGS, the L2 chosen on
        the val scans' cells (all of them) by the same class-weighted loss."""
        g = torch.Generator().manual_seed(self.seed)
        X, Y = self._probe_data(scans, g)
        Xv, Yv = self._probe_data(val, g, sub=False)
        pw = (Y == 0).sum() / (Y == 1).sum().clamp(min=1)
        pv = (Yv == 0).sum() / (Yv == 1).sum().clamp(min=1)
        best = None
        for l2 in (1e-2, 1.0, 1e2, 1e4):
            w = torch.zeros(X.shape[1] + 1, device=X.device, requires_grad=True)
            opt = torch.optim.LBFGS([w], max_iter=100, history_size=10, line_search_fn="strong_wolfe")

            def closure(opt=opt, w=w, l2=l2):
                opt.zero_grad()
                with torch.enable_grad():
                    z = X.float() @ w[:-1] + w[-1]
                    loss = F.binary_cross_entropy_with_logits(z, Y, pos_weight=pw) + l2 * w[:-1].square().mean()
                    loss.backward()
                return loss
            opt.step(closure)
            zv = torch.cat([(c.to(w.device).float() @ w[:-1] + w[-1]).cpu() for c in Xv.split(1 << 16)])
            lv = float(F.binary_cross_entropy_with_logits(zv, Yv, pos_weight=pv))
            hit = (zv > 0).float()
            print(f"[seg] selector probe l2 {l2:g}: val loss {lv:.4f}, recall {float(hit[Yv == 1].mean()):.3f}, "
                  f"false positives {float(hit[Yv == 0].mean()):.3f}", flush=True)
            if best is None or lv < best[0]:
                best = (lv, l2, w.detach().clone())
        self.sel_probe.weight.copy_(best[2][None, :-1])
        self.sel_probe.bias.copy_(best[2][-1:])
        print(f"[seg] selector probe: l2 {best[1]:g} on {len(Y)} training cells ({int(Y.sum())} foreground), "
              f"{len(Yv)} val cells", flush=True)

    # 29/10 (user: refresh cheaper): cubes on a fixed grid per scan and level (16^3 cells, stride 8), chosen greedily on
    # the selector's map at the bottleneck's resolution; the training scans' map is the selector's own output on the
    # training tiles (sel_map, EMA 0.5, initialised by the probe alone), so a refresh (every REFRESH iterations) runs no
    # network; a cube read once is kept on disk (cube_dir/read) and only cubes newly chosen are read by the encoder

    def _probe_map(self, x, ctx) -> torch.Tensor:
        """The probe's foreground probability of a whole scan on the bottleneck grid [*S / b] (CPU)."""
        S = [-(-s // a) * a for s, a in zip(x.shape[-3:], self.cum[2])]
        fe, c0, k8 = self._probe_feats(self.cache.ids[int(ctx[6])], [0, 0, 0], S)
        w = self.sel_probe.weight.device
        return gather(self.sel_probe(fe.to(w)), c0, k8, [0, 0, 0], S, self.cum[2])[0, 0].float().sigmoid().cpu()

    def _choose(self, ms: torch.Tensor, k, prev=()) -> list:
        """Cubes on the map ms [*S / b] (grid cubes of 16 cells of k voxels, stride 8): the previous ones kept while their
        region's max >= TAU / 2 (hysteresis: few changes), then greedily the highest ones >= TAU, < CAP in all, each
        region cleared -> [(lo, hi)] in cells."""
        b, ms, boxes = self.cum[2], ms.clone().float(), []
        K = [min(CUBE * kk // bb, n) for kk, bb, n in zip(k, b, ms.shape)]   # a scan thinner than a cube: the whole axis
        St = [max(1, a // 2) for a in K]
        g = [-(-n * bb // kk) for n, bb, kk in zip(ms.shape, b, k)]   # cells per axis
        reg = lambda lo: tuple(slice(a * kk // bb, a * kk // bb + kd) for a, kk, bb, kd in zip(lo, k, b, K))
        for lo, hi in prev:
            if len(boxes) < CAP and ms[reg(lo)].numel() and float(ms[reg(lo)].max()) >= TAU / 2:
                boxes.append((lo, hi))
        for lo, _ in boxes:
            ms[reg(lo)] = -1
        while len(boxes) < CAP:
            sc = F.max_pool3d(ms[None, None], K, St, ceil_mode=True)[0, 0]
            if float(sc.max()) < TAU:
                break
            c = [int(a) * st for a, st in zip(torch.unravel_index(sc.argmax(), sc.shape), St)]   # map units
            lo = [min(a * bb // kk, max(0, n - CUBE)) for a, bb, kk, n in zip(c, b, k, g)]
            ms[reg(lo)] = -1
            boxes.append((lo, [min(n, a + CUBE) for a, n in zip(lo, g)]))
        return boxes

    def _cube(self, vid: int, mm: float, lo, hi):
        """One cube's (cells, tokens) of scan vid: from the RAM cache (LRU, CUBE_RAM_GB), the disk (cube_dir/read, when
        set), else read now by the frozen encoder as one view (box-local curve keys, the curves averaged) and cached."""
        key = (vid, mm, tuple(lo))
        if key in self.ram:
            self.ram.move_to_end(key)
            return self.ram[key]
        f = None if self.cube_dir is None else self.cube_dir / "read" / f"{vid}_{mm:g}_{'_'.join(map(str, lo))}.pt"
        if f is not None and f.exists():
            q, h = torch.load(f, weights_only=False)
        else:
            q, h = self._read_box(vid, mm, lo, hi)
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                torch.save((q, h), f)
        self.ram[key] = (q, h)
        self.ram_bytes += h.numel() * 2
        while self.ram_bytes > CUBE_RAM_GB * 2 ** 30 and len(self.ram) > 1:
            self.ram_bytes -= self.ram.popitem(last=False)[1][1].numel() * 2
        return q, h

    @torch.no_grad()
    def _read_box(self, vid: int, mm: float, lo, hi):
        from ..curves import grid_coords, keys
        from ..data.volume import patchify
        x, sp = self.vols[vid]
        dev, k = self.sel_probe.weight.device, self._cell(sp.tolist(), mm)
        crop = x[0][tuple(slice(a * kk, b * kk) for a, b, kk in zip(lo, hi, k))].to(dev).float()
        n = [b - a for a, b in zip(lo, hi)]
        crop = F.pad(crop, [p for c, m in zip(reversed(crop.shape), reversed([a * kk for a, kk in zip(n, k)]))
                            for p in (0, m - c)])
        loc = grid_coords(torch.tensor(n, device=dev)[None], math.prod(n))[0]
        q, h, cu = loc + torch.tensor(lo, device=dev), 0, torch.tensor([0, len(loc)], dtype=torch.int32, device=dev)
        pe = self.encoder.patch_embed
        mm0, pe.patch_mm = pe.patch_mm, mm
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xe = pe(patchify(crop, k), pe.weights(sp.to(dev), k))
                for c in self.curves:
                    o = keys(c, loc, torch.tensor(n, device=dev), self.seed).argsort()
                    h = h + self.encoder.flat(xe[o], cu, pos=q[o]).float()[o.argsort()]
        finally:
            pe.patch_mm = mm0
        return q.short().cpu(), (h / len(self.curves)).half().cpu()

    def _src(self, vid: int, corner, T) -> dict:
        """The tile's fine tokens {mm: (cells, tokens, k)}: the scan's chosen cubes touching the tile (training: DROP of
        them left out), fetched through _cube. ponytail: cells in two cubes appear twice (the scatter keeps one)."""
        out = {}
        if vid not in self.vols:
            return out
        sp = self.vols[vid][1]
        for mm in MG:
            k = self._cell(sp.tolist(), mm)
            parts = [self._cube(vid, mm, lo, hi) for lo, hi in self.boxes.get((vid, mm), ())
                     if all(a * kk < c + t and b * kk > c for a, b, kk, c, t in zip(lo, hi, k, corner, T))
                     and not (self.training and float(torch.rand(())) < DROP)]
            if parts:
                out[mm] = (torch.cat([q for q, _ in parts]), torch.cat([h for _, h in parts]), k)
        return out

    def _set_boxes(self, vid: int, sp, ms) -> int:
        """Both levels' cubes of scan vid from its map (hysteresis on the previous ones); returns the changes."""
        n = 0
        for mm in MG:
            k = self._cell(sp.tolist(), mm)
            old = self.boxes.get((vid, mm), [])
            new = [] if not fine_ok(k) or mm in self.off else self._choose(ms, k, old)
            n += len({tuple(a) for a, _ in new} ^ {tuple(a) for a, _ in old})
            self.boxes[(vid, mm)] = new
        return n

    def refresh(self, scans, it: int) -> None:
        """segrun, every REFRESH iterations: the training scans' cubes from their selector maps (a list: nothing read)."""
        n = sum(self._set_boxes(int(c[6]), c[:3].float(), self.sel_map[int(c[6])]) for _, c in scans)
        if it % 625 == 0:
            nb = {mm: sum(len(v) for (_, m), v in self.boxes.items() if m == mm) for mm in MG}
            print(f"[seg] cubes at it {it}: {nb} open, {n} changed since the last refresh, RAM cache "
                  f"{self.ram_bytes / 2 ** 30:.1f} GB, gains {[round(float(a), 3) for a in self.gain.detach()]}", flush=True)

    def on_val(self, it: int, every: int, scans, predict, resume: bool = False) -> None:
        """segrun: at the start (it 0) and on resume the training scans (for reading their cubes), their maps from the
        probe and their cubes; after the other validations nothing (refresh does it)."""
        if it and not resume:
            return
        self.vols = {int(c[6]): (x, c[:3].float()) for x, c in scans}
        with torch.no_grad():
            self.sel_map = {int(c[6]): self._probe_map(x, c) for x, c in scans}
        self.boxes = {}
        self.refresh(scans, 0)

    @torch.no_grad()
    def prepare(self, x: torch.Tensor, ctx: torch.Tensor, predict) -> None:
        """Val / test (segrun.sliding, before its pass): the selector's map on the scan (sel_mode pass), then its cubes,
        read during the pass as the tiles touch them."""
        vid = int(ctx[6])
        self.vols = {v: s for v, s in self.vols.items() if v in self.sel_map} | {vid: (x, ctx[:3].float())}
        self.sel_mode, self.preparing = True, True
        try:
            p = predict()
        finally:
            self.sel_mode, self.preparing = False, False
        for mm in MG:
            self.boxes.pop((vid, mm), None)
        self._set_boxes(vid, ctx[:3].float(),
                        F.max_pool3d(p[1:2].float()[None], self.cum[2], self.cum[2], ceil_mode=True)[0, 0].cpu())

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        yl = self.sel_y   # the training tiles' labels (segrun.sel_target): the selector's and the deep supervision's target
        if self.vol_stats:   # z-score per scan (the stats of the scan the tile comes from)
            ms = torch.tensor([self.vol_stats[int(r[6])] for r in ctx.tolist()], device=x.device, dtype=x.dtype)
            x = (x - ms[:, 0, None, None, None, None]) / ms[:, 1, None, None, None, None]
        T0 = tuple(x.shape[2:])
        x = F.pad(x, [p for t, m in zip(reversed(T0), reversed(self.cum[2])) for p in (0, -t % m)])
        T, dev = tuple(x.shape[2:]), x.device
        fine = {1: [], 2: []}
        for r in ctx.tolist():
            src = {} if self.sel_mode else self._src(int(r[6]), [int(v) for v in r[7:10]], T)
            for mm, f in self.sparse(src, [int(v) for v in r[7:10]], T, dev).items():
                fine[NSTAGE[mm]].append(f)
        f0 = self.e0(x)
        drop = lambda t: F.dropout(t, 0.1, self.training)   # the token inputs: 4096 dims, small datasets
        f1 = torch.cat([self.e1(f0), self.gain[0] * drop(torch.cat(fine[1]).to(x.dtype))], 1)
        e2 = self.e2(f1)
        sel = None
        if self.sel_mode or (self.training and self.sel_y is not None):
            b = self.cum[2]
            pl = []
            for r in ctx.tolist():
                fe, c0, k8 = self._probe_feats(self.cache.ids[int(r[6])], [int(v) for v in r[7:10]], T)
                pl.append(gather(self.sel_probe(fe.to(dev)), c0, k8, [int(v) for v in r[7:10]], T, b))
            sel = torch.cat(pl) + self.sel_corr(e2.detach().float())   # [B, 1, *T / b]
        if self.sel_mode:   # the selector's map in the decoder's channels: softmax and sigmoid both give sigma(sel)
            up = sel.float()
            for d, a in enumerate(b):
                up = up.repeat_interleave(a, 2 + d)
            n = self.out.out_channels
            out = torch.full((len(x), n, *T), -1e4, device=dev)
            out[:, 0] = 0
            out[:, 1] = up[:, 0]
            return out[:, :, : T0[0], : T0[1], : T0[2]]
        if sel is not None:   # BCE + Tversky (false negatives x 0.7) on the bottleneck voxels holding foreground
            y = self.sel_y.to(dev)
            y = F.pad(((y > 0) & (y < 255)).float(), [p for t, m in zip(reversed(T0), reversed(b)) for p in (0, -t % m)])
            tgt = F.max_pool3d(y.reshape(len(x), 1, *T), b, b)
            pr = sel.float().sigmoid()
            tp, fn, fp = ((pr * tgt).sum((1, 2, 3, 4)), ((1 - pr) * tgt).sum((1, 2, 3, 4)),
                          (pr * (1 - tgt)).sum((1, 2, 3, 4)))
            self.aux = F.binary_cross_entropy_with_logits(sel.float(), tgt) + \
                (1 - (tp + 1) / (tp + 0.7 * fn + 0.3 * fp + 1)).mean()
            self.sel_y = None
            with torch.no_grad():   # the training scans' maps follow the selector (EMA on the tile's region)
                for r, pj in zip(ctx.tolist(), pr.detach().cpu()):
                    m = self.sel_map.get(int(r[6]))
                    if m is not None:
                        o = [int(c) // a for c, a in zip(r[7:10], b)]
                        sl = tuple(slice(a, a + n) for a, n in zip(o, pj.shape[1:]))
                        reg = m[sl]
                        reg.mul_(0.5).add_(0.5 * pj[0][tuple(slice(0, n) for n in reg.shape)])
        f2 = torch.cat([e2, drop(self.fields(ctx, T, dev).to(x.dtype)), self.gain[1] * drop(torch.cat(fine[2]).to(x.dtype))], 1)
        h = hm = self.mid(f2)
        h = h1 = self.d1(torch.cat([self._up(self.u1, h, [b // a for a, b in zip(self.cum[1], self.cum[2])]), f1], 1))
        if self.training and yl is not None and not getattr(self, "landmark", False):   # nnU-Net's weights 0.5, 0.25
            from .segrun import loss_fn
            y = F.pad(yl.to(dev).long(), [p for t, m in zip(reversed(T0), reversed(self.cum[2])) for p in (0, -t % m)],
                      value=255)
            for w, hd, head, st in ((0.5, h1, self.ds[0], self.cum[1]), (0.25, hm, self.ds[1], self.cum[2])):
                self.aux = self.aux + w * loss_fn(head(hd).float(), y[:, :: st[0], :: st[1], :: st[2]], self.out.out_channels)
        h = self.d0(torch.cat([self._up(self.u0, h, self.cum[1]), f0], 1))
        return self.out(h)[:, :, : T0[0], : T0[1], : T0[2]]

    @staticmethod
    def _up(u: nn.ConvTranspose3d, h: torch.Tensor, st) -> torch.Tensor:
        """The transposed conv with this plan's stride per axis (kernel = stride: an axis not halved keeps one tap)."""
        return F.conv_transpose3d(h, u.weight[:, :, : st[0], : st[1], : st[2]], u.bias, stride=tuple(st))
