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
        out = []
        for r in ctx.tolist():
            i, corner = self.cache.ids[int(r[6])], [int(v) for v in r[7:10]]
            k4, lv = self.cache.meta(i)["k4"], []
            for j, mm in enumerate(self.levels):
                k = [a * int(mm // 4) for a in k4]
                if mm < 8 and not fine_ok(k):   # a scan too coarse for 4 mm cells (no file): an empty level
                    lv.append(torch.zeros(1, TOK, *[t // 4 for t in T], device=dev))
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
                g = F.interpolate(g, size=[max(1, f // 4) for f in full], mode="trilinear", align_corners=False)
                o = [(c - a * kk) // 4 for c, a, kk in zip(corner, c0, k)]
                g = g[:, :, o[0]: o[0] + T[0] // 4, o[1]: o[1] + T[1] // 4, o[2]: o[2] + T[2] // 4]
                g = F.pad(g, [p for d, s in zip(reversed(g.shape[2:]), reversed([t // 4 for t in T])) for p in (0, s - d)])
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
CUBE, CAP, TAU, DROP = 16, 8, 0.1, 0.3   # cells per axis, cubes per volume and level, min foreground prob, cubes dropped in training


def gather(g: torch.Tensor, c0, k, corner, T, s: int) -> torch.Tensor:
    """g [*n, C] on the cell grid from cell c0 (cells of k voxels) -> [1, C, *T / 2^s]: each voxel of stage s takes the cell
    its centre lies in (nearest, no interpolation: a token is a cell, not a point sample)."""
    st, dev = 2 ** s, g.device
    idx = [((torch.arange(t // st, device=dev) * st + st // 2 + c) // kk - a).clamp(0, n - 1)
           for t, c, kk, a, n in zip(T, corner, k, c0, g.shape[:3])]
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
            s, w = NSTAGE[mm], NTOK[mm] + 1
            if mm not in src:
                out[mm] = torch.zeros(1, w, *[t // 2 ** s for t in T], device=dev)
                continue
            q, h, k = src[mm]
            q = q.to(dev).long()
            c0 = [c // kk for c, kk in zip(corner, k)]
            n = [-(-(c + t) // kk) - a for c, t, kk, a in zip(corner, T, k, c0)]
            lo, hi = torch.tensor(c0, device=dev), torch.tensor(c0, device=dev) + torch.tensor(n, device=dev)
            ok = ((q >= lo) & (q < hi)).all(1)
            g = torch.zeros(*n, w, device=dev)
            if ok.any():
                f = self.tok[f"g{mm:g}"](h[ok.cpu()].to(dev).float())
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

    @torch.no_grad()
    def place(self, x: torch.Tensor, ctx: torch.Tensor, level: int, predict) -> None:
        """Level `level` (1: 4 mm, 2: 2 mm) of the scan x [1, *S] (ctx: spacing, shape, volume index): predict() with the
        levels above, cubes greedily on the foreground probability (each cube's cells excluded from the next centres),
        each cube read by the frozen encoder as one view (box-local curve keys, the curves averaged)."""
        from ..curves import grid_coords, keys
        from ..data.volume import patch_dims, patchify
        vid, mm = int(ctx[6]), MG[level - 1]
        self.active, self.preparing = level - 1, True
        try:
            p = predict()
        finally:
            self.active, self.preparing = None, False
        dev, sp = p.device, ctx[:3].float()
        k = patch_dims(sp.tolist(), mm)
        self.cubes.setdefault(vid, {}).pop(mm, None)
        if not fine_ok(k):
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
