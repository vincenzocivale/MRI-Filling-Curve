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
of 2 mm (a linear map, as a ViT's patch decoder / pixel shuffle), only then interpolated onto the 1/4 voxel grid."""
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
                 sub: bool = False):
        super().__init__()
        global CH, TOK   # ponytail: module-level widths scaled per run (one decoder per process); args.dense_width
        CH, TOK = tuple(int(c * width) for c in (16, 32, 64)), int(64 * width)
        self.encoder = encoder
        self.cache = TokenCache(cache)
        self.levels = (*LEVELS, 4.0) if dense4 else LEVELS   # simpleC: + 4 mm dense (a cache with the level: --dense4)
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
        tk = self.fields(ctx, T, x.device)   # the projections train; the cached tokens are data
        h = self.mid(torch.cat([f2, tk.to(f2.dtype)], 1))
        h = self.d1(torch.cat([self.u1(h), f1], 1))
        h = self.d0(torch.cat([self.u0(h), f0], 1))
        return self.out(h)[:, :, : T0[0], : T0[1], : T0[2]]

    @staticmethod
    def tile_max(n_out: int) -> float:
        """segrun.TILE_MAX for this decoder: tiles of at most ~160^3 voxels (its activations are dense)."""
        return float(160 ** 3 * n_out)


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
