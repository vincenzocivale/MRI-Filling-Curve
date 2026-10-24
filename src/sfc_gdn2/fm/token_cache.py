"""Encoder tokens of the segmentation volumes, read once (20/10): the decoder trains on them instead of running the
frozen encoder on every tile at three resolutions.

Per volume and level (16, 8, 4 mm) the tokens of the whole scan on nested grids: the cell is k4 = round(4 / spacing)
voxels per axis at 4 mm, 2 k4 at 8 mm, 4 k4 at 16 mm, so a tile whose corner sits on the 16 mm grid holds whole
cells of every level. Each token is the encoder's reading along each curve of the run (averaged), as SegNet's: at
every level as last block ++ the decoder's taps (the width SegNet.proj takes; each decoder keeps its channels). A scan
whose grid holds more than `nmax` tokens is read in windows of `block` cells per axis with `margin` cells of context
on every side, keeping each window's centre (TotalSeg whole-body at 4 mm: ~5e6 tokens); the 4 mm level only where
fine_ok. Stored as <out>/<id>.<mm>.<curve>.npy [gx, gy, gz, C] fp16 + <out>/<id>.json {k4, spacing, shape}; <curve>
is one curve read alone or `mean` (the curves' average); a file without the tag (caches before 24/10) is the mean.
Which curves per level (24/10, user: the cached levels chosen by blocks and curve, as the live ones): job["curves"]
{mm: [names]}, else block_plan's <out>/blocks.json "cache_curves", else the mean; each curve is read once for all of them.

    python -m sfc_gdn2.fm.token_cache <job.json>   {inputs, ids, out, checkpoint, args, shard, nmax, block, margin,
                                                    curves, random: the seeded random encoder, for the random-encoder baseline}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

LEVELS = ((16.0, 4), (8.0, 2), (4.0, 1))   # (mm, cells of 4 mm per axis)


def fine_ok(k) -> bool:
    """A fine level is read only where its cell holds >= 2 voxels per axis on average (>= 8 voxels): finer cells of
    coarse scans are single voxels (TotalSeg 4.2 x 4.2 x 2 mm at 4 mm: [1, 1, 2]), nothing left to refine (20/10)."""
    return int(np.prod(k)) >= 8


def level_file(root: Path, i: str, mm: float, curve: str = "mean") -> Path:
    """The cache file of volume i, level mm, read along `curve` (a name or "mean"); untagged files are the mean."""
    p = Path(root) / f"{i}.{int(mm)}.{curve}.npy"
    old = Path(root) / f"{i}.{int(mm)}.npy"
    return old if curve == "mean" and not p.exists() and old.exists() else p


class TokenCache:
    """Crops of the cached tokens: tokens(id, level, corner voxel, cells per axis) -> [n, C] (zeros past the scan).
    Read with explicit contiguous reads (whole rows along the last axes), not a memmap: on Lustre a memmap faults its
    pages in one by one, ~80 s per tile of 4 mm tokens (20/10)."""

    def __init__(self, root: str | Path):
        from concurrent.futures import ThreadPoolExecutor
        self.root, self.ids, self._fd, self._pool = Path(root), [], {}, ThreadPoolExecutor(32)
        self._io, self._ahead, self.levels, self._meta, self._gen = ThreadPoolExecutor(16), {}, (16.0,), {}, 0
        self._recent = {}
        self.cols = None   # SegNet blocks: the columns of the decoder's blocks in a cache of more (block_plan.layout)
        self.curves = {}   # {mm: curve} the decoder's curve per cached level (block_plan), else the mean

    def meta(self, i: str) -> dict:
        if i not in self._meta:
            self._meta[i] = json.loads((self.root / f"{i}.json").read_text())
        return self._meta[i]

    def _open(self, i: str, mm: float):
        key, name = (i, mm), level_file(self.root, i, mm, self.curves.get(float(mm), "mean"))
        if key not in self._fd:
            with open(name, "rb") as f:
                major, _ = np.lib.format.read_magic(f)
                read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                shape, _, dtype = read(f)
                off = f.tell()
            self._fd[key] = (os.open(name, os.O_RDONLY), shape, np.dtype(dtype), off)
        return self._fd[key]

    def next_batch(self, keep: int = 1) -> None:
        """A new batch is being prefetched: drop the reads of the batches more than `keep` + 1 batches back."""
        self._gen += 1
        self._ahead = {k: v for k, v in self._ahead.items() if v[0] >= self._gen - keep - 1}

    def prefetch(self, ctx: torch.Tensor, T) -> None:
        """Start reading the tokens a batch of tiles will ask for (ctx rows as SegNet takes them, tiles of T voxels)
        while the current batch trains: the levels in `self.levels` (set by SegNet)."""
        for r in ctx.tolist():
            i, corner = self.ids[int(r[6])], [int(v) for v in r[7:10]]
            V = [max(1, min(int(e) - c, t)) for e, c, t in zip(r[3:6], corner, T)]
            k4 = self.meta(i)["k4"]
            for mm in self.levels:
                k = [a * int(mm // 4) for a in k4]
                if mm < 8 and not fine_ok(k):
                    continue
                n = [-(-v // kk) for v, kk in zip(V, k)]
                key = (i, mm, tuple(corner), tuple(n))
                if key not in self._ahead:
                    self._ahead[key] = (self._gen, self._io.submit(lambda *a: self._cols(self._read(*a), a[1]), i, mm, corner, n, k))

    def tokens(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        key = (i, mm, tuple(int(c) for c in corner), tuple(n))
        f = self._ahead.get(key)   # kept: micro_step reads twice
        if f is not None:
            return f[1].result()
        if key not in self._recent:   # the last few reads (adaptive decoding reads a scan's levels once per chunk)
            self._recent = dict(list(self._recent.items())[-3:]) | {key: self._cols(self._read(i, mm, corner, n, k), mm)}
        return self._recent[key]

    def _cols(self, t: torch.Tensor, mm: float) -> torch.Tensor:
        c = self.cols.get(float(mm)) if isinstance(self.cols, dict) else self.cols   # per level: {mm: columns}
        return t if c is None else t[..., c]

    def _read(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        fd, shape, dt, off = self._open(i, mm)
        C = shape[-1]
        c0 = [int(c) // int(kk) for c, kk in zip(corner, k)]
        c1 = [min(c + m, s) for c, m, s in zip(c0, n, shape[:3])]
        out = np.zeros((*n, C), dt)
        if all(b > a for a, b in zip(c0, c1)):
            row, cell = shape[2] * C * dt.itemsize, C * dt.itemsize   # one (x, y) row of every z; one token
            ny, nz = c1[1] - c0[1], c1[2] - c0[2]

            def plane(x):   # Lustre serves parallel reads in parallel: one thread per x plane
                base = off + (x * shape[1] + c0[1]) * row
                if 2 * nz >= shape[2]:   # the crop spans most of z: the plane's y range in one read
                    a = np.frombuffer(os.pread(fd, ny * row, base), dt).reshape(ny, shape[2], C)
                    out[x - c0[0], :ny, :nz] = a[:, c0[2]: c1[2]]
                else:                    # long z (whole-body scans): each row's z range
                    for y in range(ny):
                        out[x - c0[0], y, :nz] = np.frombuffer(os.pread(fd, nz * cell, base + y * row + c0[2] * cell),
                                                               dt).reshape(nz, C)
            list(self._pool.map(plane, range(c0[0], c1[0])))
        return torch.from_numpy(out.reshape(-1, C))


@torch.no_grad()
def read(encoder, curves, seed, x, sp, k, mm, nmax, block, margin, taps: bool, want=None):
    """[1, *S] voxels -> [gx, gy, gz, C] tokens of the grid of k-voxel cells (windows past nmax tokens), the mean over
    `curves`. want: names among `curves` and "mean" -> {name: tokens}, each curve read once."""
    from ..curves import grid_coords, keys
    from ..data.volume import patch_grid, patchify

    pe = encoder.patch_embed
    pe.patch_mm = mm
    w = pe.weights(sp, k)
    G = patch_grid(tuple(x.shape[1:]), k).tolist()
    width = encoder.dim * ((1 + len(encoder.taps)) if taps else 1)
    names = list(want or ["mean"])
    out = {n: np.zeros((*G, width), np.float16) for n in names}
    whole = int(np.prod(G)) <= nmax
    step = G if whole else [block] * 3
    for o0 in range(0, G[0], step[0]):
        for o1 in range(0, G[1], step[1]):
            for o2 in range(0, G[2], step[2]):
                lo = [max(0, o - (0 if whole else margin)) for o in (o0, o1, o2)]
                hi = [min(g, o + s + (0 if whole else margin)) for o, s, g in zip((o0, o1, o2), step, G)]
                box = x[(slice(None), *(slice(a * kk, b * kk) for a, b, kk in zip(lo, hi, k)))]
                g = torch.tensor([b - a for a, b in zip(lo, hi)], device=x.device)
                t = pe(patchify(box[0].float(), k), w)
                xyz = grid_coords(g[None], int(g.prod()))[0]
                cu = torch.tensor([0, len(t)], device=x.device, dtype=torch.int32)
                hs, tot = {}, 0
                for c in (curves if "mean" in names else [c for c in curves if c in names]):
                    o = keys(c, xyz, g, seed).argsort()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = encoder.flat(t[o], cu, pos=xyz[o]).float()[o.argsort()]
                    if c in names:
                        hs[c] = h
                    if "mean" in names:
                        tot = tot + h
                if "mean" in names:
                    hs["mean"] = tot / len(curves)
                c0 = [o - a for o, a in zip((o0, o1, o2), lo)]
                c1 = [min(c + s, gg) for c, s, gg in zip(c0, step, g.tolist())]
                for n, h in hs.items():
                    h = (h if taps else h[:, : encoder.dim]).reshape(*g.tolist(), -1)
                    out[n][o0: o0 + c1[0] - c0[0], o1: o1 + c1[1] - c0[1], o2: o2 + c1[2] - c0[2]] = \
                        h[c0[0]: c1[0], c0[1]: c1[1], c0[2]: c1[2]].half().cpu().numpy()
    return out if want else out["mean"]


def main(path: str) -> None:
    from ..data.volume import patch_dims
    from .api import load_config
    from .sfc import SFC

    job = json.loads(Path(path).read_text())
    cfg = load_config("configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"], cfg["args"] = job["checkpoint"], job.get("args", {})
    w = SFC(cfg, "cuda")
    if job.get("random"):   # the random-encoder baseline reads tokens of a random encoder (seeded), not the trained one
        torch.manual_seed(0)
    net = w.seg_net(2, pretrained=not job.get("random"))[0].cuda().eval()   # the decoder's taps, the run's curves
    enc, curves, seed = net.encoder, net.enc_curves, net.seed
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    i0, n = (int(v) for v in job.get("shard", "0/1").split("/"))
    ids = sorted(job["ids"])[i0::n]
    bj = out / "blocks.json"
    cv = job.get("curves") or (json.loads(bj.read_text()).get("cache_curves") if bj.exists() else None) or {}
    cv = {float(m): list(v) for m, v in cv.items()}
    print(f"[cache] curves per level {cv or 'mean'}", flush=True)
    for c, i in enumerate(ids):
        lv = [(mm, m) for mm, m in LEVELS if mm >= float(job.get("min_mm", 4))]
        if (out / f"{i}.json").exists() and all(level_file(out, i, mm, w).exists() for mm, _ in lv for w in cv.get(mm, ["mean"])):
            continue
        r = torch.load(Path(job["inputs"]) / f"{i}.pt", map_location="cpu", weights_only=False)
        x, sp = r["x"].cuda(), r["ctx"].float().cuda()
        k4 = patch_dims(sp.tolist(), 4.0)
        for mm, m in LEVELS:
            if mm < float(job.get("min_mm", 4)) or mm < 8 and not fine_ok([kk * m for kk in k4]):   # min_mm 8: fine_live4
                continue
            want = [w for w in cv.get(mm, ["mean"]) if not level_file(out, i, mm, w).exists()]
            if not want:
                continue
            a = read(enc, curves, seed, x, sp, [kk * m for kk in k4], mm, job.get("nmax", 200_000),
                     job.get("block", 64), job.get("margin", 8), taps=True, want=want)
            for w, t in a.items():
                np.save(out / f"{i}.{int(mm)}.{w}.npy", t)
        (out / f"{i}.json").write_text(json.dumps({"k4": list(k4), "spacing": sp.tolist(), "shape": list(x.shape[1:])}))
        print(f"[cache] {c + 1}/{len(ids)} {i} k4 {list(k4)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])


class LiveCache:
    """The TokenCache interface computed in the job, lazily (23/10, user: no separate cache job, the tokens read when
    needed and kept). 16 mm: the whole scan read once per volume (as main()). Finer levels: in fixed blocks, the cells
    owned by one 16 mm cell (those whose first voxel lies in it), read with a ring of one 16 mm cell of context (a ~48 mm
    cube): a token depends only on its volume and block, the same in training, val and test, and is read once.
    `inputs`: the segdec input cache (<id>.pt); `net`: the SegNet (its frozen encoder, curves, seed, taps)."""

    def __init__(self, inputs: str | Path, net):
        from collections import OrderedDict
        self.inputs, self.net, self.ids, self.levels = Path(inputs), net, [], (16.0,)
        self._vol, self._t16, self._blk, self._meta = {}, {}, OrderedDict(), {}
        # blocks kept least recently used first, at most SFC_LIVE_GB (an untrained cascade asks for nearly every block:
        # 4096-dim 2 mm tokens of every volume went past 120 GB of host memory, 23/10); evicted ones are read again
        self._cap, self._bytes = float(os.environ.get("SFC_LIVE_GB", "40")) * 2 ** 30, 0

    def _load(self, i: str):
        if i not in self._vol:
            r = torch.load(self.inputs / f"{i}.pt", map_location="cpu", weights_only=False)
            self._vol[i] = (r["x"], r["ctx"].float())
        return self._vol[i]

    def meta(self, i: str) -> dict:
        if i not in self._meta:
            from ..data.volume import patch_dims
            x, sp = self._load(i)
            self._meta[i] = {"k4": list(patch_dims(sp[:3].tolist(), 4.0)), "shape": list(x.shape[1:])}
        return self._meta[i]

    def next_batch(self, keep: int = 1) -> None:
        pass

    def prefetch(self, ctx, T) -> None:
        pass

    def _read(self, i, x, mm, k):
        enc = self.net.encoder
        mm0 = enc.patch_embed.patch_mm
        try:
            dev = next(enc.parameters()).device
            sp = self._load(i)[1][:3].to(dev)
            return read(enc, self.net.enc_curves, self.net.seed, x.to(dev), sp, k, mm, 200_000, 64, 8, taps=True)
        finally:
            enc.patch_embed.patch_mm = mm0

    def _block(self, i, mm, kf, b):
        """The tokens [*owned, C] of the level-mm cells owned by 16 mm cell b (k16 voxels per axis), and their range."""
        key = (i, mm, tuple(kf), b)
        if key in self._blk:
            self._blk.move_to_end(key)
        else:
            x, _ = self._load(i)
            k16 = [4 * a for a in self.meta(i)["k4"]]
            S = x.shape[1:]
            G = [-(-s // k) for s, k in zip(S, kf)]
            lo = [-(-bb * a // k) for bb, a, k in zip(b, k16, kf)]                    # owned cells [lo, hi)
            hi = [min(-(-(bb + 1) * a // k), g) for bb, a, k, g in zip(b, k16, kf, G)]
            ring = [-(-a // k) for a, k in zip(k16, kf)]
            c0 = [max(0, l - r) for l, r in zip(lo, ring)]                             # context cells [c0, c1)
            c1 = [min(g, h + r) for h, r, g in zip(hi, ring, G)]
            box = x[(slice(None), *(slice(a * k, b_ * k) for a, b_, k in zip(c0, c1, kf)))]
            t = self._read(i, box, mm, kf)
            own = np.ascontiguousarray(t[tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, c0))])
            self._blk[key] = (own, lo, hi)
            self._bytes += own.nbytes
            while self._bytes > self._cap and len(self._blk) > 1:
                self._bytes -= self._blk.popitem(last=False)[1][0].nbytes
        return self._blk[key]

    def tokens_at(self, i: str, mm: float, corner, n, k, keep=None) -> torch.Tensor:
        """The tile's level-mm cells (grid n from corner, raster), only rows `keep` if given -> [rows, C] float16."""
        k = [int(a) for a in k]
        if mm >= 16:
            if i not in self._t16:
                self._t16[i] = self._read(i, self._load(i)[0], 16.0, k)
            full = self._t16[i]
            c0 = [int(c) // kk for c, kk in zip(corner, k)]
            out = np.zeros((*n, full.shape[-1]), np.float16)
            c1 = [min(c + m, s) for c, m, s in zip(c0, n, full.shape[:3])]
            if all(b > a for a, b in zip(c0, c1)):
                out[: c1[0] - c0[0], : c1[1] - c0[1], : c1[2] - c0[2]] = full[c0[0]: c1[0], c0[1]: c1[1], c0[2]: c1[2]]
            t = torch.from_numpy(out.reshape(-1, out.shape[-1]))
            return t if keep is None else t[keep.cpu()]
        k16 = [4 * a for a in self.meta(i)["k4"]]
        n = [int(a) for a in n]
        rows = torch.arange(int(np.prod(n))) if keep is None else keep.cpu()
        g = torch.stack([rows // (n[1] * n[2]), rows // n[2] % n[1], rows % n[2]], 1)
        g = g + torch.tensor([int(c) // kk for c, kk in zip(corner, k)])               # global cell coordinates
        S = self.meta(i)["shape"]
        G = torch.tensor([-(-s // kk) for s, kk in zip(S, k)])
        inside = (g < G).all(1)
        blk = (g * torch.tensor(k)) // torch.tensor(k16)                                 # owner: the first voxel's cell
        C = None
        out = None
        for b in torch.unique(blk[inside], dim=0).tolist():
            t, lo, _ = self._block(i, mm, k, tuple(b))
            if out is None:
                C = t.shape[-1]
                out = np.zeros((len(rows), C), np.float16)
            sel = (inside & (blk == torch.tensor(b)).all(1)).nonzero()[:, 0]
            loc = g[sel] - torch.tensor(lo)
            out[sel.numpy()] = t[loc[:, 0].numpy(), loc[:, 1].numpy(), loc[:, 2].numpy()]
        if out is None:
            C = self.net.encoder.dim * (1 + len(self.net.encoder.taps))
            out = np.zeros((len(rows), C), np.float16)
        return torch.from_numpy(out)

    def tokens(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        return self.tokens_at(i, mm, corner, n, k)
