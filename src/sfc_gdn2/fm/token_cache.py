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
        self.sig = None    # {checkpoint, taps, curves, pretrained} of the reading encoder (SegNet sets it), checked once

    def check(self) -> None:
        """The folder's tokens are this encoder's (27/10: a blocks.json copied from another encoder's folder went
        unnoticed): <root>/cache.json written by the LiveCache that starts the folder, else compared; folders started
        before it (tokens of an unknown encoder) are never stamped and pass."""
        if self.sig is None:
            return
        f, sig, self.sig = self.root / "cache.json", json.loads(json.dumps(self.sig, default=str)), None
        if f.exists():
            old = json.loads(f.read_text())
            if old != sig:
                raise ValueError(f"token cache {self.root} holds another encoder's tokens: {old} != {sig}")
        elif isinstance(self, LiveCache) and not any(self.root.glob("*.npy")):
            self.root.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(sig, indent=1))

    def meta(self, i: str) -> dict:
        self.check()
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
        k4 = write_volume(enc, curves, seed, r["x"], r["ctx"], out, i, cv, float(job.get("min_mm", 4)),
                          job.get("nmax", 200_000), job.get("block", 64), job.get("margin", 8))
        print(f"[cache] {c + 1}/{len(ids)} {i} k4 {list(k4)}", flush=True)


def write_volume(enc, curves, seed, x, ctx, out: Path, i: str, cv: dict, min_mm: float = 4.0,
                 nmax: int = 200_000, block: int = 64, margin: int = 8) -> list:
    """Volume i's levels >= min_mm (cv: {mm: curves}, else the mean) and <out>/<i>.json, written last (the volume's
    done marker); each file through a temporary name, so a reader never opens half of one -> k4."""
    from ..data.volume import patch_dims
    x, sp = x.cuda(), ctx.float().cuda()
    k4 = patch_dims(sp.tolist(), 4.0)
    for mm, m in LEVELS:
        if mm < min_mm or mm < 8 and not fine_ok([kk * m for kk in k4]):   # min_mm 8: fine_live4
            continue
        want = [w for w in cv.get(mm, ["mean"]) if not level_file(out, i, mm, w).exists()]
        if not want:
            continue
        a = read(enc, curves, seed, x, sp, [kk * m for kk in k4], mm, nmax, block, margin, taps=True, want=want)
        for w, t in a.items():
            dst, tmp = out / f"{i}.{int(mm)}.{w}.npy", out / f"{i}.{int(mm)}.{w}.{os.getpid()}.tmp"   # per process:
            with open(tmp, "wb") as f:                                                              # two jobs may
                np.save(f, t)                                                                       # write one volume
            os.replace(tmp, dst)
    tmp = out / f"{i}.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"k4": list(k4), "spacing": sp.tolist(), "shape": list(x.shape[1:])}))
    os.replace(tmp, out / f"{i}.json")
    return k4

if __name__ == "__main__":
    main(sys.argv[1])


class LiveCache(TokenCache):
    """A TokenCache that writes the volumes it lacks itself (24/10, user: the cache live, on $FAST; no separate cache
    job): segrun calls ensure() as it loads each volume, which reads the missing levels with the job's frozen encoder
    exactly as main() (the blocks and cached curves of the folder's blocks.json) and writes them to the folder, so later
    runs (seeds, other tasks on the same volumes) only read. token_cache "live:<folder>"."""

    def __init__(self, root, net):
        super().__init__(root)
        self.net = net

    def ensure(self, i: str, x: torch.Tensor, ctx: torch.Tensor) -> None:
        self.check()
        if (self.root / f"{i}.json").exists():
            return
        from .block_plan import apply
        enc = self.net.encoder
        keep, mode = (enc.depth, enc.taps, enc.patch_embed.patch_mm), enc.training
        bj = self.root / "blocks.json"   # without one (no block_plan, 27/10): the decoder's last ++ taps, the curves' mean
        pl = json.loads(bj.read_text()) if bj.exists() else {}
        cv = {float(m): list(v) if isinstance(v, list) else [v] for m, v in (pl.get("cache_curves") or {}).items()}
        try:
            if "cache" in pl:
                apply(enc, pl["cache"])   # the union of the levels' blocks, as the cache job's encoder
            enc.eval()                # segrun loads the volumes before its first set_mode()
            with torch.no_grad():
                write_volume(enc, self.net.enc_curves, self.net.seed, x, ctx, self.root, i, cv, min(self.levels))
        finally:
            enc.depth, enc.taps, enc.patch_embed.patch_mm = keep
            enc.train(mode)
