"""Encoder tokens of the segmentation volumes, read once (20/10): the decoder trains on them instead of running the
frozen encoder on every tile at three resolutions.

Per volume and level (16, 8, 4 mm) the tokens of the whole scan on nested grids: the cell is k4 = round(4 / spacing)
voxels per axis at 4 mm, 2 k4 at 8 mm, 4 k4 at 16 mm, so a tile whose corner sits on the 16 mm grid holds whole
cells of every level. Each token is the encoder's reading along each curve of the run (averaged), as SegNet's: at
every level as last block ++ the decoder's taps (the width SegNet.proj takes; each decoder keeps its channels). A scan
whose grid holds more than `nmax` tokens is read in windows of `block` cells per axis with `margin` cells of context
on every side, keeping each window's centre (TotalSeg whole-body at 4 mm: ~5e6 tokens); the 4 mm level only where
fine_ok. Stored as <out>/<id>.<mm>.npy [gx, gy, gz, C] fp16 + <out>/<id>.json {k4, spacing, shape}.

    python -m sfc_gdn2.fm.token_cache project <job.json>   (the fine levels on a PCA basis: project())
    python -m sfc_gdn2.fm.token_cache <job.json>   {inputs, ids, out, checkpoint, args, shard, nmax, block, margin,
                                                    random: the seeded random encoder, for the random-encoder baseline}
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


class TokenCache:
    """Crops of the cached tokens: tokens(id, level, corner voxel, cells per axis) -> [n, C] (zeros past the scan).
    Read with explicit contiguous reads (whole rows along the last axes), not a memmap: on Lustre a memmap faults its
    pages in one by one, ~80 s per tile of 4 mm tokens (20/10)."""

    def __init__(self, root: str | Path):
        from concurrent.futures import ThreadPoolExecutor
        self.root, self.ids, self._fd, self._pool = Path(root), [], {}, ThreadPoolExecutor(16)
        self._io, self._ahead, self.levels, self._meta, self._gen = ThreadPoolExecutor(4), {}, (16.0,), {}, 0
        self._recent = {}
        self.pca = None   # SegNet fine_pca: the 8 / 4 mm levels from <id>.<mm>.p<pca>.npy (project())

    def meta(self, i: str) -> dict:
        if i not in self._meta:
            self._meta[i] = json.loads((self.root / f"{i}.json").read_text())
        return self._meta[i]

    def _open(self, i: str, mm: float, pca: int | None = None):
        key, name = (i, mm, pca), f"{i}.{int(mm)}{f'.p{pca}' if pca else ''}.npy"
        if key not in self._fd:
            with open(self.root / name, "rb") as f:
                major, _ = np.lib.format.read_magic(f)
                read = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
                shape, _, dtype = read(f)
                off = f.tell()
            self._fd[key] = (os.open(self.root / name, os.O_RDONLY), shape, np.dtype(dtype), off)
        return self._fd[key]

    def next_batch(self) -> None:
        """A new batch is being prefetched: drop what the batch before the current one read."""
        self._gen += 1
        self._ahead = {k: v for k, v in self._ahead.items() if v[0] >= self._gen - 1}

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
                    self._ahead[key] = (self._gen, self._io.submit(self._read, i, mm, corner, n, k))

    def tokens(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        key = (i, mm, tuple(int(c) for c in corner), tuple(n))
        f = self._ahead.get(key)   # kept: micro_step reads twice
        if f is not None:
            return f[1].result()
        if key not in self._recent:   # the last few reads (adaptive decoding reads a scan's levels once per chunk)
            self._recent = dict(list(self._recent.items())[-3:]) | {key: self._read(i, mm, corner, n, k)}
        return self._recent[key]

    def _read(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        fd, shape, dt, off = self._open(i, mm, self.pca if mm < 16 else None)
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
def read(encoder, curves, seed, x, sp, k, mm, nmax, block, margin, taps: bool) -> np.ndarray:
    """[1, *S] voxels -> [gx, gy, gz, C] tokens of the grid of k-voxel cells (windows past nmax tokens)."""
    from ..curves import grid_coords, keys
    from ..data.volume import patch_grid, patchify

    pe = encoder.patch_embed
    pe.patch_mm = mm
    w = pe.weights(sp, k)
    G = patch_grid(tuple(x.shape[1:]), k).tolist()
    width = encoder.dim * ((1 + len(encoder.taps)) if taps else 1)
    out = np.zeros((*G, width), np.float16)
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
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = sum(encoder.flat(t[o], cu).float()[o.argsort()]
                            for o in (keys(c, xyz, g, seed).argsort() for c in curves)) / len(curves)
                h = (h if taps else h[:, : encoder.dim]).reshape(*g.tolist(), -1)
                c0 = [o - a for o, a in zip((o0, o1, o2), lo)]
                c1 = [min(c + s, gg) for c, s, gg in zip(c0, step, g.tolist())]
                out[o0: o0 + c1[0] - c0[0], o1: o1 + c1[1] - c0[1], o2: o2 + c1[2] - c0[2]] = \
                    h[c0[0]: c1[0], c0[1]: c1[1], c0[2]: c1[2]].half().cpu().numpy()
    return out


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
    for c, i in enumerate(ids):
        if (out / f"{i}.json").exists():
            continue
        r = torch.load(Path(job["inputs"]) / f"{i}.pt", map_location="cpu", weights_only=False)
        x, sp = r["x"].cuda(), r["ctx"].float().cuda()
        k4 = patch_dims(sp.tolist(), 4.0)
        for mm, m in LEVELS:
            if mm < 8 and not fine_ok([kk * m for kk in k4]):
                continue
            a = read(enc, curves, seed, x, sp, [kk * m for kk in k4], mm, job.get("nmax", 200_000),
                     job.get("block", 64), job.get("margin", 8), taps=True)
            np.save(out / f"{i}.{int(mm)}.npy", a)
        (out / f"{i}.json").write_text(json.dumps({"k4": list(k4), "spacing": sp.tolist(), "shape": list(x.shape[1:])}))
        print(f"[cache] {c + 1}/{len(ids)} {i} k4 {list(k4)}", flush=True)


def _read_rows(path: Path, n_sample: int, rng) -> np.ndarray:
    """n_sample tokens of a level file in 8 contiguous runs (explicit reads), the empty (zero) ones dropped."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        shape, _, _ = (np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0)(f)
        off, C, N, run = f.tell(), shape[-1], int(np.prod(shape[:-1])), max(1, n_sample // 8)
        out = []
        for s in rng.choice(max(1, N - run), min(8, max(1, N - run)), replace=False):
            f.seek(off + int(s) * C * 2)
            out.append(np.frombuffer(f.read(run * C * 2), np.float16).reshape(-1, C))
    t = np.concatenate(out).astype(np.float32)
    return t[np.abs(t).sum(1) > 0]


def project(path: str) -> None:
    """The 8 and 4 mm levels on a fixed PCA basis (20/10): the decoder reads them only through a linear map
    (fine_tok . proj, 4096 -> 128), so the leading `dim` components keep what any such map can take, minus the
    variance dropped (512: 97-98 % of it for the pretrained encoder), at 4096 / dim the reads (Lustre-bound: 4 s per
    iteration at 4096 channels). job {cache, dim, fit: ids (train volumes only), shard "i/n"}: shard 0/n fits
    <cache>/pca<dim>.pt {mm: (mean, basis)} first (the other shards wait for it); every shard writes
    <id>.<mm>.p<dim>.npy fp16."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    job = json.loads(Path(path).read_text())
    root, dim = Path(job["cache"]), int(job["dim"])
    i0, n = (int(v) for v in job.get("shard", "0/1").split("/"))
    bfile = root / f"pca{dim}.pt"
    if i0 == 0 and not bfile.exists():
        rng, basis = np.random.default_rng(0), {}
        for mm in (8, 4):
            files = [root / f"{i}.{mm}.npy" for i in sorted(job["fit"]) if (root / f"{i}.{mm}.npy").exists()]
            pick = [files[j] for j in rng.choice(len(files), min(len(files), 96), replace=False)]
            with ThreadPoolExecutor(16) as ex:
                x = torch.from_numpy(np.concatenate(list(ex.map(lambda q: _read_rows(q, 3200, rng), pick)))).cuda().double()
            mu = x.mean(0)
            ev, V = torch.linalg.eigh((x - mu).T @ (x - mu) / len(x))
            basis[mm] = (mu.float().cpu(), V[:, -dim:].flip(1).float().cpu())
            print(f"[pca] {mm} mm: {len(x)} tokens of {len(pick)} train volumes, variance kept "
                  f"{float(ev[-dim:].sum() / ev.sum()):.4f}", flush=True)
        torch.save(basis, bfile.with_suffix(".tmp"))
        os.replace(bfile.with_suffix(".tmp"), bfile)
    while not bfile.exists():
        time.sleep(30)
    basis = {mm: (m.cuda(), V.cuda()) for mm, (m, V) in torch.load(bfile).items()}
    ids = sorted(p.name.split(".")[0] for p in root.glob("*.json"))[i0::n]
    for c, i in enumerate(ids):
        for mm, (mu, V) in basis.items():
            src, dst = root / f"{i}.{mm}.npy", root / f"{i}.{mm}.p{dim}.npy"
            if not src.exists() or dst.exists():
                continue
            step, tmp = 1 << 16, dst.with_suffix(".tmp")
            # sequential reads and writes of row blocks (a memmap on Lustre faults page by page)
            with open(src, "rb") as f, open(tmp, "wb") as g:
                major, _ = np.lib.format.read_magic(f)
                shape, _, _ = (np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0)(f)
                np.lib.format.write_array_header_1_0(g, {"descr": "<f2", "fortran_order": False,
                                                         "shape": (*shape[:3], dim)})
                N, C = int(np.prod(shape[:3])), shape[-1]
                for r in range(0, N, step):
                    m = min(step, N - r)
                    t = torch.from_numpy(np.frombuffer(f.read(m * C * 2), np.float16).reshape(m, C)).cuda().float()
                    keep = (t.abs().sum(1, keepdim=True) > 0).float()   # past the scan: stays zero
                    g.write((((t - mu) @ V) * keep).half().cpu().numpy().tobytes())
            os.replace(tmp, dst)
        if c % 20 == 0:
            print(f"[pca] {c + 1}/{len(ids)} {i}", flush=True)


if __name__ == "__main__":
    project(sys.argv[2]) if sys.argv[1] == "project" else main(sys.argv[1])
