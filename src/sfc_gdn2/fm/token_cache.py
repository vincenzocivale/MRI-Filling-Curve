"""Encoder tokens of the segmentation volumes, read once (20/10): the decoder trains on them instead of running the
frozen encoder on every tile at three resolutions.

Per volume and level (16, 8, 4 mm) the tokens of the whole scan on nested grids: the cell is k4 = round(4 / spacing)
voxels per axis at 4 mm, 2 k4 at 8 mm, 4 k4 at 16 mm, so a tile whose corner sits on the 16 mm grid holds whole
cells of every level. Each token is the encoder's reading along each curve of the run (averaged), as SegNet's: at
every level as last block ++ the decoder's taps (the width SegNet.proj takes; each decoder keeps its channels). A scan
whose grid holds more than `nmax` tokens is read in windows of `block` cells per axis with `margin` cells of context
on every side, keeping each window's centre (TotalSeg whole-body at 4 mm: ~5e6 tokens). Stored as
<out>/<id>.<mm>.npy [gx, gy, gz, C] fp16 + <out>/<id>.json {k4, spacing, shape}.

    python -m sfc_gdn2.fm.token_cache <job.json>   {inputs, ids, out, checkpoint, args, shard, nmax, block, margin,
                                                    random: the seeded random encoder, for the random-encoder baseline}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

LEVELS = ((16.0, 4), (8.0, 2), (4.0, 1))   # (mm, cells of 4 mm per axis)


class TokenCache:
    """Crops of the cached tokens: tokens(id, level, corner voxel, cells per axis) -> [n, C] (zeros past the scan)."""

    def __init__(self, root: str | Path):
        self.root, self.ids, self._mm = Path(root), [], {}

    def meta(self, i: str) -> dict:
        return json.loads((self.root / f"{i}.json").read_text())

    def tokens(self, i: str, mm: float, corner, n, k) -> torch.Tensor:
        key = (i, mm)
        if key not in self._mm:
            self._mm[key] = np.load(self.root / f"{i}.{int(mm)}.npy", mmap_mode="r")
        a = self._mm[key]
        c0 = [int(c) // int(kk) for c, kk in zip(corner, k)]
        out = np.zeros((*n, a.shape[-1]), np.float16)
        src = tuple(slice(c, min(c + m, s)) for c, m, s in zip(c0, n, a.shape[:3]))
        out[tuple(slice(0, s.stop - s.start) for s in src)] = a[src]
        return torch.from_numpy(out.reshape(-1, a.shape[-1]))


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
            a = read(enc, curves, seed, x, sp, [kk * m for kk in k4], mm, job.get("nmax", 200_000),
                     job.get("block", 40), job.get("margin", 8), taps=True)
            np.save(out / f"{i}.{int(mm)}.npy", a)
        (out / f"{i}.json").write_text(json.dumps({"k4": list(k4), "spacing": sp.tolist(), "shape": list(x.shape[1:])}))
        print(f"[cache] {c + 1}/{len(ids)} {i} grid4 {np.load(out / f'{i}.4.npy', mmap_mode='r').shape[:3]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
