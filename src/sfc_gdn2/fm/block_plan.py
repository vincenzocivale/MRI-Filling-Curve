"""Which encoder blocks a decoder reads, chosen from the data (user, 24/10): tokviz showed the last blocks poor for the
dense tasks (schwannoma cells at 4 mm: AUROC 0.86 at block 3, 0.53 at the last one) and the fixed quarter taps keep the
last block and miss 4-6. Per encoder, once, from a few train / val volumes of each task:

  every block's tokens (and the patch embedding, -1) at 16 / 8 / 4 mm -> per cell its classes' voxel fractions;
  a ridge readout fitted on the train cells, scored on the val cells (mean AUROC over the classes and levels);
  greedy: per task the block that adds most to the val score, up to `max_blocks`, while it adds >= `min_gain`.

-> <out>/blocks.json: per task its blocks (the decoder: its last block = the deepest, the others its taps, the
encoder run only that deep, live levels included) and the cache's (their union, layout() order). Seconds of ridge
fits on the GPU after one encoder pass over 16 volumes.

    python -m sfc_gdn2.fm.block_plan <job.json>   {"checkpoint", "inputs", "tasks": {task: {"train": [ids], "val": [ids]}},
                                                   "out", "args" (the decoder's: taps, refine, curves ...)}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LEVELS = (16.0, 8.0, 4.0, 2.0)   # the cascade's levels; 4 / 2 mm read live in a window around the target (WIN cells)
WIN = 16


def layout(blocks) -> list[int]:
    """The order of a token row's blocks: the deepest (the encoder's last, final norm) ++ the others ascending."""
    b = sorted({int(x) for x in blocks})
    return [b[-1]] + b[:-1]


def columns(cache_blocks, blocks, dim: int) -> list[int]:
    """Columns of `blocks` (layout order) in a row of `cache_blocks` (layout order), dim per block."""
    lc = layout(cache_blocks)
    return [lc.index(b) * dim + c for b in layout(blocks) for c in range(dim)]


def apply(enc, blocks) -> None:
    """The encoder returns `blocks`: run up to the deepest (its output through the final norm), the others as taps."""
    lb = layout(blocks)
    enc.depth, enc.taps = lb[0] + 1, tuple(lb[1:])


def _auroc(s: torch.Tensor, y: torch.Tensor) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    r = s.argsort().argsort().double() + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _score(Xtr, Ytr, Xva, Yva) -> float:
    """Ridge readout (lambda: 0.1 of the mean eigenvalue) train -> val, mean AUROC over the classes present."""
    mu = Xtr.mean(0)
    A, B = Xtr - mu, Xva - mu
    G = A.T @ A
    G.diagonal().add_(0.1 * G.diagonal().mean())
    W = torch.linalg.solve(G, A.T @ (Ytr - Ytr.mean(0)))
    P = B @ W
    a = [_auroc(P[:, c], Yva[:, c] > 0) for c in range(Yva.shape[1])]
    a = [v for v in a if not np.isnan(v)]
    return float(np.mean(a)) if a else float("nan")


def _cells(y: torch.Tensor, k, k_classes: int, cap: int, rng) -> tuple:
    """Labels [*S] -> the sampled cells' indices (raster, grid of k-voxel cells) and their classes' fractions [n, K-1]:
    the cells holding a class (at most cap) against the cells around them (2 cells, as tokviz's cellprobe: random
    background is told apart by every block, the last one best) and a few random ones."""
    pad = [p for n, q in zip(reversed(y.shape), reversed(k)) for p in (0, -n % q)]
    fr = torch.stack([F.avg_pool3d(F.pad((y == c).float()[None, None], pad), k, k)[0, 0]
                      for c in range(1, k_classes)], -1)
    m = fr.sum(-1) > 0
    ring = F.max_pool3d(m.float()[None, None], 5, 1, 2)[0, 0].bool() & ~m
    fr = fr.flatten(0, 2)
    fg, near = m.flatten().nonzero()[:, 0].cpu().numpy(), ring.flatten().nonzero()[:, 0].cpu().numpy()
    far = (~m & ~ring).flatten().nonzero()[:, 0].cpu().numpy()
    pick = lambda a, n: rng.choice(a, min(n, len(a)), replace=False) if len(a) else a
    fg = pick(fg, cap)
    idx = np.concatenate([fg, pick(near, max(len(fg), 200)), pick(far, max(len(fg), 200) // 4)])
    return idx, fr[idx]


def main(path: str) -> None:
    from ..data.volume import patch_dims
    from .api import load_config
    from .sfc import SFC
    from .token_cache import fine_ok, read

    job = json.loads(Path(path).read_text())
    cfg = load_config("configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"], cfg["args"] = job["checkpoint"], dict(job.get("args", {}), blocks=None, taps=())
    net = SFC(cfg, "cuda").seg_net(2)[0].cuda().eval()
    enc, curves, seed = net.encoder, net.enc_curves, net.seed
    n, d = len(enc.blocks), enc.dim
    enc.taps, enc.depth = tuple(range(-1, n - 1)), None   # every block: row = last ++ (-1, 0, .., n - 2)
    order = [n - 1] + list(range(-1, n - 1))
    rng = np.random.default_rng(0)
    cap, max_blocks, min_gain = int(job.get("cap", 1000)), int(job.get("max_blocks", 2)), float(job.get("min_gain", 0.002))
    out, res = Path(job["out"]), {"tasks": {}, "scores": {}}
    feats = {}   # (task, split, level) -> [X [cells, n + 1, d] fp16, Y]
    for task, sp_ids in job["tasks"].items():
        for split, ids in sp_ids.items():
            for i in ids:
                r = torch.load(Path(job["inputs"]) / f"{i}.pt", map_location="cpu", weights_only=False)
                x, sp = r["x"].cuda(), r["ctx"][:3].float().cuda()
                y = (r["y"][0] if r["y"].dim() == 4 else r["y"]).cuda()
                K = int(job["k"][task])
                for mm in LEVELS:
                    k = list(patch_dims(sp.tolist(), mm))
                    if mm < 8 and not fine_ok(k):
                        continue
                    xc, yc = x, y
                    if mm < 8:   # as the cascade reads them: a window of WIN cells around a target voxel, not the scan
                        fgv = (y > 0).nonzero()
                        if len(fgv):
                            c = fgv[int(rng.integers(len(fgv)))].tolist()
                            lo = [max(0, min(cc - WIN * kk // 2, s_ - WIN * kk)) // kk * kk for cc, kk, s_ in zip(c, k, y.shape)]
                            sl = tuple(slice(a, a + WIN * kk) for a, kk in zip(lo, k))
                            xc, yc = x[(slice(None), *sl)], y[sl]
                    idx, fr = _cells(yc, k, K, cap, rng)
                    with torch.no_grad():
                        a = read(enc, curves, seed, xc, sp, k, mm, 30_000, 32, 8, taps=True)   # all 17 blocks: small windows (OOM at 200k)
                    a = torch.from_numpy(a.reshape(-1, n + 1, d)[idx])
                    f = feats.setdefault((task, split, mm), [[], []])
                    f[0].append(a)
                    f[1].append(fr.cpu())
                print(f"[blocks] {task} {split} {i}", flush=True)
    for task in job["tasks"]:   # per level (user, 24/10: each level decides something else; no mean over levels)
        res["tasks"][task], res["scores"][task] = {}, {}
        for mm in LEVELS:
            if (task, "train", mm) not in feats or (task, "val", mm) not in feats:
                continue
            D = {s_: [torch.cat(v).cuda() for v in feats[(task, s_, mm)]] for s_ in ("train", "val")}

            def score(blocks, D=D):
                cs = [order.index(b) for b in blocks]
                return _score(D["train"][0][:, cs].flatten(1).float(), D["train"][1],
                              D["val"][0][:, cs].flatten(1).float(), D["val"][1])
            single = {b: score([b]) for b in order}
            chosen, best, steps = [], 0.5, []
            while len(chosen) < max_blocks:
                cand = {b: score(chosen + [b]) for b in order if b not in chosen}
                b = max(cand, key=cand.get)
                if chosen and cand[b] - best < min_gain:
                    break
                chosen, best = chosen + [b], cand[b]
                steps.append({"block": b, "val_auroc": round(best, 4)})
            res["tasks"][task][str(int(mm))] = sorted(chosen)
            res["scores"][task][str(int(mm))] = {"single": {str(b): round(v, 4) for b, v in single.items()},
                                                 "greedy": steps, "cells": [len(D["train"][1]), len(D["val"][1])]}
            print(f"[blocks] {task} {mm:g} mm: " + " ".join(f"{b}:{v:.3f}" for b, v in sorted(single.items())) +
                  f" -> {sorted(chosen)} ({best:.3f})", flush=True)
            D.clear()
    res["cache"] = sorted({b for v in res["tasks"].values() for mm, bl in v.items() if float(mm) >= 8 for b in bl})
    res["layout"] = layout(res["cache"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "blocks.json").write_text(json.dumps(res, indent=1))
    print(f"[blocks] cache {res['cache']} -> {out / 'blocks.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
