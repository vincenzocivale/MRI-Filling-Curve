"""Which encoder blocks a decoder reads, chosen from the data (user, 24/10): tokviz showed the last blocks poor for the
dense tasks (schwannoma cells at 4 mm: AUROC 0.86 at block 3, 0.53 at the last one) and the fixed quarter taps keep the
last block and miss 4-6. Per encoder, once, from a few train / val volumes of each task:

  every block's tokens (and the patch embedding, -1) at 16 / 8 / 4 mm -> per cell its classes' voxel fractions;
  a ridge readout fitted on the train cells, scored on the val cells (mean AUROC over the classes and levels);
  greedy: per task and level the block that adds most to the val score, up to `max_blocks`, a second one only if it
  improves it (1 or 2 blocks per level: SegNet projects each level's width on its own); at the live levels (4 / 2 mm) only blocks up
  to the shallowest depth within live_tol (0.01) of the best single block, at most live_cap (9);
  per level also the curve: each curve read alone, and at the cached levels (16 / 8 mm) also their mean, the one with
  the best val AUROC (user, 24/10).

-> <out>/blocks.json: per task its blocks (the decoder: its last block = the deepest, the others its taps, the
encoder run only that deep, live levels included) and its curve per level ("curves" {task: {mm: curve | "mean"}});
the cache's blocks (their union, layout() order) and curves per cached level ("cache_curves", the union over tasks). Seconds of ridge
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
    cap, max_blocks = int(job.get("cap", 1000)), int(job.get("max_blocks", 2))
    out, res = Path(job["out"]), {"tasks": {}, "scores": {}}
    feats = {}   # (task, split, level, curve) -> [X [cells, n + 1, d] fp16, Y]
    cands = lambda mm: (["mean"] if mm >= 8 else []) + list(curves)
    for task, sp_ids in job["tasks"].items():
        for split, ids in sp_ids.items():
            used = 0   # volumes holding the target (CMB: many scans have no lesion), up to n[split] of the ids given
            for i in ids:
                if used >= int(job.get("n", {}).get(split, len(ids))):
                    break
                r = torch.load(Path(job["inputs"]) / f"{i}.pt", map_location="cpu", weights_only=False)
                if not bool((r["y"] > 0).any()) or bool((r["y"] == 255).all()):
                    continue
                used += 1
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
                    # each curve alone (one curve per level, chosen here: user, 24/10, a third of the live reading),
                    # and at the cached levels also their mean (the cache before 24/10); each curve read once
                    with torch.no_grad():   # all 17 blocks: small windows (OOM at 200k)
                        A = read(enc, curves, seed, xc, sp, k, mm, 30_000, 32, 8, taps=True, want=cands(mm))
                    for cv, a in A.items():
                        f = feats.setdefault((task, split, mm, cv), [[], []])
                        f[0].append(torch.from_numpy(a.reshape(-1, n + 1, d)[idx]))
                        f[1].append(fr.cpu())
                print(f"[blocks] {task} {split} {i}", flush=True)
    def select(D, mm):   # one level's blocks (and its single-block scores) on train / val cells D
        def score(blocks):
            cs = [order.index(b) for b in blocks]
            return _score(D["train"][0][:, cs].flatten(1).float(), D["train"][1],
                          D["val"][0][:, cs].flatten(1).float(), D["val"][1])
        single = {b: score([b]) for b in order}
        pool = order
        if mm < 8:   # live levels (user via Tullio, 24/10): the encoder only as deep as needed, the shallowest depth
            top = max(single.values())   # whose best single block is within `live_tol` of the best one, at most
            cap = int(job.get("live_cap", 9))   # live_cap (the argmax picked 14 / 15 by 0.003, noise: no cut)
            d = next((d for d in range(-1, cap + 1) if max(single[b] for b in order if b <= d) >= top - float(job.get("live_tol", 0.01))), cap)
            pool = [b for b in order if b <= d]
        chosen, best, steps = [], 0.5, []
        while len(chosen) < max_blocks:
            cand = {b: score(chosen + [b]) for b in pool if b not in chosen}
            if not cand:
                break
            b = max(cand, key=cand.get)
            if chosen and cand[b] <= best:   # a second block only if it improves the val AUROC
                break
            chosen, best = chosen + [b], cand[b]
            steps.append({"block": b, "val_auroc": round(best, 4)})
        return sorted(chosen), best, steps, single

    res["curves"] = {}
    for task in job["tasks"]:   # per level (user, 24/10: each level decides something else; no mean over levels)
        res["tasks"][task], res["scores"][task], res["curves"][task] = {}, {}, {}
        for mm in LEVELS:
            out_c = {}
            for cv in cands(mm):
                if (task, "train", mm, cv) not in feats or (task, "val", mm, cv) not in feats:
                    continue
                D = {s_: [torch.cat(v).cuda() for v in feats[(task, s_, mm, cv)]] for s_ in ("train", "val")}
                out_c[cv] = select(D, mm) + ([len(D["train"][1]), len(D["val"][1])],)
                del D
            if not out_c:
                continue
            cv = max(out_c, key=lambda c: out_c[c][1])   # the curve with the best val AUROC (ties: the mean)
            chosen, best, steps, single, cells = out_c[cv]
            res["tasks"][task][str(int(mm))] = chosen
            res["curves"][task][str(int(mm))] = cv
            res["scores"][task][str(int(mm))] = {"single": {str(b): round(v, 4) for b, v in single.items()}, "greedy": steps,
                                                 "cells": cells, "curve": cv,
                                                 "per_curve": {str(c): round(o[1], 4) for c, o in out_c.items()}}
            print(f"[blocks] {task} {mm:g} mm {cv}: " +
                  " ".join(f"{b}:{v:.3f}" for b, v in sorted(single.items())) + f" -> {chosen} ({best:.3f})" +
                  " curves " + " ".join(f"{c}:{o[1]:.3f}" for c, o in out_c.items()), flush=True)
    for task, tl in res["tasks"].items():   # levels not scored (cells under 8 voxels, no target): the nearest one's
        have = sorted(float(m) for m in tl)
        for mm in LEVELS:
            if have and str(int(mm)) not in tl:
                tl[str(int(mm))] = tl[str(int(min(have, key=lambda h: (abs(h - mm), -h))))]
        tc = res["curves"][task]
        for mm in (4, 2):
            tc.setdefault(str(mm), tc.get("4", tc.get("2", curves[0])))
        for mm in (16, 8):
            tc.setdefault(str(mm), "mean")
    res["cache"] = sorted({b for v in res["tasks"].values() for mm, bl in v.items() if float(mm) >= 8 for b in bl})   # (a set)
    res["layout"] = layout(res["cache"])
    res["cache_curves"] = {mm: sorted({tc[mm] for tc in res["curves"].values()}) for mm in ("16", "8")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "blocks.json").write_text(json.dumps(res, indent=1))
    print(f"[blocks] cache {res['cache']}, curves {res['cache_curves']} -> {out / 'blocks.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
