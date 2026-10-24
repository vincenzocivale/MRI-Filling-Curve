"""The cascade's plan of a task, from its training labels and cached tokens only (user, 23/10: "le deve fare in automatico
se non ci sono quando si invia il job"): the finest level g_min from the targets' size, the starting level g0 from how
well each level's tokens tell pure from mixed cells. segrun.train computes it when an args.cascade_* is "auto" and keeps
it next to the results (<out>.plan.json).

- g_min: the largest of 16 / 8 / 4 / 2 mm not above half the volume-weighted 10th percentile of the components'
  equivalent diameters (the size holding 10 % of the target voxels: fragments of a few voxels in a mask do not count);
- g0: per cached level (16 / 8 mm) a softmax head [pure class k | mixed] on the train volumes' cached tokens, the same
  ones the decoder reads (block_plan's blocks and curve of this task per level, from <tok>/blocks.json; without it every
  column, the curves' mean); down the levels from each candidate g0 on the val volumes, each level's threshold the
  fewest refinements with mislabelled voxels <= (1 - r) of the target voxels reaching it, the cells still mixed at 8 mm
  assumed right below it; the cheapest g0 (cells read) whose ceiling Dice (finer levels assumed perfect) is within 1 %
  of the best.
  g0 = 4 mm is no longer a candidate (24/10): the cache holds 16 / 8 mm only (fine_live4) and starting at 4 mm means
  reading every 4 mm cell of the scan live; in the previous plans it always cost the full 4 mm grid (8 / 16 mm: 5-20 %)
  for no ceiling gain (vs, cmb: 0.9988 / 1.0 at 8 mm too).
Prototype and the 16-task study: sfc_runs/tokviz/plan.py (22-23/10)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .token_cache import level_file

LV = (16, 8)   # the cached levels


def _labels(inp: Path, tok: Path, i: str):
    r = torch.load(inp / f"{i}.pt", map_location="cpu", weights_only=False)
    y = np.asarray(r["y"])
    y = y[0] if y.ndim == 4 else y
    meta = json.loads((tok / f"{i}.json").read_text())
    return y, len(r["classes"]), np.array(meta["k4"]), np.array(meta["spacing"])


def _counts(y, K, k, G):
    """[G, K + 1] label counts of each cell of k voxels (255 ignored)."""
    yp = np.full(np.array(G) * k, 255, np.uint8)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(y.shape, yp.shape))
    yp[sl] = y[sl]
    blk = yp.reshape(G[0], k[0], G[1], k[1], G[2], k[2]).transpose(0, 2, 4, 1, 3, 5).reshape(*G, -1)
    return np.stack([(blk == c).sum(-1) for c in range(K + 1)], -1).astype(np.float32)


def _up(c, G):   # counts onto the parent grid G (factor 2, clipped)
    out = np.zeros((*G, c.shape[-1]), np.float32)
    np.add.at(out, np.ix_(*[np.minimum(np.arange(n) // 2, g - 1) for n, g in zip(c.shape[:3], G)]), c)
    return out


def _purity(c):  # k if only class k present, K + 1 = mixed; empty cells (all ignored) -> 0
    present = (c > 0).sum(-1)
    lab = np.where(present == 1, c.argmax(-1), c.shape[-1])
    lab[present == 0] = 0
    return lab


def _gmin(diam):
    d = np.array(diam) if diam else np.array([[0.0, 1.0]])
    o = np.argsort(d[:, 0])
    cw = np.cumsum(d[o, 1]) / d[:, 1].sum()
    p10 = float(d[o, 0][min(int(np.searchsorted(cw, 0.1)), len(o) - 1)])
    med = float(d[o, 0][min(int(np.searchsorted(cw, 0.5)), len(o) - 1)])
    return next((g for g in (16, 8, 4, 2) if g <= p10 / 2), 2), p10, med


def _reader(tok: Path, task: str | None):
    """(volume, level) -> its cached tokens [*G, C] as the task's decoder reads them: its blocks' columns, its curve."""
    bj = tok / "blocks.json"
    pl = json.loads(bj.read_text()) if bj.exists() and task else None
    bl = {int(float(m)): v for m, v in pl["tasks"][task].items()} if pl else {}
    curve = {g: ((pl or {}).get("curves") or {}).get(task, {}).get(str(g), "mean") for g in LV}

    def load(i, g):
        from .block_plan import columns
        t = np.load(level_file(tok, i, g, curve[g]))
        return t[..., columns(pl["cache"], bl[g], t.shape[-1] // len(pl["cache"]))] if g in bl else t
    return load, curve, bl


def plan(items: list, inp: str | Path, tok: str | Path, r_keep: float = 0.995, seed: int = 0, task: str | None = None) -> dict:
    """items: the job's items ({"id", "split"}); inp: segdec inputs (<id>.pt with the labels); tok: the token cache;
    task: its blocks / curves in <tok>/blocks.json."""
    inp, tok = Path(inp), Path(tok)
    rng = np.random.default_rng(seed)
    load, curve, blocks = _reader(tok, task)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok = [it for it in items if (tok / f"{it['id']}.json").exists() and (inp / f"{it['id']}.pt").exists()]
    tr = [it["id"] for it in ok if it["split"] == "train"]
    va = [it["id"] for it in ok if it["split"] == "val"]
    tr, va = list(rng.permutation(tr)[:40]), list(rng.permutation(va)[:15])
    X = {g: [] for g in LV}
    Y = {g: [] for g in LV}
    diam, K = [], None
    for n, i in enumerate(tr):
        y, K, k4, sp = _labels(inp, tok, i)
        if n < 15:   # component sizes: (diameter, volume) of each connected component of each class
            for c in range(1, K + 1):
                lab, nc = ndimage.label(y == c)
                if nc:
                    v = np.bincount(lab.ravel())[1:] * float(np.prod(sp))
                    diam += list(zip((6 * v / np.pi) ** (1 / 3), v))
        ts = {g: load(i, g) for g in LV}
        cg = {8: _counts(y, K, 2 * k4, ts[8].shape[:3])}
        cg[16] = _up(cg[8], ts[16].shape[:3])
        for g in LV:
            lab = _purity(cg[g])
            nz = np.abs(ts[g][..., :8].astype(np.float32)).sum(-1) > 0
            keep = nz & ((lab == cg[g].shape[-1]) | (rng.random(lab.shape) < 30000 / max(lab.size, 1)))
            X[g].append(ts[g][keep])
            Y[g].append(lab[keep])
    gmin, p10, med = _gmin(diam)
    heads = {}
    for g in LV:   # softmax head [pure k | mixed] on the task's blocks (1-2 x d: no reduction), class weights from train
        x = torch.from_numpy(np.concatenate(X[g])).to(dev).float()
        yy = torch.from_numpy(np.concatenate(Y[g]).astype(np.int64)).to(dev)
        X[g] = Y[g] = None
        mu = x.mean(0)
        sd = (x - mu).std(0) + 1e-4
        z = (x - mu) / sd
        C = K + 2
        freq = torch.bincount(yy, minlength=C).float()
        w = (freq.sum() / (C * freq.clamp(min=1))).clamp(max=100.0)
        W = torch.zeros(z.shape[1], C, device=dev, requires_grad=True)
        b = torch.zeros(C, device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200)

        def clo(z=z, yy=yy, w=w, W=W, b=b, opt=opt):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(z @ W + b, yy, weight=w) + 1e-3 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(clo)
        heads[g] = (mu, sd, W.detach(), b.detach())
        del x, z

    V = []
    for i in va:   # val volumes: each level's class probabilities and label counts
        y, _, k4, _ = _labels(inp, tok, i)
        P = {}
        for g in LV:
            t = load(i, g)
            mu, sd, W, b = heads[g]
            x = torch.from_numpy(t.reshape(-1, t.shape[-1]))
            with torch.no_grad():
                p = np.concatenate([torch.softmax(((ch.to(dev).float() - mu) / sd) @ W + b, 1).cpu().numpy()
                                    for ch in x.split(50000)]).reshape(*t.shape[:3], -1)
            nz = np.abs(t[..., :8].astype(np.float32)).sum(-1) > 0
            p[~nz] = 0
            p[~nz, 0] = 1
            P[g] = p
        c8 = _counts(y, K, 2 * k4, P[8].shape[:3])
        V.append({"P": P, "c": {8: c8, 16: _up(c8, P[16].shape[:3])}})

    def down(v, thr, g0):   # candidates at each level under the thresholds of the coarser ones
        P, cand, out = v["P"], np.ones(v["P"][g0].shape[:3], bool), {}
        for g in [x for x in LV if x <= g0]:
            if g < g0:
                prev = out[g * 2][1]
                cand = prev[np.ix_(*[np.minimum(np.arange(n) // 2, m - 1) for n, m in zip(P[g].shape[:3], prev.shape)])]
            refine = cand & (P[g][..., -1] > thr[g])
            out[g] = (cand, refine)
        return out

    def ceiling(thr, g0):
        acc = np.zeros((K + 1, 3))
        read = full = 0
        for v in V:
            lv = down(v, thr, g0)
            for g, (cand, refine) in lv.items():
                read += int(cand.sum())
                fill = cand & ~refine
                k = v["P"][g][..., :-1].argmax(-1)[fill]
                cc = v["c"][g][fill]
                hit = cc[np.arange(len(k)), k]
                for cl in range(1, K + 1):
                    m = k == cl
                    acc[cl] += [hit[m].sum(), (cc[m].sum(1) - hit[m]).sum(), cc[~m, cl].sum()]
                if g == LV[-1]:   # still mixed at the last cached level: the finer levels assumed right
                    acc[1:, 0] += v["c"][g][refine][:, 1:].sum(0)
            full += v["P"][LV[-1]][..., 0].size
        d = [2 * a / max(2 * a + b + c, 1) for a, b, c in acc[1:] if a + c > 0]
        return float(np.mean(d)) if d else float("nan"), read / max(full, 1)

    def choose(g0):
        thr = {g: 1.1 for g in LV}
        for g in [x for x in LV if x <= g0]:
            grid = np.unique(np.quantile(np.concatenate([v["P"][g][..., -1].ravel() for v in V]), np.linspace(0, 1, 60)))
            best = grid[0] - 1e-6
            for t in grid[::-1]:
                thr[g] = t
                err = tgt = 0.0
                for v in V:
                    cand, _ = down(v, thr, g0)[g]
                    fill = cand & (v["P"][g][..., -1] <= t)
                    k = v["P"][g][..., :-1].argmax(-1)[fill]
                    cc = v["c"][g][fill]
                    err += float((cc.sum(1) - cc[np.arange(len(k)), k]).sum())
                    tgt += float(v["c"][g][cand][:, 1:].sum())
                if err <= (1 - r_keep) * max(tgt, 1):
                    best = t
                    break
            thr[g] = best
        return thr

    per = {g0: ceiling(choose(g0), g0) for g0 in LV}
    top = max(d for d, _ in per.values())
    g0 = min((g for g in per if per[g][0] >= top - 0.01), key=lambda g: per[g][1])
    return {"g0": float(g0), "gmin": float(min(gmin, g0)), "diam_p10_mm": round(p10, 2), "diam_median_mm": round(med, 2),
            "per_g0": {str(g): {"val_ceiling_dice": round(d, 4), "cells_vs_full_8mm": round(c, 4)} for g, (d, c) in per.items()},
            "blocks": {str(g): b for g, b in blocks.items()}, "curves": {str(g): c for g, c in curve.items()},
            "volumes": {"train": len(tr), "val": len(va)}, "r_keep": r_keep}
