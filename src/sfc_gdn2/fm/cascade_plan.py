"""The cascade's plan of a task, from its training labels and cached tokens only (user, 23/10: "le deve fare in automatico
se non ci sono quando si invia il job"): the finest level g_min from the targets' size, the starting level g0 from how
well each level's tokens tell pure from mixed cells. segrun.train computes it when an args.cascade_* is "auto" and keeps
it next to the results (<out>.plan.json).

- g_min: the largest of 16 / 8 / 4 / 2 mm not above half the volume-weighted 10th percentile of the components'
  equivalent diameters (the size holding 10 % of the target voxels: fragments of a few voxels in a mask do not count);
- g0: per level (16 / 8 / 4 mm) a softmax head [pure class k | mixed] on the cached tokens of the train volumes; down the
  levels from each candidate g0 on the val volumes, each level's threshold the fewest refinements with mislabelled voxels
  <= (1 - r) of the target voxels reaching it; the cheapest g0 (tokens read) whose ceiling Dice (voxel head assumed
  perfect) is within 1 % of the best.
Prototype and the 16-task study: sfc_runs/tokviz/plan.py (22-23/10)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

NAME = {16: "16", 8: "8.p512", 4: "4.p512"}


def _labels(inp: Path, tok: Path, i: str):
    r = torch.load(inp / f"{i}.pt", map_location="cpu", weights_only=False)
    y = np.asarray(r["y"])
    y = y[0] if y.ndim == 4 else y
    meta = json.loads((tok / f"{i}.json").read_text())
    return y, len(r["classes"]), np.array(meta["k4"]), np.array(meta["spacing"])


def _counts(y, K, k4, G4):
    """[G4, K + 1] label counts of each 4 mm cell (255 ignored)."""
    yp = np.full(np.array(G4) * k4, 255, np.uint8)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(y.shape, yp.shape))
    yp[sl] = y[sl]
    blk = yp.reshape(G4[0], k4[0], G4[1], k4[1], G4[2], k4[2]).transpose(0, 2, 4, 1, 3, 5).reshape(*G4, -1)
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


def plan(items: list, inp: str | Path, tok: str | Path, r_keep: float = 0.995, seed: int = 0) -> dict:
    """items: the job's items ({"id", "split"}); inp: segdec inputs (<id>.pt with the labels); tok: the token cache."""
    inp, tok = Path(inp), Path(tok)
    rng = np.random.default_rng(seed)
    ok = [it for it in items if (tok / f"{it['id']}.json").exists() and (inp / f"{it['id']}.pt").exists()]
    tr = [it["id"] for it in ok if it["split"] == "train"]
    va = [it["id"] for it in ok if it["split"] == "val"]
    tr, va = list(rng.permutation(tr)[:40]), list(rng.permutation(va)[:15])
    X = {g: [] for g in NAME}
    Y = {g: [] for g in NAME}
    diam, K = [], None
    for n, i in enumerate(tr):
        y, K, k4, sp = _labels(inp, tok, i)
        if n < 15:   # component sizes: (diameter, volume) of each connected component of each class
            for c in range(1, K + 1):
                lab, nc = ndimage.label(y == c)
                if nc:
                    v = np.bincount(lab.ravel())[1:] * float(np.prod(sp))
                    diam += list(zip((6 * v / np.pi) ** (1 / 3), v))
        ts = {g: np.load(tok / f"{i}.{NAME[g]}.npy", mmap_mode="r") for g in NAME}
        cg = {4: _counts(y, K, k4, ts[4].shape[:3])}
        cg[8] = _up(cg[4], ts[8].shape[:3])
        cg[16] = _up(cg[8], ts[16].shape[:3])
        for g in NAME:
            lab = _purity(cg[g])
            nz = np.abs(np.asarray(ts[g][..., :8], np.float32)).sum(-1) > 0
            keep = nz & ((lab == cg[g].shape[-1]) | (rng.random(lab.shape) < 30000 / max(lab.size, 1)))
            X[g].append(np.asarray(ts[g][keep], np.float16))
            Y[g].append(lab[keep])
    gmin, p10, med = _gmin(diam)
    heads = {}
    for g in NAME:   # softmax head [pure k | mixed], PCA 512 for the 16 mm tokens, class weights from train
        x = torch.from_numpy(np.concatenate(X[g]).astype(np.float32))
        yy = torch.from_numpy(np.concatenate(Y[g]).astype(np.int64))
        X[g] = Y[g] = None
        mu = x.mean(0)
        proj = torch.pca_lowrank(x - mu, q=512, center=False)[2] if x.shape[1] > 512 else None
        z = (x - mu) @ proj if proj is not None else x - mu
        sd = z.std(0) + 1e-4
        z = z / sd
        C = K + 2
        freq = torch.bincount(yy, minlength=C).float()
        w = (freq.sum() / (C * freq.clamp(min=1))).clamp(max=100.0)
        W = torch.zeros(z.shape[1], C, requires_grad=True)
        b = torch.zeros(C, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200)

        def clo(z=z, yy=yy, w=w, W=W, b=b, opt=opt):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(z @ W + b, yy, weight=w) + 1e-3 * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(clo)
        heads[g] = (mu, proj, sd, W.detach(), b.detach())

    V = []
    for i in va:   # val volumes: each level's class probabilities and label counts
        y, _, k4, _ = _labels(inp, tok, i)
        P = {}
        for g, name in NAME.items():
            t = np.load(tok / f"{i}.{name}.npy", mmap_mode="r")
            mu, proj, sd, W, b = heads[g]
            x = torch.from_numpy(np.asarray(t, np.float32).reshape(-1, t.shape[-1]))
            ps = [torch.softmax((((ch - mu) @ proj if proj is not None else ch - mu) / sd) @ W + b, 1).numpy()
                  for ch in x.split(50000)]
            p = np.concatenate(ps).reshape(*t.shape[:3], -1)
            nz = np.abs(np.asarray(t[..., :8], np.float32)).sum(-1) > 0
            p[~nz] = 0
            p[~nz, 0] = 1
            P[g] = p
        c4 = _counts(y, K, k4, P[4].shape[:3])
        c8 = _up(c4, P[8].shape[:3])
        V.append({"P": P, "c": {4: c4, 8: c8, 16: _up(c8, P[16].shape[:3])}})

    def down(v, thr, g0):   # candidates at each level under the thresholds of the coarser ones
        P, cand, out = v["P"], np.ones(v["P"][g0].shape[:3], bool), {}
        for g in [x for x in (16, 8, 4) if x <= g0]:
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
                if g == 4:
                    acc[1:, 0] += v["c"][4][refine][:, 1:].sum(0)
            full += v["P"][4][..., 0].size
        d = [2 * a / max(2 * a + b + c, 1) for a, b, c in acc[1:] if a + c > 0]
        return float(np.mean(d)) if d else float("nan"), read / max(full, 1)

    def choose(g0):
        thr = {16: 1.1, 8: 1.1, 4: 1.1}
        for g in [x for x in (16, 8, 4) if x <= g0]:
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

    per = {g0: ceiling(choose(g0), g0) for g0 in (16, 8, 4)}
    top = max(d for d, _ in per.values())
    g0 = min((g for g in per if per[g][0] >= top - 0.01), key=lambda g: per[g][1])
    return {"g0": float(g0), "gmin": float(min(gmin, g0)), "diam_p10_mm": round(p10, 2), "diam_median_mm": round(med, 2),
            "per_g0": {str(g): {"val_ceiling_dice": round(d, 4), "tokens_vs_full_4mm": round(c, 4)} for g, (d, c) in per.items()},
            "volumes": {"train": len(tr), "val": len(va)}, "r_keep": r_keep}
