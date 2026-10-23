"""Where to read at 4 mm? A small head on the FROZEN encoder's 8 mm tokens predicts which 8 mm cells hold a class boundary
(the BATS target: a cell with >= 2 labels, background included; arXiv 2607.26829), and is judged against the other
criteria at the same token count - no segmentation decoder is trained.

Policy: read the scan at 8 mm (uniform), split into its 8 children (4 mm) the top share f of the cells by a score. Scores:
  head      the trained head (token, token minus its 3x3x3 neighbours' mean, cell mean / std / range / entropy);
  edge      the norm of token minus neighbours' mean (SegNet.to_fine's refine_frac, no training);
  ent / std / rng   intensity entropy / std / range of the cell (the criteria of prune_eval.py);
  random    the control;  oracle   the true label of the cell (the bound).
Per f, pooled over the test volumes: tokens as a share of uniform 4 mm, labelled voxels in leaves holding >= 2 labels
(`mixed`, lower is better; uniform 8 mm and 4 mm are f = 0 and 1), and the head's precision / recall on the boundary cells.
Splits as the benchmark's (bench/split.py): the head is fitted on train volumes, the epoch chosen on val (average
precision), the test volumes are read once.

    python scripts/learn_split.py <run dir> --out $SCRATCH/learn_split.json [--n-train 150 --n-val 40 --n-test 100]
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from prune_eval import CACHE, LEVELS, flat, level_stats
from torch import nn

import sfc_gdn2.fm.sfc as m
from sfc_gdn2.apt import base_dims
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.data.volume import patch_grid, patchify
from sfc_gdn2.io import load_yaml

FRACS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
CRIT = ("head", "edge", "ent", "std", "rng", "random", "oracle")


def volume(f, net, dev):
    """Cached scan -> (level stats [16, 8, 4 mm], tokens [g1, 512] of the 8 mm cells, grid g1)."""
    r = torch.load(f, map_location="cpu", weights_only=False)
    x, y = r["x"][0].float().to(dev), r["y"].to(dev)
    sp = r["ctx"][:3]
    k0 = base_dims(sp.tolist(), 16.0, LEVELS)
    pad = [p for n, q in zip(reversed(x.shape), reversed(k0)) for p in (0, -n % q)]
    ones = F.pad(torch.ones_like(x), pad)
    x, y = F.pad(x, pad), F.pad(y, pad, value=255)
    st = level_stats(x, y, ones, k0)
    k1 = tuple(q // 2 for q in k0)
    g1 = patch_grid(tuple(x.shape), k1).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bi = net.extract({"patches": [patchify(x, k1)], "grid": g1[None], "k": torch.tensor([k1], device=dev),
                          "spacing": sp[None].float().to(dev)})["bi"][0]
    tok = bi[: int(g1.prod()), : net.encoder.dim].float()
    return st, tok.view(*g1.tolist(), -1)


def features(st, tok):
    """[n, F] per 8 mm cell: token, token - mean of its 3x3x3 neighbours, cell mean / std / range / entropy."""
    t = tok.permute(3, 0, 1, 2)[None]
    edge = t - F.avg_pool3d(F.pad(t, (1,) * 6, mode="replicate"), 3, 1)
    s = st[1]
    hand = torch.stack([s["mean"], s["std"], s["rng"], s["ent"]], -1)
    return torch.cat([tok, edge[0].permute(1, 2, 3, 0), hand], -1).flatten(0, 2), edge[0].norm(dim=0).flatten()


def ap(score, target):
    """Average precision."""
    o = score.argsort(descending=True)
    t = target[o].float()
    prec = t.cumsum(0) / torch.arange(1, len(t) + 1, device=t.device)
    return float((prec * t).sum() / t.sum().clamp_min(1))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("run")
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--n-train", type=int, default=150)
    ap_.add_argument("--n-val", type=int, default=40)
    ap_.add_argument("--n-test", type=int, default=100)
    ap_.add_argument("--epochs", type=int, default=30)
    a = ap_.parse_args()
    dev = torch.device("cuda")
    w = m.SFC({"checkpoint": a.run, "args": {"patch_mm": 8, "chunk": 131072}}, "cuda")
    net = w.seg_net(51, pretrained=True)[0].to(dev).eval()
    net.encoder.patch_embed.patch_mm = 8.0

    bench = load_yaml("configs/bench/bench.yaml")
    tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
    vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(bench["out_dir"]) / "splits.csv")
    items = [i for i in bs.task_items(tasks, vols, splits, 0)["totalseg"] if (CACHE / f"{i['id']}.pt").exists()]
    pick = {}
    for s, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        ids = [i["id"] for i in items if i["split"] == s]
        pick[s] = random.Random(0).sample(ids, min(n, len(ids)))
    print({s: len(v) for s, v in pick.items()}, flush=True)

    def collect(ids, cap):
        X, Y = [], []
        for i in ids:
            st, tok = volume(CACHE / f"{i}.pt", net, dev)
            x, edge = features(st, tok)
            ok = st[1]["valid"].flatten()
            y = st[1]["mixed"].flatten()
            idx = ok.nonzero()[:, 0]
            idx = idx[torch.randperm(len(idx), device=dev)[:cap]]
            X.append(x[idx].half().cpu())
            Y.append(y[idx].cpu())
        return torch.cat(X), torch.cat(Y)

    Xtr, Ytr = collect(pick["train"], 20000)
    Xva, Yva = collect(pick["val"], 20000)
    print(f"cells: train {len(Ytr)} ({float(Ytr.float().mean()):.3f} boundary), val {len(Yva)}", flush=True)
    head = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev), Yva.to(dev)
    best = (-1.0, None, 0)
    for ep in range(1, a.epochs + 1):
        head.train()
        for b in torch.randperm(len(Xtr), device=dev).split(8192):
            loss = F.binary_cross_entropy_with_logits(head(Xtr[b].float())[:, 0], Ytr[b].float(),
                                                      pos_weight=torch.tensor(2.0, device=dev))   # recall over precision
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = ap(head(Xva.float())[:, 0], Yva)
        if v > best[0]:
            best = (v, {k: t.clone() for k, t in head.state_dict().items()}, ep)
    head.load_state_dict(best[1])
    print(f"head: val AP {best[0]:.3f} at epoch {best[2]} (boundary share {float(Yva.float().mean()):.3f} = AP of chance)", flush=True)
    del Xtr, Ytr, Xva, Yva

    acc = {}
    gen = torch.Generator(dev).manual_seed(0)
    for n, i in enumerate(pick["test"]):
        st, tok = volume(CACHE / f"{i}.pt", net, dev)
        x, edge = features(st, tok)
        g1 = st[1]["mean"].shape
        valid1, mixed1, nlab1 = (st[1][k].flatten() for k in ("valid", "mixed", "nlab"))
        valid2, mixed2, nlab2 = st[2]["valid"], st[2]["mixed"], st[2]["nlab"]
        total_lab = int(nlab2.sum())
        if total_lab == 0:
            continue
        with torch.no_grad():
            p = head(x)[:, 0]
        noise = torch.rand(len(p), device=dev, generator=gen) * 1e-4
        scores = {"head": p, "edge": edge, "ent": st[1]["ent"].flatten(), "std": st[1]["std"].flatten(),
                  "rng": st[1]["rng"].flatten(), "random": noise, "oracle": mixed1.float()}
        nv = int(valid1.sum())
        for crit in CRIT:
            s = scores[crit].float() + (noise if crit != "random" else 0)
            s = torch.where(valid1, s, s.new_full((), -float("inf")))
            order = s.argsort(descending=True)
            for f in (0.0, *FRACS, 1.0):
                sel = torch.zeros(len(s), dtype=torch.bool, device=dev)
                sel[order[: round(f * nv)]] = True
                m2 = sel.view(g1)
                for ax in range(3):
                    m2 = m2.repeat_interleave(2, ax)
                leaf1 = valid1 & ~sel
                tok_n = int(leaf1.sum()) + int((valid2 & m2).sum())
                mix = int(nlab1.flatten()[leaf1 & mixed1].sum()) + int(nlab2[m2 & mixed2].sum())
                tp = int((sel & mixed1 & valid1).sum())
                r = acc.setdefault((crit, f), dict(tok=0, mix=0, lab=0, sel=0, tp=0, pos=0))
                r["tok"] += tok_n
                r["mix"] += mix
                r["lab"] += total_lab
                r["sel"] += int(sel.sum())
                r["tp"] += tp
                r["pos"] += int((mixed1 & valid1).sum())
        if (n + 1) % 20 == 0:
            print(f"{n + 1} test volumes", flush=True)

    uni4 = acc[("oracle", 1.0)]["tok"]
    res = []
    print(f"\n{'f':>5} " + "".join(f"{c:>20}" for c in CRIT) + "\n      (tokens/uni4, mixed) ; head precision / recall on boundary cells below")
    for f in (0.0, *FRACS, 1.0):
        row = {c: (acc[(c, f)]["tok"] / uni4, acc[(c, f)]["mix"] / acc[(c, f)]["lab"]) for c in CRIT}
        h = acc[("head", f)]
        pr = (h["tp"] / max(h["sel"], 1), h["tp"] / max(h["pos"], 1))
        res.append({"f": f, "policies": row, "head_precision_recall": pr})
        print(f"{f:>5} " + "".join(f"{row[c][0]:>9.3f}/{row[c][1]:<10.3f}" for c in CRIT) + f"  P {pr[0]:.2f} R {pr[1]:.2f}")
    Path(a.out).write_text(json.dumps({"val_ap": best[0], "epoch": best[2], "n": {s: len(v) for s, v in pick.items()},
                                       "results": res}, indent=1))


if __name__ == "__main__":
    main()
