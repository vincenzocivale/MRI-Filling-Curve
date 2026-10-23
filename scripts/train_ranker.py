"""Split ranker v1: which 8 mm cells to read at 4 mm. A head on the frozen encoder's 8 / 16 mm tokens (cache_tokens.py) is
trained with the labels as privileged information (the gain of splitting a cell, see cache_tokens.py) and judged at fixed
budgets: split the top share f of the cells by score.

Variants (all on the same cache and splits; the epoch is chosen on val, test is scored once):
  mlp_bce        per-cell MLP on token, token - neighbours' mean, hand features; BCE on "the cell holds >= 2 labels"
                 (learn_split.py's head);
  conv_bce       3D ConvNeXt-style head (3x3x3 depthwise blocks) on token, 16 mm parent token, hand features; BCE;
  conv_reg       same head, Huber regression of log(1 + class-weighted gain / mean gain of the scan);
  conv_list      same head, ListNet: softmax of the scores against the gains normalised in the scan;
  conv_regl      regression + 0.5 x ListNet;
  conv_regl_no16 as conv_regl without the 16 mm parent token (what the context adds).
Baselines: random, oracle (true class-weighted gain), oracle_mixed (true binary target).
Reported per f: tokens as a share of uniform 4 mm, `mixed` (labelled voxels in leaves with >= 2 labels, lower is better),
class-weighted gain captured (share of the gain of all cells that the selected cells hold), plain gain captured, and per
class the share of its gain captured.

    python scripts/train_ranker.py --cache $SCRATCH/ranker_cache --out $SCRATCH/ranker_v1.json [--epochs 40]
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

FRACS = (0.05, 0.1, 0.2, 0.3, 0.5)
VARIANTS = {"mlp_bce": ("bce", dict(ctx16=False, blocks=0, edge=True)),
            "conv_bce": ("bce", dict()), "conv_reg": ("reg", dict()), "conv_list": ("list", dict()),
            "conv_regl": ("regl", dict()), "conv_regl_no16": ("regl", dict(ctx16=False))}


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.dw, self.norm = nn.Conv3d(w, w, 3, padding=1, groups=w), nn.LayerNorm(w)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))

    def forward(self, h):   # [gx, gy, gz, w]
        t = self.dw(h.permute(3, 0, 1, 2)[None])[0].permute(1, 2, 3, 0)
        return h + self.mlp(self.norm(t))


class Ranker(nn.Module):
    def __init__(self, ctx16=True, blocks=3, width=128, edge=False):
        super().__init__()
        self.ctx16, self.edge = ctx16, edge
        din = 512 * (1 + ctx16 + edge) + 4
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(blocks))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, d):
        g1, parts = d["g1"], [d["tok8"]]
        if self.edge:
            t = d["tok8"].view(*g1, -1).permute(3, 0, 1, 2)[None]
            parts.append((t - F.avg_pool3d(F.pad(t, (1,) * 6, mode="replicate"), 3, 1))[0].permute(1, 2, 3, 0).flatten(0, 2))
        if self.ctx16:   # the parent cell's token, repeated on its 8 children
            u = d["tok16"].view(*d["g0"], -1)
            for ax in range(3):
                u = u.repeat_interleave(2, ax)
            parts.append(u.flatten(0, 2))
        parts.append(d["hand"])
        h = self.inp(torch.cat(parts, -1)).view(*g1, -1)
        for b in self.blocks:
            h = b(h)
        return self.out(h.flatten(0, 2))[:, 0]


def to_dev(d, dev):
    o = {k: (v.to(dev).float() if k in ("tok8", "tok16", "hand", "gain", "gain_w") else v.to(dev) if torch.is_tensor(v) else v)
         for k, v in d.items() if k not in ("coo", "classes", "id", "split")}
    if "coo" in d:
        o["coo"] = tuple(t.to(dev).long() for t in d["coo"])
    return o


TRAIN_KEYS = ("tok8", "tok16", "hand", "valid1", "mixed1", "gain_w")


def crops(d, rng, side=38, n=2):
    """A training volume with more than side^3 cells -> n random blocks of at most side cells per axis (even sizes and
    starts, so the 16 mm grid stays aligned); smaller ones are kept whole. Only what the losses read is kept (RAM)."""
    g1 = d["g1"]
    if g1[0] * g1[1] * g1[2] <= side ** 3:
        return [{**{k: d[k] for k in TRAIN_KEYS}, "g1": g1, "g0": d["g0"]}]
    out = []
    for _ in range(n):
        size = [min(g, side) for g in g1]
        start = [2 * rng.randrange(0, (g - sz) // 2 + 1) for g, sz in zip(g1, size)]
        sl = tuple(slice(a, a + sz) for a, sz in zip(start, size))
        sl0 = tuple(slice(a // 2, (a + sz) // 2) for a, sz in zip(start, size))
        c = {"g1": size, "g0": [sz // 2 for sz in size]}
        for k in TRAIN_KEYS:
            t = d[k]
            sh, cut = (d["g0"], sl0) if k == "tok16" else (g1, sl)
            c[k] = t.view(*sh, *t.shape[1:])[cut].flatten(0, 2).clone()
        out.append(c)
    return out


def loss_fn(kind, s, d):
    v = d["valid1"]
    s = s[v]
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(s, d["mixed1"][v].float(), pos_weight=torch.tensor(2.0, device=s.device))
    g = d["gain_w"][v]
    tot = g.sum()
    if tot <= 0:
        return s.sum() * 0
    reg = F.smooth_l1_loss(s, torch.log1p(g / g.mean()))
    lst = -(g / tot * F.log_softmax(s, 0)).sum()
    return {"reg": reg, "list": lst, "regl": reg + 0.5 * lst}[kind]


@torch.no_grad()
def evaluate(score_fn, vols, dev, B):
    acc = {f: dict(tok=0, mix=0, lab=0, gw=0.0, gwt=0.0, g=0.0, gt=0.0, ratio=[], cs=torch.zeros(B, dtype=torch.float64),
                   ct=torch.zeros(B, dtype=torch.float64)) for f in FRACS}
    tok4 = 0
    for raw in vols:
        d = to_dev(raw, dev)
        v = d["valid1"]
        nv = int(v.sum())
        if nv == 0:
            continue
        s = score_fn(d).float().masked_fill(~v, -float("inf"))
        order = s.argsort(descending=True)
        tok4 += int(d["ntok2"][v].sum())
        cell, cls, val = d["coo"]
        for f in FRACS:
            sel = torch.zeros(len(s), dtype=torch.bool, device=dev)
            sel[order[: round(f * nv)]] = True
            a = acc[f]
            a["tok"] += int((v & ~sel).sum()) + int(d["ntok2"][sel].sum())
            a["mix"] += int(d["lab1"][v & d["mixed1"] & ~sel].sum()) + int(d["lab2mixed"][sel].sum())
            a["lab"] += int(d["lab1"].sum())
            a["gw"] += float(d["gain_w"][sel].sum())
            a["gwt"] += float(d["gain_w"].sum())
            a["g"] += float(d["gain"][sel].sum())
            a["gt"] += float(d["gain"].sum())
            if float(d["gain_w"].sum()) > 0:
                a["ratio"].append(float(d["gain_w"][sel].sum() / d["gain_w"].sum()))
            B_ = a["cs"].shape[0]
            m, pos = sel[cell], val.clamp_min(0).double()   # per class: only the voxels a split turns right (the sign of the
            a["cs"] += torch.bincount(cls[m], weights=pos[m], minlength=B_).cpu()   # majority-vote attribution is an artefact)
            a["ct"] += torch.bincount(cls, weights=pos, minlength=B_).cpu()
    return {f: {"tokens": a["tok"] / tok4, "mixed": a["mix"] / max(a["lab"], 1), "gain_w_captured": a["gw"] / max(a["gwt"], 1e-9),
                "gain_w_captured_mean_scan": sum(a["ratio"]) / max(len(a["ratio"]), 1), "gain_captured": a["g"] / max(a["gt"], 1e-9),
                "cls_sel": a["cs"], "cls_tot": a["ct"]} for f, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--only", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, help="volumes per split (smoke test)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    vols, rng_load = {"train": [], "val": [], "test": []}, random.Random(0)
    for f in sorted(Path(a.cache).glob("*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if a.limit and len(vols[d["split"]]) >= a.limit:
            continue
        vols[d["split"]] += crops(d, rng_load) if d["split"] == "train" else [d]
    classes = next(d for d in vols["val"] + vols["test"])["classes"]
    B = len(classes) + 1
    print({s: len(v) for s, v in vols.items()}, flush=True)

    results, models = {}, {}
    for name in a.only:
        kind, cfg = VARIANTS[name]
        torch.manual_seed(0)
        model = Ranker(**cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        steps = a.epochs * (len(vols["train"]) // a.accum)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 1), pct_start=0.1)
        best, rng = (-1.0, None, 0), random.Random(0)
        for ep in range(1, a.epochs + 1):
            model.train()
            order = list(range(len(vols["train"])))
            rng.shuffle(order)
            for n, i in enumerate(order[: len(order) // a.accum * a.accum]):
                d = to_dev(vols["train"][i], dev)
                (loss_fn(kind, model(d), d) / a.accum).backward()
                if (n + 1) % a.accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
            if ep % 5 == 0 or ep == a.epochs:
                model.eval()
                val = evaluate(model, vols["val"], dev, B)[0.2]["gain_w_captured"]
                print(f"{name} epoch {ep}: val class-weighted gain captured @f=0.2 {val:.4f}", flush=True)
                if val > best[0]:
                    best = (val, {k: t.clone() for k, t in model.state_dict().items()}, ep)
        model.load_state_dict(best[1])
        model.eval()
        models[name] = model
        results[name] = {"val": best[0], "epoch": best[2]}

    test = {n: evaluate(m, vols["test"], dev, B) for n, m in models.items()}
    gen = torch.Generator(dev).manual_seed(0)
    test["random"] = evaluate(lambda d: torch.rand(len(d["valid1"]), device=dev, generator=gen), vols["test"], dev, B)
    test["oracle"] = evaluate(lambda d: d["gain_w"], vols["test"], dev, B)
    test["oracle_mixed"] = evaluate(lambda d: d["mixed1"].float() + 1e-4 * torch.rand(len(d["valid1"]), device=dev, generator=gen),
                                    vols["test"], dev, B)
    print("\nTEST (val-chosen epoch), tokens/uni4 | mixed | class-weighted gain captured | plain gain captured")
    for f in FRACS:
        print(f"\nf = {f}")
        for n, r in test.items():
            x = r[f]
            print(f"  {n:<16}{x['tokens']:.3f} | {x['mixed']:.3f} | {x['gain_w_captured']:.3f} (mean/scan {x['gain_w_captured_mean_scan']:.3f}) | {x['gain_captured']:.3f}")
    ref = "conv_regl" if "conv_regl" in test else next(iter(models))
    print(f"\nShare of each class's gain captured at f = 0.2 (classes with >= 2000 voxels of gain, the 15 lowest for {ref}):")
    ct = test[ref][0.2]["cls_tot"]
    frac = {n: test[n][0.2]["cls_sel"] / ct.clamp_min(1) for n in test}
    ok = [c for c in range(1, B) if ct[c] >= 2000]
    for c in sorted(ok, key=lambda c: float(frac[ref][c]))[:15]:
        print(f"  {classes[c - 1]:<28}" + " ".join(f"{n}:{float(frac[n][c]):.2f}" for n in ("mlp_bce", ref, "random", "oracle") if n in frac))
    out = {n: {str(f): {k: v for k, v in x.items() if not k.startswith("cls")} for f, x in r.items()} for n, r in test.items()}
    out["val"] = results
    out["per_class_f0.2"] = {n: {classes[c - 1]: float(frac[n][c]) for c in ok} for n in test}
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
