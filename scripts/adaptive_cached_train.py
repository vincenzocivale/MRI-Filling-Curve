"""Does our adaptive 8 / 4 mm segmentation (ranker trained with the labels as privileged information + leaf decoder:
sfc_gdn2.adaptive_cached.CachedAdaptiveNet) segment TotalSeg-MRI well on the updated checkpoint (p13), whose frozen tokens
are in fcorrent's token cache?  Same protocol as `sfc bench-segdec` (batch 2, AdamW lr 1e-3 poly decay, CE + soft Dice, 160^3
tiles with 1/3 centred on a class, 12 500 iterations, validation every 2 500 on the benchmark's 100 validation volumes, the
test split read once at the end), plus the oracle warm-up (q: 0.8 -> 0 over the first quarter) and the ranker's ListNet loss.
Tiles start on the 16 mm grid of the cache. Validation / test: the whole scan in non-overlapping aligned tiles (no sliding
window), Dice on the model's grid (segrun.dice_on), as the benchmark's validation curve.

    python scripts/adaptive_cached_train.py --tokens <tok_pre> --out $SCRATCH/adaptive_cached/a30.json [--frac 0.3]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfc_gdn2.adaptive_cached import CachedAdaptiveNet
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.fm import segrun
from sfc_gdn2.io import load_yaml

T = 160   # tile edge in voxels, as the benchmark's SEG_PATCH


def tile_extent(k4, cap):
    """Voxels per axis of a tile: T, shrunk (in whole 16 mm cells) until it holds at most `cap` 8 mm cells. A tile of 160^3
    voxels of a scan with 1.3 mm voxels is ~200 mm and 50^3 8 mm cells: its decoding and its dense logits were 9 s per
    tile (profile of 21/10), against < 1 s for coarse scans."""
    k8, unit = [2 * q for q in k4], [4 * q for q in k4]
    ext = [T] * 3
    while int(np.prod([-(-e // a) for e, a in zip(ext, k8)])) > cap:
        scale = (cap / int(np.prod([-(-e // a) for e, a in zip(ext, k8)]))) ** (1 / 3)
        new = [max(u, int(e * scale) // u * u) for e, u in zip(ext, unit)]
        if new == ext:   # rounding down to whole 16 mm cells did not move: take one cell off the longest axis
            j = int(np.argmax([e / u for e, u in zip(ext, unit)]))
            if ext[j] <= unit[j]:
                break
            new[j] = ext[j] - unit[j]
        ext = new
    return ext


def seg_loss(lgs, ys, k):
    """segrun.loss_fn (CE + soft Dice with batch statistics) on tiles of different shapes, as flat voxel lists."""
    lg = torch.cat([l.flatten(1) for l in lgs], 1).float()   # [k, N]
    y = torch.cat([v.flatten() for v in ys]).long()
    ce = F.cross_entropy(lg.T, y, ignore_index=255)
    ok = (y != 255).float()
    p = lg.softmax(0) * ok
    yc = torch.where(y == 255, torch.zeros_like(y), y)
    inter = p.new_zeros(k).scatter_add(0, yc, p.gather(0, yc[None])[0])
    tsum = p.new_zeros(k).scatter_add(0, yc, ok)
    return ce + 1 - ((2 * inter[1:] + 1e-5) / (p.sum(1)[1:] + tsum[1:] + 1e-5)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="token cache root (<id>.<mm>.npy, <id>.json, *.p512.npy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.3, help="share of the 8 mm cells read at 4 mm")
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--cap", type=int, default=16000, help="most 8 mm cells in a tile (tiles of fine-voxel scans are cut)")
    ap.add_argument("--iters", type=int, default=12500)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-max", type=int, default=100)
    ap.add_argument("--train-max", type=int)
    ap.add_argument("--test", action="store_true", help="read the test split once at the end")
    ap.add_argument("--test-max", type=int)
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.0, 0.15, 0.5, 1.0])
    ap.add_argument("--sweep-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bench = load_yaml("configs/bench/bench.yaml")
    tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
    vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(bench["out_dir"]) / "splits.csv")
    items = bs.task_items(tasks, vols, splits, a.seed)["totalseg"]
    cache = Path(bench["segdec"]["cache"]) / "sfc_gdn2"
    tok = Path(a.tokens)
    split = {sp: [i for i in items if i["split"] == sp and (cache / f"{i['id']}.pt").exists() and (tok / f"{i['id']}.json").exists()]
             for sp in ("train", "val", "test")}
    split["train"], split["val"] = split["train"][: a.train_max], split["val"][: a.val_max]
    split["test"] = split["test"][: a.test_max]
    seg = Path(split["train"][0]["seg"])
    k = 1 + len(list(seg.glob("*.nii*")))
    print(f"[acached] {({s: len(v) for s, v in split.items()})}, k={k}", flush=True)

    torch.manual_seed(a.seed)
    net = CachedAdaptiveNet(a.tokens, k, a.frac, a.cells).to(dev)
    params = [p for p in net.parameters()]
    print(f"[acached] trainable {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731

    def load(it, fg=False):
        r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
        v = {"id": it["id"], "x": r["x"], "y": r["y"], "k4": net.cache.meta(it["id"])["k4"], "fg": []}
        if fg:   # voxels of each present class (a subsample: the class balance of nnU-Net's oversampling)
            y = v["y"]
            nz = ((y > 0) & (y < 255)).nonzero()
            if len(nz):
                nz = nz[torch.randperm(len(nz))[:200000]]
                lab = y[nz[:, 0], nz[:, 1], nz[:, 2]].long()
                order = lab.argsort()
                nz, lab = nz[order], lab[order]
                cnt = torch.bincount(lab, minlength=k)
                st = cnt.cumsum(0) - cnt
                v["fg"] = [nz[s:s + c] for s, c in zip(st.tolist(), cnt.tolist()) if c > 0]
        return v

    train_v = [load(i, True) for i in split["train"]]
    val_v = [load(i) for i in split["val"]]
    rng = np.random.default_rng(a.seed)

    def sample():
        v = train_v[int(rng.integers(len(train_v)))]
        shape = v["x"].shape[1:]
        ext = [min(e, s) for e, s in zip(tile_extent(v["k4"], a.cap), shape)]
        if v["fg"] and rng.random() < 1 / 3:
            cls = v["fg"][int(rng.integers(len(v["fg"])))]
            c = cls[int(rng.integers(len(cls)))].numpy()
            start = np.clip(c - np.array(ext) // 2, 0, np.array(shape) - np.array(ext))
        else:
            start = np.array([int(rng.integers(s - e + 1)) for s, e in zip(shape, ext)])
        unit = [4 * q for q in v["k4"]]   # the 16 mm cell: tiles start on this grid
        corner = [int(s) // u * u for s, u in zip(start, unit)]
        V = [min(e, s - c) for e, s, c in zip(tile_extent(v["k4"], a.cap), shape, corner)]
        return v, corner, V

    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / a.iters) ** 0.9)
    key = {"iters": a.iters, "frac": a.frac, "cells": a.cells, "train": len(split["train"]), "seed": a.seed}
    last, best, curve, start_it, t0 = out.with_suffix(".last.pt"), {"val": -1.0}, [], 1, time.time()
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        if st["key"] == key:
            net.load_state_dict(st["net"])
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            rng.bit_generator.state = st["rng"]
            torch.set_rng_state(st["torch_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            best, curve, start_it, t0 = st["best"], st["curve"], st["it"] + 1, t0 - st["min"] * 60
            print(f"[acached] resumed at iter {st['it']} ({st['min']:.1f} min done)", flush=True)

    def evaluate(vs, frac):
        net.eval()
        net.split_frac = frac
        net.stats.update(leaves=0, uni4=0)
        ds, secs, peak = [], [], 0.0
        with torch.no_grad():
            for v in vs:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t = time.time()
                shape = v["x"].shape[1:]
                unit = [4 * q for q in v["k4"]]
                step = [max(u, e // u * u) for u, e in zip(unit, tile_extent(v["k4"], a.cap))]
                pred = torch.zeros(shape, dtype=torch.uint8, device=dev)
                for c0 in range(0, shape[0], step[0]):
                    for c1 in range(0, shape[1], step[1]):
                        for c2 in range(0, shape[2], step[2]):
                            corner = [c0, c1, c2]
                            V = [min(s, sh - c) for s, sh, c in zip(step, shape, corner)]
                            xv = v["x"][0, c0:c0 + V[0], c1:c1 + V[1], c2:c2 + V[2]].to(dev).float()
                            with amp():
                                lg = net(v["id"], corner, xv)
                            pred[c0:c0 + V[0], c1:c1 + V[1], c2:c2 + V[2]] = lg.argmax(0).to(torch.uint8)
                torch.cuda.synchronize()
                secs.append(time.time() - t)
                peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
                ds.append(segrun.dice_on(pred, v["y"].to(dev), k))
        return {"dice": float(np.nanmean(ds)), "per_volume": [float(d) for d in ds], "s_per_volume": float(np.mean(secs)),
                "peak_gb": peak, "tokens_vs_uni4": net.stats["leaves"] / max(net.stats["uni4"], 1), "n": len(ds)}

    def trained():
        return {n: t.detach().to("cpu", copy=True) for n, t in net.state_dict().items()}

    last_save, run = time.time(), {}
    nxt = [sample() for _ in range(2)]
    for v, c, V in nxt:
        net.prefetch(v["id"], c, V)
    net.train()
    for it in range(start_it, a.iters + 1):
        net.q = 0.8 * max(0.0, 1 - it / (0.25 * a.iters))
        net.split_frac = a.frac
        batch, nxt = nxt, [sample() for _ in range(2)]
        net.cache.next_batch()
        for v, c, V in nxt:
            net.prefetch(v["id"], c, V)
        lg, ys, aux, dg = [], [], 0.0, {}
        for v, c, V in batch:
            xv = v["x"][0, c[0]:c[0] + V[0], c[1]:c[1] + V[1], c[2]:c[2] + V[2]].to(dev).float()
            yv = v["y"][c[0]:c[0] + V[0], c[1]:c[1] + V[1], c[2]:c[2] + V[2]].to(dev)
            with amp():
                o = net(v["id"], c, xv, yv)
            aux = aux + net.aux / len(batch)
            lg.append(o)
            ys.append(yv)
            for key_, val_ in net.diag.items():
                dg.setdefault(key_, []).append(val_)
        loss = seg_loss(lg, ys, k) + aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 12)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"iter {it}: loss {float(loss)}, grad norm {float(gn)}")
        opt.step()
        sched.step()
        for key_, vals_ in dg.items():
            run.setdefault(key_, []).extend(vals_)
        if it % 50 == 0:
            m = {key_: float(np.mean(v_)) for key_, v_ in run.items()}
            run = {}
            print(f"[acached] it {it} seg {float(loss) - float(aux):.3f} aux {float(aux):.3f} ranker top{a.frac:g} captured "
                  f"{m.get('captured', float('nan')):.3f} (chance {a.frac:g}) kl {m.get('kl', float('nan')):.2f} q {net.q:.2f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if it % a.val_every == 0 or it == a.iters:
            r = evaluate(val_v, a.frac)
            curve.append({"iter": it, "loss": float(loss), **{x: y for x, y in r.items() if x != "per_volume"},
                          "min": (time.time() - t0) / 60})
            print(f"[acached] it {it} val dice {r['dice']:.4f} tokens/uni4 {r['tokens_vs_uni4']:.3f} {r['s_per_volume']:.1f} s/vol "
                  f"peak {r['peak_gb']:.1f} GB ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if r["dice"] > best["val"]:
                best = {"val": r["dice"], "iter": it, "state": trained()}
            net.train()
        if time.time() - last_save > 600 and it < a.iters:
            torch.save({"key": key, "it": it, "net": trained(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(), "best": best, "curve": curve,
                        "min": (time.time() - t0) / 60}, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)
            last_save = time.time()

    net.load_state_dict(best.pop("state"))
    sweep = {}
    for f in a.sweep:
        r = evaluate(val_v[: a.sweep_n], f)
        sweep[str(f)] = {x: y for x, y in r.items() if x != "per_volume"}
        print(f"[acached] sweep f={f} (first {a.sweep_n} val volumes): {sweep[str(f)]}", flush=True)
    test = None
    if a.test:   # the test split, read once, with the model chosen on validation
        tv = [load(i) for i in split["test"]]
        test = evaluate(tv, a.frac)
        print(f"[acached] TEST dice {test['dice']:.4f} on {test['n']} volumes, tokens/uni4 {test['tokens_vs_uni4']:.3f}, "
              f"{test['s_per_volume']:.1f} s/vol, peak {test['peak_gb']:.1f} GB", flush=True)
    out.write_text(json.dumps({"best": best, "curve": curve, "sweep": sweep, "test": test, "frac": a.frac, "cells": a.cells,
                               "iters": a.iters, "n_val": len(val_v), "trainable_params": sum(p.numel() for p in params)},
                              indent=1))
    last.unlink(missing_ok=True)
    print(f"[acached] best val dice {best['val']:.4f} (iter {best['iter']}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
