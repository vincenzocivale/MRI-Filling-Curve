"""Does reading 8 mm everywhere and 4 mm where a learned ranker says give more Dice than the default for a reasonable cost?
Trains sfc_gdn2.adaptive.AdaptiveSegNet (ranker + decoder; the encoder stays frozen) on TotalSeg-MRI with the recipe of
`sfc bench-segdec` (fm/segrun.py `train`: batch 2, AdamW lr 1e-3 poly decay, CE + soft Dice, 160^3 tiles with 1/3 centred
on a class, validation by sliding window every `val-every` iterations on the same 30 volumes as scripts/seg_res.py), plus
the oracle warm-up (q: 0.8 -> 0 over the first quarter) and the ranker's ListNet loss. The test split is not read.
At the end the best state is run on the validation volumes at several shares of cells read at 4 mm: Dice, tokens as a share
of uniform 4 mm, seconds and peak GPU memory per volume.

    python scripts/adaptive_train.py <run dir> --out $SCRATCH/adaptive/a30.json [--frac 0.3] [--iters 12500]
"""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sfc_gdn2.adaptive import AdaptiveSegNet
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.fm import segrun
from sfc_gdn2.fm.sfc import SEG_PATCH, SFC
from sfc_gdn2.io import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.3, help="share of the 8 mm cells read at 4 mm (training and validation)")
    ap.add_argument("--cells", type=int, default=4, help="PatchDecode cells per leaf edge")
    ap.add_argument("--iters", type=int, default=12500)
    ap.add_argument("--val-every", type=int, default=2500)
    ap.add_argument("--val-max", type=int, default=30)
    ap.add_argument("--train-max", type=int, help="smoke test")
    ap.add_argument("--chunk", type=int, default=131072)
    ap.add_argument("--sweep", type=float, nargs="+", default=[0.0, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--sweep-n", type=int, help="validation volumes for the final sweep (default all of --val-max)")
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
    s = bench["segdec"]
    cache = Path(s["cache"]) / "sfc_gdn2"
    split = {sp: [i for i in items if i["split"] == sp and (cache / f"{i['id']}.pt").exists()] for sp in ("train", "val")}
    split["val"] = split["val"][: a.val_max]
    split["train"] = split["train"][: a.train_max]
    seg = Path(split["train"][0]["seg"])
    k = 1 + len(list(seg.glob("*.nii*")))

    w = SFC({"checkpoint": a.checkpoint, "args": {"patch_mm": 8, "chunk": a.chunk}}, "cuda")
    enc = w._encoder().to(dev).eval().requires_grad_(False)
    enc.taps, enc.chunk = (), a.chunk
    torch.manual_seed(a.seed)
    net = AdaptiveSegNet(enc, w.curves, int(w.run_cfg["seed"]), k, a.frac, a.cells).to(dev)
    params = [p for n, p in net.named_parameters() if not n.startswith("encoder.") and not n.startswith("apt.")]
    for n, p in net.named_parameters():
        p.requires_grad_(not (n.startswith("encoder.") or n.startswith("apt.")))
    print(f"[adaptive] trainable {sum(p.numel() for p in params) / 1e6:.2f}M (ranker + decoder), frozen encoder, "
          f"{len(split['train'])} train / {len(split['val'])} val volumes, k={k}", flush=True)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731

    def load(it):
        r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
        return {"x": r["x"], "y": r["y"], "grid_from_gt": r["grid_from_gt"], "id": it["id"],
                "ctx": torch.cat([r["ctx"].float(), torch.tensor(r["x"].shape[1:], dtype=torch.float)])}

    train_v, val_v = [load(i) for i in split["train"]], [load(i) for i in split["val"]]
    rng = np.random.default_rng(a.seed)
    data = segrun.Patches(train_v, SEG_PATCH, k, rng)
    del train_v

    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / a.iters) ** 0.9)
    key = {"iters": a.iters, "frac": a.frac, "cells": a.cells, "train": len(split["train"]), "seed": a.seed}
    last, best, curve, start, t0 = out.with_suffix(".last.pt"), {"val": -1.0}, [], 1, time.time()
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        if st["key"] == key:
            net.load_state_dict(st["net"], strict=False)
            opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"])
            rng.bit_generator.state = st["rng"]
            torch.set_rng_state(st["torch_rng"])
            torch.cuda.set_rng_state(st["cuda_rng"])
            best, curve, start, t0 = st["best"], st["curve"], st["it"] + 1, t0 - st["min"] * 60
            print(f"[adaptive] resumed at iter {st['it']} ({st['min']:.1f} min done)", flush=True)

    def trained():
        return {n: t.detach().to("cpu", copy=True) for n, t in net.state_dict().items()
                if not n.startswith("encoder.") and not n.startswith("apt.")}

    def set_mode():
        net.train()
        net.encoder.eval()

    def validate(n=None, frac=None):
        net.eval()
        net.split_frac = a.frac if frac is None else frac
        net.stats.update(leaves=0, uni4=0)
        ds, secs, peak = [], [], 0.0
        for v in val_v[:n]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            p = segrun.sliding(net, v["x"].to(dev).float(), SEG_PATCH, k, amp, v["ctx"])
            torch.cuda.synchronize()
            secs.append(time.time() - t)
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
            ds.append(segrun.dice_on(p.argmax(0).to(dev), v["y"].to(dev), k))
            del p
        return {"dice": float(np.nanmean(ds)), "s_per_volume": float(np.mean(secs)), "peak_gb": peak,
                "tokens_vs_uni4": net.stats["leaves"] / max(net.stats["uni4"], 1), "n": len(ds)}

    last_save = time.time()
    set_mode()
    for it in range(start, a.iters + 1):
        net.q = 0.8 * max(0.0, 1 - it / (0.25 * a.iters))
        x, y, c = data.sample(2)
        yd = y.to(dev)
        with amp():
            logits = net(x.to(dev).float(), c.to(dev), y=yd)
        loss = segrun.loss_fn(logits, yd, k) + net.aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 12)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"iter {it}: loss {float(loss)}, grad norm {float(gn)}")
        opt.step()
        sched.step()
        if it % 50 == 0:
            print(f"[adaptive] it {it} loss {float(loss):.3f} aux {float(net.aux):.3f} q {net.q:.2f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if it % a.val_every == 0 or it == a.iters:
            r = validate()
            curve.append({"iter": it, "loss": float(loss), **r, "min": (time.time() - t0) / 60})
            print(f"[adaptive] it {it} val dice {r['dice']:.4f} tokens/uni4 {r['tokens_vs_uni4']:.3f} "
                  f"{r['s_per_volume']:.1f} s/vol peak {r['peak_gb']:.1f} GB ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if r["dice"] > best["val"]:
                best = {"val": r["dice"], "iter": it, "state": trained()}
            set_mode()
        if time.time() - last_save > 600 and it < a.iters:
            torch.save({"key": key, "it": it, "net": trained(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(), "best": best, "curve": curve,
                        "min": (time.time() - t0) / 60}, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)
            last_save = time.time()

    net.load_state_dict(best.pop("state"), strict=False)
    sweep = {}
    for f in a.sweep:
        sweep[str(f)] = validate(a.sweep_n, f)
        print(f"[adaptive] sweep f={f}: {sweep[str(f)]}", flush=True)
    out.write_text(json.dumps({"best": best, "curve": curve, "sweep": sweep, "frac": a.frac, "cells": a.cells,
                               "iters": a.iters, "n_val": len(val_v), "trainable_params": sum(p.numel() for p in params)},
                              indent=1))
    last.unlink(missing_ok=True)
    print(f"[adaptive] best val dice {best['val']:.4f} (iter {best['iter']}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
