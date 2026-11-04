"""Fine-tuned volume classification (04/11, Francesco: P2), inside a model's env: `python -m sfc_gdn2.fm.clsrun <job>`.
The same protocol for every model: the encoder unfrozen, read as in the table's probes (wrapper hooks), a linear head.

Wrapper hooks (fm/*.py):
- `cls_input(prepared) -> (x [1, *S] float, ctx | None)`: the model's input of one volume (ours: the whole scan and its
  spacing; nnFoundation ViT: the centred crop of crop_gap_last).
- `cls_net(pretrained) -> (module, encoder prefixes)`: module(x, ctx) -> the volume's feature [D], with gradient; the
  prefixes name the pretrained parameters (their own lr, frozen during the warm-up).

job = {"stage": "prep" | "train", "cfg", "cache", "device", "items": [{"id", "image", "split", "y"}] (y: class index),
       train: "iters", "val_every", "batch" (volumes per step, one backward each), "lr" (head), "enc_lr" (encoder lr /
       lr), "warm" (fraction of the iterations with the encoder frozen), "seed", "out"}
prep: <cache>/<id>.pt = {x fp16, ctx}. train: AdamW (wd 1e-4), poly decay, class-weighted cross-entropy on
LayerNorm(feature) -> Linear (LazyLinear: any width), bf16 autocast; every val_every iterations the val AUROC (binary)
or balanced accuracy, the best state kept in RAM; test once with it. Out: <out> = {val, test predictions (probabilities)
per id, curve}; the metrics and CIs are the launcher's (bench/clsft.py, probe.bootstrap). Stdlib + numpy + torch only.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .worker import build, fail, prepared


def prep(job: dict) -> None:
    w = build(job["cfg"], job["device"])
    cache = Path(job["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    items = [it for it in job["items"] if not (cache / f"{it['id']}.pt").exists()]
    print(f"[cls] prep {job['cfg']['model']}: {len(items)} of {len(job['items'])} to do", flush=True)
    for it, x, err, _ in prepared(w, job["cfg"], items, int(job.get("workers", 0))):
        try:
            if err:
                raise RuntimeError(err)
            xin, ctx = w.cls_input(x)
            dst = cache / f"{it['id']}.pt"
            torch.save({"x": xin.half(), "ctx": ctx}, dst.with_suffix(".tmp"))
            os.replace(dst.with_suffix(".tmp"), dst)
        except Exception:  # noqa: BLE001 -- logged per item
            import traceback
            fail(cache, it["id"], err or traceback.format_exc())


def score(p: np.ndarray, y: np.ndarray) -> float:
    """AUROC (binary, Mann-Whitney with ties) or balanced accuracy."""
    if p.shape[1] == 2:
        s = p[:, 1]
        r = np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float)
        for v in np.unique(s):   # tied scores share their mean rank
            r[s == v] = r[s == v].mean()
        pos = y == 1
        return float((r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum()))
    return float(np.mean([(p[y == c].argmax(1) == c).mean() for c in np.unique(y)]))


def train(job: dict) -> None:
    dev, cache, out = torch.device(job["device"]), Path(job["cache"]), Path(job["out"])
    if gb := os.environ.get("SFC_GPU_GB"):   # (04/11) a smaller GPU's memory, emulated: fine-tuning must fit 12 GB
        i = dev.index or 0
        torch.cuda.set_per_process_memory_fraction(float(gb) * 2**30 / torch.cuda.get_device_properties(i).total_memory, i)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    w = build(job["cfg"], job["device"])
    print(f"[cls] wrapper built ({time.time() - t_start:.0f} s)", flush=True)
    rng = np.random.default_rng(job["seed"])
    torch.manual_seed(job["seed"])
    enc, prefixes = w.cls_net(pretrained=True)
    k = 1 + max(int(it["y"]) for it in job["items"])
    head = torch.nn.LazyLinear(k)   # on the layer-normed feature (scale-free across models)
    enc, head = enc.to(dev), head.to(dev)
    split = {s: [it for it in job["items"] if it["split"] == s and (cache / f"{it['id']}.pt").exists()]
             for s in ("train", "val", "test")}
    y_tr = np.array([int(it["y"]) for it in split["train"]])
    wts = torch.tensor(len(y_tr) / (k * np.maximum(np.bincount(y_tr, minlength=k), 1)), dtype=torch.float, device=dev)
    load = lambda it: torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev.type == "cuda" else contextlib.nullcontext

    def logits(it) -> torch.Tensor:
        r = load(it)
        f = enc(r["x"].to(dev), r["ctx"]).float()
        return head(F.layer_norm(f, f.shape[-1:])[None])[0]

    with torch.no_grad(), amp():   # LazyLinear's width from one volume
        logits(split["train"][0])
    print(f"[cls] first volume read ({time.time() - t_start:.0f} s from start)", flush=True)
    is_enc = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    enc_p = [p for n, p in enc.named_parameters() if is_enc(n)]
    other = [p for n, p in enc.named_parameters() if not is_enc(n)] + list(head.parameters())
    iters, every, B = int(job["iters"]), int(job["val_every"]), int(job["batch"])
    warm = int(float(job.get("warm", 0)) * iters)
    opt = torch.optim.AdamW([{"params": other}, {"params": enc_p, "lr": job["lr"] * float(job["enc_lr"])}],
                            lr=job["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / iters) ** 0.9)
    print(f"[cls] {job['cfg']['model']}: encoder {sum(p.numel() for p in enc_p) / 1e6:.1f}M at lr x{job['enc_lr']}, "
          f"frozen for {warm} of {iters} iterations; train/val/test {[len(v) for v in split.values()]} volumes, k {k}",
          flush=True)

    @torch.no_grad()
    def predict(items) -> np.ndarray:
        enc.eval(), head.eval()
        with amp():
            p = np.stack([logits(it).float().softmax(-1).cpu().numpy() for it in items])
        enc.train(), head.train()
        return p

    y_val = np.array([int(it["y"]) for it in split["val"]])
    best, curve, t0, start = {"val": -1.0}, [], time.time(), 1
    # a killed job resumes (chained short jobs): the whole state every SFC_CLS_SAVE_S seconds next to the result
    last, key = out.with_suffix(".last.pt"), {n: job[n] for n in ("iters", "val_every", "batch", "lr", "enc_lr", "warm",
                                                                  "seed")}
    if last.exists() and (st := torch.load(last, map_location="cpu", weights_only=False))["key"] == key:
        enc.load_state_dict(st["enc"]), head.load_state_dict(st["head"]), opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        best, curve, start, t0 = st["best"], st["curve"], st["it"] + 1, t0 - st["sec"]
        print(f"[cls] resumed at iter {st['it']}", flush=True)
    t_save, save_s = time.time(), float(os.environ.get("SFC_CLS_SAVE_S", "600"))
    enc.train(), head.train()
    t1 = time.time()
    for it in range(start, iters + 1):
        if it in (start, warm + 1):
            for p in enc_p:
                p.requires_grad_(it > warm)
            print(f"[cls] encoder {'trained' if it > warm else 'frozen (warm-up)'} from iter {it}", flush=True)
        loss = 0.0
        for j in rng.integers(len(split["train"]), size=B):   # one backward per volume: memory of one scan
            with amp():
                lg = logits(split["train"][j])
            ls = F.cross_entropy(lg.float()[None], torch.tensor([int(y_tr[j])], device=dev), weight=wts) / B
            ls.backward()
            loss += float(ls)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 12)
        opt.step(), opt.zero_grad(set_to_none=True), sched.step()
        if it % 25 == 0 or it <= 5 or it == warm + 1:   # the first iterations one by one: speed and memory early
            print(f"[cls] iter {it} loss {loss:.4f} {(time.time() - t1) / (it - start + 1):.2f} s/iter "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if it % every == 0 or it == iters:
            v = score(predict(split["val"]), y_val)
            curve.append({"iter": it, "val": v})
            print(f"[cls] iter {it}: val {v:.4f} (best {max(v, best['val']):.4f})", flush=True)
            if v > best["val"]:
                best = {"val": v, "iter": it, "state": [{n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
                                                        for m in (enc, head)]}
        if time.time() - t_save > save_s and it < iters:
            torch.save({"key": key, "it": it, "enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state, "best": best, "curve": curve,
                        "sec": time.time() - t0}, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)
            t_save = time.time()
    enc.load_state_dict(best["state"][0]), head.load_state_dict(best["state"][1])
    res = {"model": job["cfg"]["model"], "best_iter": best["iter"], "val": best["val"], "curve": curve,
           "minutes": (time.time() - t0) / 60, "job": {n: job[n] for n in ("iters", "val_every", "batch", "lr", "enc_lr",
                                                                           "warm", "seed")}}
    for s in ("val", "test"):
        res[f"{s}_pred"] = {"id": [it["id"] for it in split[s]], "prob": predict(split[s]).tolist()}
    out.write_text(json.dumps(res))
    torch.save({"best": best, "job": res["job"]}, out.with_suffix(".best.pt"))   # kept (never delete checkpoints)
    print(f"[cls] done: best val {best['val']:.4f} at iter {best['iter']} -> {out}", flush=True)


def main(path: str) -> None:
    job = json.loads(Path(path).read_text())
    {"prep": prep, "train": train}[job["stage"]](job)


if __name__ == "__main__":
    main(sys.argv[1])
