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

L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # bench/probe.py's (pandas-free copy: externals' envs)


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
            torch.save({"x": xin.half() if torch.is_tensor(xin) else xin, "ctx": ctx}, dst.with_suffix(".tmp"))
            os.replace(dst.with_suffix(".tmp"), dst)
        except Exception:  # noqa: BLE001 -- logged per item
            import traceback
            fail(cache, it["id"], err or traceback.format_exc())


class OrdHead(torch.nn.Module):
    """bench.probe.Head(ordinal=True): body -> one score + k-1 learned thresholds (CORAL)."""

    def __init__(self, d: int, k: int, hidden: int):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU()) if hidden else torch.nn.Identity()
        self.out = torch.nn.Linear(hidden or d, 1)
        self.th = torch.nn.Parameter(torch.zeros(k - 1))

    def forward(self, x):
        return self.out(self.body(x)) + self.th


def qwk(pred: np.ndarray, y: np.ndarray, k: int) -> float:   # bench.probe.qwk
    o = np.zeros((k, k))
    np.add.at(o, (y, np.clip(pred, 0, k - 1)), 1)
    e = o.sum(1, keepdims=True) @ o.sum(0, keepdims=True) / o.sum()
    i = np.arange(k)
    w = (i[:, None] - i[None]) ** 2 / max((k - 1) ** 2, 1)
    return float(1 - (w * o).sum() / max((w * e).sum(), 1e-12))


def score(p: np.ndarray, y: np.ndarray, ordinal: bool = False) -> float:
    """AUROC (binary), balanced accuracy (multi-class), or QWK (ordinal: p = the k-1 threshold logits)."""
    if ordinal:
        return qwk((p > 0).sum(1), y, p.shape[1] + 1)
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
    kw = {"basis": torch.load(job["basis"])} if job.get("basis") else {}   # (05/11) +covsqrt's frozen basis
    enc, prefixes = w.cls_net(pretrained=True, feature=job["feature"], **kw)
    k = 1 + max(int(it["y"]) for it in job["items"])
    ORD = job.get("kind") == "ordinal"   # (05/11) CORAL as the table's ordinal probes (LCA)

    def lossf(o, y):   # the table's losses: class-weighted CE, or CORAL's BCE on y > j
        if ORD:
            return F.binary_cross_entropy_with_logits(o, (y[:, None] > torch.arange(k - 1, device=o.device)).float())
        return F.cross_entropy(o, y, weight=wts, label_smoothing=float(job.get("smooth", 0.0)))
    post = (lambda o: o) if ORD else (lambda o: o.softmax(-1))   # what predictions store (ordinal: threshold logits)
    enc = enc.to(dev)
    split = {s: [it for it in job["items"] if it["split"] == s and (cache / f"{it['id']}.pt").exists()]
             for s in ("train", "val", "test")}
    y_tr = np.array([int(it["y"]) for it in split["train"]])
    y_val = np.array([int(it["y"]) for it in split["val"]])
    wts = torch.tensor(len(y_tr) / (k * np.maximum(np.bincount(y_tr, minlength=k), 1)), dtype=torch.float, device=dev)
    load = lambda it: torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev.type == "cuda" else contextlib.nullcontext

    inp = lambda r: r["x"].to(dev).float() if torch.is_tensor(r["x"]) else r["x"]   # ours: the extraction's dict

    @torch.no_grad()
    def feats(items) -> torch.Tensor:   # the frozen encoder's features (eval, the same path as logits)
        enc.eval()
        with amp():
            f = torch.stack([enc(inp(r), r["ctx"]).float() for r in map(load, items)])
        enc.train()
        return f

    # LP-FT (04/11, Francesco: the table's head): the table probe's (feature, head, L2) chosen on val, refitted on the
    # frozen features as bench.probe.fit (train mean / std standardisation, L-BFGS, class-weighted CE, L2 on weight
    # matrices); the fine-tuning starts from it: at iteration 0 the val / test are the table's frozen probe's
    # the frozen probe saved once (05/11: chained 30-min blocks recomputed ~1200 volumes' features each time and never
    # reached the fine-tuning); a later block loads it
    hidden = 256 if job["head"] == "mlp" else 0
    lpf, lpkey = out.with_suffix(".lp.pt"), {n: job.get(n) for n in ("feature", "head", "l2")}
    if lpf.exists() and (c := torch.load(lpf, map_location="cpu", weights_only=False))["key"] == lpkey:
        mu, sd, lp = c["mu"].to(dev), c["sd"].to(dev), c["lp"]
        D = mu.numel()
        head = (OrdHead(D, k, hidden) if ORD else torch.nn.Sequential(
            *([torch.nn.Linear(D, hidden), torch.nn.GELU()] if hidden else []), torch.nn.Linear(hidden or D, k))).to(dev)
        head.load_state_dict(c["head"])
        print(f"[cls] frozen probe loaded from {lpf}: {lp}", flush=True)
    else:
        F_tr = feats(split["train"])
        mu, sd = F_tr.mean(0), F_tr.std(0).clamp_min(1e-6)
        y_t, X_tr = torch.tensor(y_tr, device=dev), (F_tr - mu) / sd

        def fit(l2: float, X=None):
            X = X_tr if X is None else X
            torch.manual_seed(0)
            h = (OrdHead(X.shape[1], k, hidden) if ORD else torch.nn.Sequential(
                *([torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU()] if hidden else []),
                torch.nn.Linear(hidden or X.shape[1], k))).to(dev)
            o = torch.optim.LBFGS(h.parameters(), max_iter=500, line_search_fn="strong_wolfe")

            def closure():
                o.zero_grad()
                ls = lossf(h(X), y_t) + l2 * sum(p.square().sum() for p in h.parameters()
                                                                           if p.ndim > 1)
                ls.backward()
                return ls
            o.step(closure)
            with torch.no_grad():
                h.train_ce = float(lossf(h(X), y_t))   # the fine-tuning's first loss must match it
            return h

        if job.get("l2") is not None:   # the table's L2 (its reading)
            l2 = float(job["l2"])
            head = fit(l2)
        else:   # (04/11, the lighter reading: one curve, 16 + 8 mm) the table's L2 grid, chosen on the val
            X_val = (feats(split["val"]) - mu) / sd
            cand = []
            for l2 in L2_GRID:
                h = fit(l2)
                with torch.no_grad():
                    cand.append((score(post(h(X_val)).cpu().numpy(), y_val, ORD), l2, h))
                print(f"[cls] frozen probe l2 {l2:g}: val {cand[-1][0]:.4f}", flush=True)
            _, l2, head = max(cand, key=lambda c: c[0])
            del X_val, cand
        lp = {"feature": job["feature"], "head": job["head"], "l2": l2}
        lp["train_ce"] = head.train_ce
        print(f"[cls] frozen probe head: {lp} ({time.time() - t_start:.0f} s from start)", flush=True)
    lp_state = {"key": lpkey, "mu": mu.cpu(), "sd": sd.cpu(), "head": {n: t.cpu() for n, t in head.state_dict().items()}}
    if not lpf.exists():
        torch.save(lp_state | {"lp": lp}, lpf)

    def augment(r: dict) -> dict:
        """(04/11, training only, the same for every model) a flip of each axis with p 0.5 and the intensities scaled
        by U(0.9, 1.1). Ours: the extraction's dict (volume [*S], value scale a, c); others: x [1, *S] (+ ctx, the
        crop's mask, flipped with it)."""
        dims = [d for d in range(3) if rng.random() < 0.5]
        sc = float(rng.uniform(0.9, 1.1))
        if torch.is_tensor(r["x"]):
            fl = (lambda t: t.flip([d + t.dim() - 3 for d in dims]) if dims else t)
            return {"x": fl(r["x"]) * sc, "ctx": fl(r["ctx"]) if torch.is_tensor(r["ctx"]) else r["ctx"]}
        x = dict(r["x"])
        x["volume"] = x["volume"].flip(dims) if dims else x["volume"]
        a_, c_ = x["value"]
        x["value"] = (a_ * sc, c_ * sc)
        return {"x": x, "ctx": r["ctx"]}

    def logits(it, aug: bool = False) -> torch.Tensor:
        r = load(it)
        r = augment(r) if aug else r
        f = enc(inp(r), r["ctx"]).float()
        return head(((f - mu) / sd)[None])[0]

    is_enc = lambda n: any(n == p or n.startswith(p + ".") for p in prefixes)
    enc_p = [p for n, p in enc.named_parameters() if is_enc(n)]
    other = [p for n, p in enc.named_parameters() if not is_enc(n)] + list(head.parameters())
    iters, every, B = int(job["iters"]), int(job["val_every"]), int(job["batch"])
    warm = int(float(job.get("warm", 0)) * iters)
    opt = torch.optim.AdamW([{"params": other}, {"params": enc_p, "lr": job["lr"] * float(job["enc_lr"])}],
                            lr=job["lr"], weight_decay=1e-4)
    wu = max(1, int(float(job.get("lr_warm", 0.0)) * iters))   # (04/11) linear lr warm-up, then poly decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / wu) * (1 - i / iters) ** 0.9)
    print(f"[cls] {job['cfg']['model']}: encoder {sum(p.numel() for p in enc_p) / 1e6:.1f}M at lr x{job['enc_lr']}, "
          f"frozen for {warm} of {iters} iterations; train/val/test {[len(v) for v in split.values()]} volumes, k {k}",
          flush=True)

    @torch.no_grad()
    def predict(items) -> np.ndarray:
        enc.eval(), head.eval()
        with amp():
            p = np.stack([post(logits(it).float()[None])[0].cpu().numpy() for it in items])
        enc.train(), head.train()
        return p

    if "val" not in lp:   # iteration 0 = the frozen probe: its val and test predictions, once (kept in the .lp.pt)
        lp["val"] = score(predict(split["val"]), y_val, ORD)
        lp["test_pred"] = predict(split["test"]).tolist()
        torch.save(lp_state | {"lp": lp}, lpf)
    v0 = lp["val"]
    print(f"[cls] iter 0 (the frozen probe): val {v0:.4f}", flush=True)
    best = {"val": v0, "iter": 0, "state": [{n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
                                            for m in (enc, head)]}
    curve, t0, start = [{"iter": 0, "val": v0}], time.time(), 1
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
                lg = logits(split["train"][j], aug=bool(job.get("aug")))
            ls = lossf(lg.float()[None], torch.tensor([int(y_tr[j])], device=dev)) / B   # (04/11) the table's head is
            # often perfect on train (CE ~0, no gradient): smoothing keeps a signal; iteration 0 is still the table's
            ls.backward()
            loss += float(ls)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 12)
        opt.step(), opt.zero_grad(set_to_none=True), sched.step()
        if it % 25 == 0 or it <= 5 or it == warm + 1:   # the first iterations one by one: speed and memory early
            print(f"[cls] iter {it} loss {loss:.4f} {(time.time() - t1) / (it - start + 1):.2f} s/iter "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if it % every == 0 or it == iters:
            v = score(predict(split["val"]), y_val, ORD)
            curve.append({"iter": it, "val": v})
            print(f"[cls] iter {it}: val {v:.4f} (best {max(v, best['val']):.4f})", flush=True)
            if v > best["val"]:
                best = {"val": v, "iter": it, "state": [{n: t.detach().cpu().clone() for n, t in m.state_dict().items()}
                                                        for m in (enc, head)]}
            pat = int(job.get("patience") or 0)   # (04/11) stop after `patience` validations without a new best, not
            if pat and it >= iters // 2 and sum(r["iter"] > best["iter"] for r in curve) >= pat:   # before half the run
                print(f"[cls] early stop at iter {it}: {pat} validations without a new best", flush=True)   # (noisy val)
                break
        if time.time() - t_save > save_s and it < iters:
            torch.save({"key": key, "it": it, "enc": enc.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state, "best": best, "curve": curve,
                        "sec": time.time() - t0}, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)
            t_save = time.time()
    enc.load_state_dict(best["state"][0]), head.load_state_dict(best["state"][1])
    res = {"model": job["cfg"]["model"], "best_iter": best["iter"], "val": best["val"], "curve": curve,
           "frozen_probe": lp,
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
