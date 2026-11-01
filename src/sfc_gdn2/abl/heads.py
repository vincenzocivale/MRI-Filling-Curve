"""Ablation basket (01/11, user), step 2: simple segmentation heads on Small's dense tokens (abl/tokens.py), one
recipe for every head, variant and data seed, so the tokens are what differs.

Heads (per 4 mm cell; the cell = k4 = round(4 / spacing) native voxels per axis, the input grid of segdec):
- linear: logits = a linear map of the cell's 4 mm token (last block) ++ its 8 mm and 16 mm parents' tokens (last ++
  taps): 1024 + 4096 + 4096 = 9216 features; logits at 4 mm, trilinear (on logits) to the native voxels.
- mlp: the same input, Linear 9216 -> 512, GELU, Linear 512 -> K.
- primus: Primus' decoder (nnssl / dynamic_network_architectures PatchDecode, the one SegNet used until 27/10, then
  at a 16^3 sub-grid): each 4 mm token (last block, 1024) alone -> a 4^3 sub-grid of logits (two transposed-conv
  stages of stride 2, LayerNorm + GELU between: per token, no mixing across tokens), trilinear to the cell's k4
  voxels (k4 varies with the scan's spacing, so a fixed 4^3 sub-grid for every scan, not k4 itself).
Recipe (fixed): AdamW lr 1e-3, wd 1e-4 (segrun's), poly decay, EPOCHS epochs of CROPS crops per train volume, batches
of BATCH crops of CROP^3 4 mm cells (1/3 centred on a voxel of a random present foreground class, segrun's Patches),
loss segrun.loss_fn (CE + batch soft Dice) on the crop's native voxels; init seed 0, the crops' rng seed 0. Each
epoch: val Dice (segrun.dice_on on the input grid, argmax) of whole val scans, the best epoch kept (selection on val
only). Then segrun's test: post-processing chosen on the best state's val predictions (segrun.choose_postproc, the
rule family of every segdec run), class probabilities trilinear to every GT voxel (segrun.dice_full_res), Dice per
volume, 95% CI over test-subject resamples (probe.subject_ci). Crops read the token files by contiguous reads
(TokenCache), val scans are held in RAM, test scans read one at a time.

--data-seed k (1, 2, 3): 90 % of the train subjects (bench/segdec.select's SFC_DATA_SEED, rng 1000 + k); 0: all.
Out: <out>/<task>/<head>__<variant>__d<k>.json {test_dice, test_ci95, val_dice, per_volume, ...}.

    python -m sfc_gdn2.abl.heads --task wmh_seg --head mlp --tok $SCRATCH/sfc_runs/abl/tok/mean/wmh_seg
        --out $SCRATCH/sfc_runs/abl/heads [--curve mean] [--variant mean] [--data-seed 1] [--epochs n] [--sample k]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..fm.token_cache import TokenCache
from .tokens import INPUTS, task_items

LR, WD, EPOCHS, CROPS, BATCH, CROP = 1e-3, 1e-4, 30, 8, 4, 16   # the one recipe (user 01/11: same for all)
SUB = 4          # primus: logits per 4 mm token edge (PatchDecode's patch: a power of 2)
HIDDEN = 512     # mlp
SEED = 0         # init and crop order (user 01/11: fixed; only --data-seed varies the input)
FG = 1 / 3       # crops centred on a foreground voxel (segrun's Patches)


class Head(nn.Module):
    def __init__(self, kind: str, k: int, d4: int, d8: int):
        super().__init__()
        self.kind = kind
        self.P = SUB if kind == "primus" else 1
        if kind == "primus":
            from dynamic_network_architectures.building_blocks import patch_encode_decode as pe
            LayerNormNd, PatchDecode = pe.LayerNormNd, pe.PatchDecode
            self.net = PatchDecode((SUB,) * 3, d4, k, norm=LayerNormNd, activation=nn.GELU)
        else:
            f = d4 + 2 * d8
            self.net = nn.Linear(f, k) if kind == "linear" else nn.Sequential(nn.Linear(f, HIDDEN), nn.GELU(),
                                                                                nn.Linear(HIDDEN, k))

    def forward(self, f: dict) -> torch.Tensor:
        """{4, 8, 16: [nx, ny, nz, C] tokens on the 4 mm cells (parents repeated)} -> [K, P nx, P ny, P nz]."""
        if self.kind == "primus":
            return self.net(f[4].float().permute(3, 0, 1, 2)[None])[0]
        return self.net(torch.cat([f[4], f[8], f[16]], -1).float()).permute(3, 0, 1, 2)


class Tok(TokenCache):
    """The token files of one curve name (TokenCache reads `mean` only): contiguous reads of a box of cells."""

    def __init__(self, root, curve: str):
        super().__init__(root)
        self.curve = curve

    def _open(self, i: str, mm: float):
        key = (i, mm)
        if key not in self._fd:
            name = self.root / f"{i}.{int(mm)}.{self.curve}.npy"
            with open(name, "rb") as f:
                major, _ = np.lib.format.read_magic(f)
                shape, _, dtype = (np.lib.format.read_array_header_1_0 if major == 1 else
                                   np.lib.format.read_array_header_2_0)(f)
                off = f.tell()
            self._fd[key] = (os.open(name, os.O_RDONLY), shape, np.dtype(dtype), off)
        return self._fd[key]

    def box(self, i: str, mm: int, lo, hi) -> np.ndarray:
        n = [b - a for a, b in zip(lo, hi)]
        return self._read(i, float(mm), list(lo), n, [1, 1, 1]).numpy().reshape(*n, -1)


def boxes(get, lo, hi) -> dict:
    """The 4 mm cells [lo, hi) and the boxes of their 8 / 16 mm parents: get(mm, lo, hi) -> [n.., C] array."""
    return {mm: get(mm, [a // m for a in lo], [(b - 1) // m + 1 for b in hi]) for mm, m in ((4, 1), (8, 2), (16, 4))}


def levels(bx: dict, lo, hi, dev) -> dict:
    """boxes() on the GPU, the parents repeated on their 4 mm cells -> {4, 8, 16: [nx, ny, nz, C]}."""
    out = {}
    for mm, m in ((4, 1), (8, 2), (16, 4)):
        t = torch.from_numpy(np.ascontiguousarray(bx[mm])).to(dev, non_blocking=True)
        ix = [torch.arange(a, b, device=dev) // m - a // m for a, b in zip(lo, hi)]
        out[mm] = t if m == 1 else t[ix[0][:, None, None], ix[1][None, :, None], ix[2][None, None, :]]
    return out


def up(z: torch.Tensor, k, P: int) -> torch.Tensor:
    """[K, P n..] logits of n cells -> [K, n k..] voxels, trilinear (align_corners False: cell centres stay)."""
    n = [s // P for s in z.shape[1:]]
    return F.interpolate(z[None], size=[a * b for a, b in zip(n, k)], mode="trilinear", align_corners=False)[0]


@torch.no_grad()
def predict(head: Head, get, meta: dict, dev, slab: int = 8) -> torch.Tensor:
    """Whole scan -> class probabilities [K, *shape] fp16 on dev, in slabs of 4 mm cells along x with one cell of
    context per side (the same trilinear values as the whole grid at once)."""
    g, k, S = meta["levels"]["4"]["grid"], meta["k4"], meta["shape"]
    out = None
    for a in range(0, g[0], slab):
        b, lo, hi = min(a + slab, g[0]), max(a - 1, 0), min(a + slab + 1, g[0])
        z = up(head(levels(boxes(get, [lo, 0, 0], [hi, g[1], g[2]]), [lo, 0, 0], [hi, g[1], g[2]], dev)), k, head.P)
        z = z[:, (a - lo) * k[0]: (b - lo) * k[0], : S[1], : S[2]].softmax(0).half()
        out = torch.empty((len(z), S[0], S[1], S[2]), dtype=torch.half, device=dev) if out is None else out
        out[:, a * k[0]: min(b * k[0], S[0])] = z[:, : min(b * k[0], S[0]) - a * k[0]]
    return out


def ram(tok: Tok, i: str) -> callable:
    """A scan's three levels read whole into RAM -> get(mm, lo, hi)."""
    g = json.loads((tok.root / f"{i}.json").read_text())["levels"]
    arr = {mm: tok.box(i, mm, [0, 0, 0], g[str(mm)]["grid"]) for mm in (4, 8, 16)}
    return lambda mm, lo, hi: arr[mm][lo[0]: hi[0], lo[1]: hi[1], lo[2]: hi[2]]


def crop(rng, v: dict) -> tuple:
    """A training crop of CROP^3 4 mm cells (FG of them centred on a voxel of a random present foreground class) ->
    (cells lo, hi), with one cell of context per side where the scan has it (lm, hm)."""
    g, k = v["meta"]["levels"]["4"]["grid"], v["meta"]["k4"]
    n = [min(CROP, a) for a in g]
    if v["fg"] and rng.random() < FG:
        pts = v["fg"][int(rng.integers(len(v["fg"])))]
        c = pts[int(rng.integers(len(pts)))] // np.asarray(k)
        lo = [int(min(max(cc - nn // 2, 0), a - nn)) for cc, nn, a in zip(c, n, g)]
    else:
        lo = [int(rng.integers(a - nn + 1)) for nn, a in zip(n, g)]
    hi = [a + nn for a, nn in zip(lo, n)]
    return lo, hi, [max(a - 1, 0) for a in lo], [min(b + 1, gg) for b, gg in zip(hi, g)]


def crop_logits(head: Head, bx: dict, v: dict, lo, hi, lm, hm, dev) -> tuple:
    """Logits on the crop's voxels (boxes() of the context box lm..hm, cut to lo..hi after the trilinear) and the
    labels there -> ([K, N], [N]) where labelled."""
    k, S = v["meta"]["k4"], v["meta"]["shape"]
    z = up(head(levels(bx, lm, hm, dev)), k, head.P)
    v0, v1 = [a * kk for a, kk in zip(lo, k)], [min(b * kk, s) for b, kk, s in zip(hi, k, S)]
    o = [(a - b) * kk for a, b, kk in zip(lo, lm, k)]
    z = z[:, o[0]: o[0] + v1[0] - v0[0], o[1]: o[1] + v1[1] - v0[1], o[2]: o[2] + v1[2] - v0[2]]
    y = v["y"][v0[0]: v1[0], v0[1]: v1[1], v0[2]: v1[2]].to(dev).flatten()
    z = z.flatten(1)
    ok = y != 255
    return z[:, ok], y[ok]


def load_y(it: dict) -> tuple:
    r = torch.load(Path(INPUTS) / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    y = r["y"][0] if r["y"].dim() == 4 else r["y"]
    return y, r["grid_from_gt"]


def run(task: str, kind: str, tok_dir: str, curve: str, variant: str, data_seed: int, out_dir: str,
        epochs: int = EPOCHS, sample: int | None = None, dev: str = "cuda") -> dict:
    from ..bench.geom import gt_labels
    from ..bench.probe import subject_ci
    from ..fm.segrun import _components, choose_postproc, dice_full_res, dice_on, loss_fn, postproc

    t0 = time.time()
    items = task_items(task, data_seed or None, sample)
    k = 1 + len(items[0]["classes"]) if items[0].get("classes") else 2   # segdec's count (these tasks list classes)
    tok = Tok(tok_dir, curve)
    have = {it["id"] for it in items if (tok.root / f"{it['id']}.json").exists()}
    split = {s: [it for it in items if it["split"] == s and it["id"] in have] for s in ("train", "val", "test")}
    miss = sum(it["id"] not in have for it in items)
    print(f"[abl] {task} {kind} {variant} d{data_seed}: train {len(split['train'])} val {len(split['val'])} "
          f"test {len(split['test'])} (missing tokens: {miss}), K {k}", flush=True)
    assert not miss or sample, f"{miss} volumes without tokens in {tok_dir}"
    meta = {it["id"]: json.loads((tok.root / f"{it['id']}.json").read_text()) for it in items if it["id"] in have}

    tr = []
    for it in split["train"]:
        y, _ = load_y(it)
        fg = [(y == c).nonzero().numpy() for c in range(1, k)]
        fg = [p[np.random.default_rng(0).permutation(len(p))[:10000]] for p in fg if len(p)]
        tr.append({"id": it["id"], "y": y, "meta": meta[it["id"]], "fg": fg})
    va = [{"id": it["id"], "y": load_y(it)[0], "meta": meta[it["id"]], "get": ram(tok, it["id"])} for it in split["val"]]
    print(f"[abl] loaded {len(tr)} train labels, {len(va)} val scans ({time.time() - t0:.0f} s)", flush=True)

    torch.manual_seed(SEED)
    d = meta[tr[0]["id"]]["levels"]
    head = Head(kind, k, d["4"]["width"], d["8"]["width"]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    spe = max(1, -(-len(tr) * CROPS // BATCH))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / (epochs * spe)) ** 0.9)
    rng = np.random.default_rng(SEED)
    pool = ThreadPoolExecutor(BATCH)
    get = lambda i: (lambda mm, lo, hi: tok.box(i, mm, lo, hi))

    def draw():   # the next batch's crops, their tokens read ahead by threads
        vs = [tr[int(rng.integers(len(tr)))] for _ in range(BATCH)]
        return [(v, c, pool.submit(boxes, get(v["id"]), c[2], c[3])) for v, c in ((v, crop(rng, v)) for v in vs)]
    best, curve_ = {"val": -1.0}, []
    nxt = draw()
    for ep in range(epochs):
        head.train()
        ls, t1 = [], time.time()
        for _ in range(spe):
            cur, nxt = nxt, draw()
            zs, ys = [], []
            for v, c, fut in cur:
                z, y = crop_logits(head, fut.result(), v, *c, dev)
                zs.append(z)
                ys.append(y)
            z, y = torch.cat(zs, 1), torch.cat(ys)
            loss = loss_fn(z[None, :, :, None, None], y[None, :, None, None], k)   # one batch: batch Dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ls.append(loss.item())
        head.eval()
        ds = [dice_on(predict(head, v["get"], v["meta"], dev).argmax(0), v["y"].to(dev), k) for v in va]
        val = float(np.nanmean(ds)) if ds else float("nan")
        curve_.append({"epoch": ep + 1, "loss": float(np.mean(ls)), "val_dice": val, "min": (time.time() - t0) / 60})
        print(f"[abl] epoch {ep + 1}/{epochs}: loss {np.mean(ls):.4f} val dice {val:.4f} ({time.time() - t1:.0f} s)",
              flush=True)
        if val > best["val"] or best["val"] < 0:
            best = {"val": val, "epoch": ep + 1, "val_vol": ds, "state": copy.deepcopy(head.state_dict())}
    pool.shutdown(wait=False)
    head.load_state_dict(best["state"])
    head.eval()

    vals = []   # the post-processing rule on the best state's val predictions (segrun's, chosen on val only)
    for v in va:
        p = predict(head, v["get"], v["meta"], dev)
        q = (p.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy() if k <= 8 else None
        y = v["y"].numpy()
        vals.append((p.argmax(0).to(torch.uint8).cpu().numpy(), q, np.where(y == 255, 0, y), None, None))
    sizes = {}
    for v in tr[:20]:
        for c in range(1, k):
            cc, n = _components(v["y"].numpy() == c)
            if n:
                sizes.setdefault(c, []).extend(np.bincount(cc.ravel())[1:].tolist())
    sizes = {c: [int(np.percentile(s, 1)), int(np.percentile(s, 5))] for c, s in sizes.items()}
    rule = choose_postproc(vals, k, sizes, False) if vals else {}
    del vals, va

    res = {"per_volume": [], "per_class": [], "test_ids": []}
    for it in split["test"]:
        _, gfg = load_y(it)
        p = predict(head, ram(tok, it["id"]), meta[it["id"]], dev)
        lab, _, _ = gt_labels(it["seg"], it.get("classes"), it.get("ignore"))
        if rule:
            q = (p.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy() if k <= 8 else None
            pl = postproc(p.argmax(0).to(torch.uint8).cpu().numpy(), q, rule)
            p = F.one_hot(torch.from_numpy(pl).long(), k).permute(3, 0, 1, 2).half()
        dv, per = dice_full_res(p, gfg, lab, k, dev)
        res["per_volume"].append(dv)
        res["per_class"].append(per)
        res["test_ids"].append(it["id"])
        print(f"[abl] test {len(res['test_ids'])}/{len(split['test'])} {it['id']}: dice {dv:.4f}", flush=True)
    subj = {it["id"]: it["subject"] for it in items}
    dvol = np.asarray(res["per_volume"], dtype=float)
    out = {"task": task, "head": kind, "variant": variant, "curve": curve, "tokens": str(tok.root),
           "data_seed": data_seed, "init_seed": SEED, "k": k,
           "recipe": {"lr": LR, "wd": WD, "epochs": epochs, "crops_per_volume": CROPS, "batch": BATCH, "crop_cells": CROP,
                      "sub": SUB, "hidden": HIDDEN, "fg": FG},
           "params": sum(p.numel() for p in head.parameters()), "n": {s: len(v) for s, v in split.items()},
           "train_ids": [it["id"] for it in split["train"]], "best_epoch": best["epoch"], "val_dice": best["val"],
           "val_per_volume": best["val_vol"], "curve_val": curve_, "postproc": {str(c): r for c, r in rule.items()},
           "test_dice": float(np.nanmean(dvol)) if len(dvol) else float("nan"),
           "test_ci95": subject_ci(lambda idx: np.nanmean(dvol[idx]), [subj[i] for i in res["test_ids"]])
           if len(dvol) else None, **res, "minutes": (time.time() - t0) / 60}
    dst = Path(out_dir) / task / f"{kind}__{variant}__d{data_seed}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    torch.save({"state": best["state"], "key": {k_: out[k_] for k_ in ("task", "head", "variant", "data_seed")}},
               dst.with_suffix(".best.pt"))   # the head kept (user: never delete checkpoints)
    print(f"[abl] {task} {kind} {variant} d{data_seed}: val {best['val']:.4f} (epoch {best['epoch']}) test "
          f"{out['test_dice']:.4f} {out['test_ci95']} -> {dst} ({out['minutes']:.1f} min)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--head", required=True, nargs="+", choices=["linear", "mlp", "primus"])
    ap.add_argument("--tok", required=True, help="abl/tokens.py folder of this task and variant")
    ap.add_argument("--curve", default="mean", help="the token files' curve tag")
    ap.add_argument("--variant", help="name in the json (default the curve)")
    ap.add_argument("--data-seed", type=int, default=0, help="1..3: 90 %% of the train subjects; 0: all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="tests only: the recipe is EPOCHS")
    ap.add_argument("--sample", type=int, help="tests: the first k volumes per split")
    a = ap.parse_args()
    for h in a.head:
        run(a.task, h, a.tok, a.curve, a.variant or a.curve, a.data_seed, a.out, a.epochs, a.sample)


if __name__ == "__main__":
    main()
