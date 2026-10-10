"""Runs inside a model's conda env: `python -m sfc_gdn2.fm.segrun <job.json>`. Segmentation through each model's
OFFICIAL downstream network (its repo's segmentation decoder, built and initialised as the repo does, encoder weights
loaded as the repo loads them) with the pretrained part FROZEN and the rest trained the same way for every model.

Wrapper hooks (fm/*.py):
- `seg_preprocess(image)` (optional, default the extraction path `preprocess` / `preprocess_cpu` + `preprocess_gpu`):
  the repo's preprocessing for segmentation, when it differs from its embedding pipeline.
- `seg_input(prepared) -> (x [C, *grid] float, m)`: the whole-volume network input on the model's grid and the 4x4
  map source voxel (i, j, k of the NIfTI given to the model) -> grid voxel.
- `seg_net(n_out, pretrained) -> (network, frozen, patch)`: the repo's downstream segmentation network with n_out
  output channels at the input resolution, `frozen` = module-name prefixes of what the repo loads from the
  checkpoint, `patch` = the repo's training patch (also the sliding-window tile; None = whole volume).

job = {"stage": "prep" | "train", "cfg": model config, "cache": dir, "device", "workers",
       "items": [{"id", "image", "seg", "split"}]  (prep: every volume; train: train/val/test of one task),
       train only: "k" (classes incl. background), "init": "pretrained" | "random", "iters", "val_every", "batch",
       "lr", "seed", "out": result json}
prep: <cache>/<id>.pt = {x fp16 [C, *grid], y uint8 [*grid] (GT pulled to the grid by nearest neighbour, 255 =
outside the GT), m, grid_from_gt (4x4 GT voxel -> grid voxel)}; shared by the models of a preprocessing group.
train: AdamW(lr, wd 1e-4) on the non-frozen parameters, poly decay, batches of `batch` patches (1/3 centred on a
voxel of a random foreground class present, nnU-Net's oversampling), loss CE + soft Dice (foreground classes,
batch Dice), bf16 autocast (fp32 if the wrapper sets `seg_bf16 = False`); every `val_every` iterations Dice on the val volumes (sliding window, Gaussian, step
0.5, on the grid), the best state kept. Test: class probabilities interpolated (trilinear, from_cells) to every GT
voxel, argmax, Dice per foreground class present in GT or prediction, mean per volume (as bench_seg.full_res_dice).
Only stdlib + numpy + torch + nibabel here.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..bench_geom import gt_labels
from .worker import build, prepared


def pull_labels(lab: np.ndarray, grid_from_gt: np.ndarray, grid, device) -> torch.Tensor:
    """GT labels at every grid voxel (nearest GT voxel, 255 where it falls outside the GT volume)."""
    inv = torch.tensor(np.linalg.inv(grid_from_gt), dtype=torch.float32, device=device)
    L = torch.from_numpy(lab).to(device)
    y = torch.full(tuple(grid), 255, dtype=torch.uint8, device=device)
    jk = torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in grid[1:]],
                                    indexing="ij")).reshape(2, -1)
    for i in range(grid[0]):  # one slab at a time
        g = torch.cat([torch.full_like(jk[:1], i), jk])
        s = (inv[:3, :3] @ g + inv[:3, 3:]).round().long()
        ok = ((s >= 0) & (s < torch.tensor(L.shape, device=device)[:, None])).all(0)
        row = torch.full((g.shape[1],), 255, dtype=torch.uint8, device=device)
        row[ok] = L[s[0, ok], s[1, ok], s[2, ok]].to(torch.uint8)
        y[i] = row.view(grid[1:])
    return y.cpu()


def prep(job: dict) -> None:
    w = build(job["cfg"], job["device"])
    cache = Path(job["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    items = [it for it in job["items"] if not (cache / f"{it['id']}.pt").exists()]
    fn = "seg_preprocess" if hasattr(w, "seg_preprocess") else None
    print(f"[seg] prep {job['cfg']['model']}: {len(items)} of {len(job['items'])} to do", flush=True)
    for it, x, err, t_pre in prepared(w, job["cfg"], items, int(job.get("workers", 0)), fn):
        try:
            if err:
                raise RuntimeError(err)
            if fn is None and int(job.get("workers", 0)) > 0 and hasattr(w, "preprocess_gpu"):
                x = w.preprocess_gpu(x)
            t0 = time.time()
            xin, m = w.seg_input(x)
            lab, gt_aff, names = gt_labels(it["seg"])
            grid_from_gt = np.asarray(m, dtype=np.float64) @ np.linalg.inv(nib.load(it["image"]).affine) @ gt_aff
            y = pull_labels(lab, grid_from_gt, tuple(xin.shape[1:]), job["device"])
            dst = cache / f"{it['id']}.pt"
            xin = xin.half() if xin.abs().max() < 6e4 else xin.float()  # raw intensities (BSF atlas) overflow fp16
            torch.save({"x": xin, "y": y, "m": np.asarray(m), "grid_from_gt": grid_from_gt,
                        "classes": names}, dst.with_suffix(".tmp"))
            os.replace(dst.with_suffix(".tmp"), dst)
            print(f"[seg] {it['id']}: grid {tuple(xin.shape)} fg {(y[y < 255] > 0).float().mean():.4f} "
                  f"pre={t_pre:.1f}s lab={time.time() - t0:.1f}s", flush=True)
        except Exception:  # noqa: BLE001 -- logged per item
            (cache / "_failed").mkdir(exist_ok=True)
            (cache / "_failed" / f"{it['id']}.txt").write_text(err or traceback.format_exc())
            print(f"[seg] FAILED {it['id']}: {(err or traceback.format_exc()).strip().splitlines()[-1]}", flush=True)


# ---------------------------------------------------------------------------------------------- training
def pad_to(x: torch.Tensor, y: torch.Tensor | None, patch) -> tuple:
    """Right-pad the spatial axes up to the patch (x with 0, y with 255): grid voxel indices are unchanged."""
    pad = [p for s, q in zip(reversed(x.shape[1:]), reversed(patch)) for p in (0, max(q - s, 0))]
    return F.pad(x, pad), (None if y is None else F.pad(y, pad, value=255))


def gaussian(patch, device) -> torch.Tensor:
    axes = [torch.exp(-0.5 * ((torch.arange(p, device=device) - (p - 1) / 2) / (p / 8)) ** 2) for p in patch]
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (g / g.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding(net, x: torch.Tensor, patch, n_out: int, amp) -> torch.Tensor:
    """[C, *S] -> class probabilities [n_out, *S] (Gaussian-weighted tiles, step 0.5 of the patch)."""
    S = tuple(x.shape[1:])
    patch = S if patch is None else tuple(patch)
    xp, _ = pad_to(x, None, patch)
    P = tuple(xp.shape[1:])
    starts = [np.unique(np.round(np.linspace(0, s - p, math.ceil((s - p) / (p / 2)) + 1)).astype(int)) if s > p else [0]
              for s, p in zip(P, patch)]
    g = gaussian(patch, x.device)
    try:  # fp16 accumulator on the device, else on the CPU (nnU-Net's predictor does both)
        out = torch.zeros(n_out, *P, dtype=torch.half, device=x.device)
    except torch.cuda.OutOfMemoryError:
        out = torch.zeros(n_out, *P, dtype=torch.half)
    wsum = torch.zeros(P, device=x.device)
    for a, b, c in product(*starts):
        sl = (slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2]))
        with amp():
            logits = net(xp[(slice(None), *sl)][None])
        out[(slice(None), *sl)] += (logits[0].float().softmax(0) * g).to(out)
        wsum[sl] += g
    return out.div_(wsum.to(out.device))[(slice(None),) + tuple(slice(0, s) for s in S)]


def dice_on(pred: torch.Tensor, y: torch.Tensor, k: int) -> float:
    ok = y != 255
    p, t = pred[ok], y[ok].long()
    ds = [2 * ((p == c) & (t == c)).sum().item() / ((p == c).sum() + (t == c).sum()).item()
          for c in range(1, k) if (t == c).any() or (p == c).any()]
    return float(np.mean(ds)) if ds else float("nan")


@torch.no_grad()
def dice_full_res(probs: torch.Tensor, grid_from_gt: np.ndarray, lab: np.ndarray, k: int,
                  dev: str | torch.device | None = None) -> tuple[float, list]:
    """Probabilities [k, *grid] (on any device) -> every GT voxel (trilinear, border clamped), argmax, Dice per class;
    slab-wise, moving to `dev` only the part of the grid each GT slab samples."""
    dev = dev or probs.device
    M = torch.tensor(grid_from_gt, dtype=torch.float32, device=dev)
    G = torch.tensor(probs.shape[1:], device=dev)
    jk = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in lab.shape[1:]],
                                    indexing="ij")).reshape(2, -1)
    L = torch.from_numpy(lab).to(dev)
    tp, np_, nt = (torch.zeros(k, device=dev) for _ in range(3))
    for i in range(lab.shape[0]):
        c = M[:3, :3] @ torch.cat([torch.full_like(jk[:1], i), jk]) + M[:3, 3:]
        lo = torch.minimum(c.amin(1).floor().long().clamp_min(0), G - 1)  # the slab's grid box, border clamping kept
        hi = torch.maximum(torch.minimum(c.amax(1).floor().long() + 2, G), lo + 1)
        sub = probs[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].to(dev).float()
        g = (2 * (c - lo[:, None] + 0.5) / (hi - lo)[:, None] - 1).flip(0).T.reshape(1, 1, 1, -1, 3)  # x=W, y=H, z=D
        pred = F.grid_sample(sub[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0, 0]
        pred = pred.argmax(0)
        t = L[i].reshape(-1).long()
        tp += torch.bincount(t[pred == t], minlength=k)[:k].float()
        np_ += torch.bincount(pred, minlength=k)[:k].float()
        nt += torch.bincount(t, minlength=k)[:k].float()
    per = [(2 * tp[c] / (np_[c] + nt[c])).item() if (np_[c] + nt[c]) > 0 else float("nan") for c in range(1, k)]
    return (float(np.nanmean(per)) if not all(math.isnan(d) for d in per) else float("nan")), per


class Patches:
    """Random training patches from volumes held in RAM; 1/3 centred on a voxel of a random present class."""

    def __init__(self, vols: list[dict], patch, k: int, rng: np.random.Generator):
        self.vols, self.patch, self.rng = [], patch, rng
        for v in vols:
            x, y = pad_to(v["x"], v["y"], patch) if patch else (v["x"], v["y"])
            fg = {}
            for c in range(1, k):
                idx = (y == c).nonzero()
                if len(idx):
                    fg[c] = idx[torch.from_numpy(rng.choice(len(idx), min(len(idx), 2000), replace=False))]
            self.vols.append((x, y, fg))

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(n):
            x, y, fg = self.vols[self.rng.integers(len(self.vols))]
            S = np.array(y.shape)
            P = S if self.patch is None else np.array(self.patch)
            if fg and self.rng.random() < 1 / 3:
                vox = fg[list(fg)[self.rng.integers(len(fg))]]
                c = vox[self.rng.integers(len(vox))].numpy()
                start = np.clip(c - P // 2, 0, S - P)
            else:
                start = np.array([self.rng.integers(s - p + 1) for s, p in zip(S, P)])
            sl = tuple(slice(int(a), int(a + p)) for a, p in zip(start, P))
            xs.append(x[(slice(None), *sl)].float())  # the cache mixes fp16 and fp32 volumes
            ys.append(y[sl])
        return torch.stack(xs), torch.stack(ys)


def loss_fn(logits: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    y = y.long()
    ce = F.cross_entropy(logits.float(), y, ignore_index=255)
    ok = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * ok
    t = F.one_hot(torch.where(y == 255, 0, y), k).permute(0, 4, 1, 2, 3).float() * ok
    dims = (0, 2, 3, 4)
    dice = (2 * (p * t).sum(dims)[1:] + 1e-5) / (p.sum(dims)[1:] + t.sum(dims)[1:] + 1e-5)
    return ce + 1 - dice.mean()


def train(job: dict) -> None:
    dev = torch.device(job["device"])
    rng = np.random.default_rng(job["seed"])
    k, cache = int(job["k"]), Path(job["cache"])
    w = build(job["cfg"], job["device"])
    torch.manual_seed(job["seed"])
    ref, frozen, patch = w.seg_net(k, pretrained=False)
    torch.manual_seed(job["seed"])  # same random init: the pretrained build differs exactly where the checkpoint loads
    pre = w.seg_net(k, pretrained=True)[0]
    ref_p = dict(ref.named_parameters())
    loaded = {n for n, p in pre.named_parameters() if not torch.equal(p.detach(), ref_p[n].detach())}
    is_frozen = lambda n: any(n == f or n.startswith(f + ".") for f in frozen)
    names = {n for n, _ in pre.named_parameters()}
    frozen_names = {n for n in names if is_frozen(n)} | loaded
    print(f"[seg] loaded from the checkpoint: {len(loaded)} tensors; frozen prefixes {frozen}: "
          f"{len(frozen_names - loaded)} frozen but not loaded {sorted(frozen_names - loaded)[:4]}, "
          f"{len(loaded - {n for n in names if is_frozen(n)})} loaded outside the prefixes (frozen too)", flush=True)
    net = (pre if job["init"] == "pretrained" else ref).to(dev)
    del ref, pre, ref_p
    n_frozen = n_train = 0
    for n, p in net.named_parameters():
        p.requires_grad_(n not in frozen_names)
        n_frozen += p.numel() * (not p.requires_grad)
        n_train += p.numel() * p.requires_grad
    frozen_mods = [m for n, m in net.named_modules() if n and is_frozen(n)]
    print(f"[seg] {job['cfg']['model']} init={job['init']} patch={patch} frozen={n_frozen / 1e6:.1f}M "
          f"trained={n_train / 1e6:.1f}M prefixes={frozen}", flush=True)
    amp = contextlib.nullcontext  # wrappers with `seg_bf16 = False` run fp32 (their repo's precision)
    if dev.type == "cuda" and torch.cuda.is_bf16_supported() and getattr(w, "seg_bf16", True):
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    def load(it):
        r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
        return {"x": r["x"], "y": r["y"], "grid_from_gt": r["grid_from_gt"], "id": it["id"]}

    split = {s: [it for it in job["items"] if it["split"] == s and (cache / f"{it['id']}.pt").exists()]
             for s in ("train", "val", "test")}
    train_v = [load(it) for it in split["train"]]
    val_v = [load(it) for it in split["val"]]
    data = Patches(train_v, patch, k, rng)
    del train_v
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=1e-4)
    iters, every = int(job["iters"]), int(job["val_every"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / iters) ** 0.9)
    best, curve, t0 = {"val": -1.0}, [], time.time()

    def set_mode():
        net.train()
        for m in frozen_mods:
            m.eval()

    set_mode()
    for it in range(1, iters + 1):
        x, y = data.sample(int(job["batch"]))
        with amp():
            logits = net(x.to(dev, non_blocking=True).float())
        loss = loss_fn(logits, y.to(dev), k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 12)
        opt.step()
        sched.step()
        if it % every == 0 or it == iters:
            net.eval()
            ds = [dice_on(sliding(net, v["x"].to(dev).float(), patch, k, amp).argmax(0).to(dev), v["y"].to(dev), k)
                  for v in val_v]
            val = float(np.nanmean(ds))
            curve.append({"iter": it, "loss": float(loss), "val_dice": val, "min": (time.time() - t0) / 60})
            print(f"[seg] it {it} loss {float(loss):.3f} val dice {val:.4f} ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
            if "state" not in best or val > best["val"]:
                best = {"val": val, "iter": it, "state": copy.deepcopy({n: t.cpu() for n, t in net.state_dict().items()})}
            set_mode()
    net.load_state_dict(best.pop("state"))
    net.eval()
    del val_v, data
    res = {"per_volume": [], "per_class": [], "test_ids": []}
    for it in split["test"]:
        v = load(it)
        probs = sliding(net, v["x"].to(dev).float(), patch, k, amp)
        lab, _, _ = gt_labels(it["seg"])
        d, per = dice_full_res(probs, v["grid_from_gt"], lab, k, dev)
        res["per_volume"].append(d)
        res["per_class"].append(per)
        res["test_ids"].append(it["id"])
    out = {"model": job["cfg"]["model"], "init": job["init"], "lr": job["lr"], "iters": iters, "patch": patch,
           "frozen_prefixes": frozen, "params_frozen": n_frozen, "params_trained": n_train, "best": best,
           "curve": curve, "n": {s: len(v) for s, v in split.items()},
           "test_dice": float(np.nanmean(res["per_volume"])), **res}
    Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["out"]).write_text(json.dumps(out, indent=1))
    print(f"[seg] test dice {out['test_dice']:.4f} on {len(res['per_volume'])} volumes -> {job['out']}", flush=True)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    {"prep": prep, "train": train}[job["stage"]](job)


if __name__ == "__main__":
    main(sys.argv[1])
