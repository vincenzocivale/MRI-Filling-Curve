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
- `seg_context(prepared) -> tensor` (optional): per-volume input the network takes besides the image, called as
  `net(x, context)` with contexts stacked over the batch (ours: the voxel spacing, fm/sfc.py).

job = {"stage": "prep" | "train", "cfg": model config, "cache": dir, "device", "workers",
       "items": [{"id", "image", "seg", "split"}]  (prep: every volume; train: train/val/test of one task),
       train only: "k" (classes incl. background), "init": "pretrained" | "random", "iters", "val_every", "batch",
       "lr", "seed", "out": result json}
prep: <cache>/<id>.pt = {x fp16 [C, *grid], y uint8 [*grid] (GT pulled to the grid by nearest neighbour, 255 =
outside the GT), m, grid_from_gt (4x4 GT voxel -> grid voxel)}; shared by the models of a preprocessing group.
train: AdamW(lr, wd 1e-4) on the non-frozen parameters, poly decay, batches of `batch` patches (1/3 centred on a
voxel of a random foreground class present, nnU-Net's oversampling), loss CE + soft Dice (foreground classes,
batch Dice), bf16 autocast (fp32 if the wrapper sets `seg_bf16 = False`); every `val_every` iterations Dice on the val volumes (sliding window, Gaussian, step
0.5, on the grid), the best state kept. Test: class probabilities interpolated (trilinear) to every GT
voxel, argmax, Dice per foreground class present in GT or prediction, mean per volume .
A killed job resumes from what it left next to the result json: the training state (every 10 min, SFC_SEG_SAVE_S),
the best state, the test volumes done.
Only stdlib + numpy + torch + nibabel here.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..bench.geom import gt_labels
from .worker import build, fail, prepared

# token-cache runs (20/10): tiles of TILE_CELLS 16 mm cells per axis (16 cm, as the resampled models' 160 voxels at
# ~1 mm) instead of 160 native voxels (67 cm of TotalSeg's 4.2 mm voxels, 7 cm of SOOP's 0.45 mm)
TILE_CELLS = 10
AHEAD = 6   # batches whose cached tokens are read in advance
TILE_MAX = 1e9   # voxels x classes of one tile
SAVE_S = float(os.environ.get("SFC_SEG_SAVE_S", "600"))  # training state to disk every SAVE_S seconds


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
            ctx = w.seg_context(x) if hasattr(w, "seg_context") else None
            lab, gt_aff, names = gt_labels(it["seg"], it.get("classes"), it.get("ignore"))
            grid_from_gt = np.asarray(m, dtype=np.float64) @ np.linalg.inv(nib.load(it["image"]).affine) @ gt_aff
            y = pull_labels(lab, grid_from_gt, tuple(xin.shape[1:]), job["device"])
            dst = cache / f"{it['id']}.pt"
            xin = xin.half() if xin.abs().max() < 6e4 else xin.float()  # raw intensities (BSF atlas) overflow fp16
            torch.save({"x": xin, "y": y, "m": np.asarray(m), "grid_from_gt": grid_from_gt,
                        "classes": names, "ctx": ctx}, dst.with_suffix(".tmp"))
            os.replace(dst.with_suffix(".tmp"), dst)
            print(f"[seg] {it['id']}: grid {tuple(xin.shape)} fg {(y[y < 255] > 0).float().mean():.4f} "
                  f"pre={t_pre:.1f}s lab={time.time() - t0:.1f}s", flush=True)
        except Exception:  # noqa: BLE001 -- logged per item
            fail(cache, it["id"], err or traceback.format_exc())


# ---------------------------------------------------------------------------------------------- training
def pad_to(x: torch.Tensor, y: torch.Tensor | None, patch) -> tuple:
    """Right-pad the spatial axes up to the patch (x with 0, y with 255): grid voxel indices are unchanged."""
    pad = [p for s, q in zip(reversed(x.shape[1:]), reversed(patch)) for p in (0, max(q - s, 0))]
    return F.pad(x, pad), (None if y is None else F.pad(y, pad, value=255))


def gaussian(patch, device) -> torch.Tensor:
    axes = [torch.exp(-0.5 * ((torch.arange(p, device=device) - (p - 1) / 2) / (p / 8)) ** 2) for p in patch]
    g = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (g / g.max()).clamp_min(1e-3)


def call(net, x: torch.Tensor, ctx):
    return net(x) if ctx is None else net(x, ctx.to(x.device))


def tile_of(patch, align, k: int = 2):
    """The tile in voxels: the wrapper's patch, or under the token cache TILE_CELLS cells of 16 mm per axis (`align`),
    fewer where voxels x classes would pass TILE_MAX (TotalSeg's 0.25 mm scans: 640 x 640 x 200 voxels x 50 classes,
    16 GB for one softmax, OOM even one scan at a time; never with 2 classes)."""
    if align is None:
        return patch
    n = TILE_CELLS
    while n > 1 and math.prod(n * a for a in align) * k > TILE_MAX:
        n -= 1
    return [n * a for a in align]


@torch.no_grad()
def sliding(net, x: torch.Tensor, patch, n_out: int, amp, ctx=None) -> torch.Tensor:
    """[C, *S] -> class probabilities [n_out, *S] (Gaussian-weighted tiles, step 0.5 of the patch); ours (fm/sfc.SegNet,
    the cascade) the whole scan in one pass, coarse to fine."""
    lm = getattr(net, "landmark", False)   # landmark heatmaps: sigmoid per channel, the generic tile path
    if getattr(net, "cube", False) and not net.preparing and ctx is not None:   # fm/dense_dec cubes: placed on this scan
        net.prepare(x, ctx, lambda: sliding(net, x, patch, n_out, amp, ctx))
    if getattr(net, "cascade", False) and ctx is not None:
        with amp():
            return net.landmark_predict(x, ctx.to(x.device), n_out) if lm else net.cascade_predict(x, ctx.to(x.device), n_out)
    S = tuple(x.shape[1:])
    patch = S if patch is None else tuple(patch)
    xp, _ = pad_to(x, None, patch)
    P = tuple(xp.shape[1:])
    starts = [np.unique(np.round(np.linspace(0, s - p, math.ceil((s - p) / (p / 2)) + 1)).astype(int)) if s > p else [0]
              for s, p in zip(P, patch)]
    g = gaussian(patch, x.device)
    # fp16 accumulator on the device if it takes <= 1/4 of the free memory, else on the CPU (nnU-Net's predictor does
    # both; "fits at all" left the tiles' forward no room: nnF CNN OOM on whole-body TotalSeg grids)
    fits = x.is_cuda and n_out * math.prod(P) * 2 <= torch.cuda.mem_get_info(x.device)[0] / 4
    out = torch.zeros(n_out, *P, dtype=torch.half, device=x.device if fits else "cpu")
    wsum = torch.zeros(P, device=x.device)
    for a, b, c in product(*starts):   # one tile at a time
        sl = (slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2]))
        with amp():
            cx = ctx if ctx is None or not getattr(net, "tile_corner", False) else \
                torch.cat([ctx.float(), torch.tensor([a, b, c], dtype=torch.float)])   # fm/dense_dec: tokens by corner
            lg = call(net, xp[(slice(None), *sl)][None], None if cx is None else cx[None])[0]
        out[(slice(None), *sl)] += ((lg.float().sigmoid() if lm else lg.float().softmax(0)) * g).to(out)
        wsum[sl] += g
    wsum = wsum.to(out.device)
    for c in range(n_out):  # one class at a time: dividing the whole fp16 accumulator by an fp32 tensor made an fp32
        out[c].div_(wsum)   # copy of it (whole-body nnF grids: 60 GB accumulator, RAM OOM at 240 GB)
    return out[(slice(None),) + tuple(slice(0, s) for s in S)]


def dice_on(pred: torch.Tensor, y: torch.Tensor, k: int) -> float:
    ok = y != 255
    p, t = pred[ok], y[ok].long()
    ds = [2 * ((p == c) & (t == c)).sum().item() / ((p == c).sum() + (t == c).sum()).item()
          for c in range(1, k) if (t == c).any() or (p == c).any()]
    return float(np.mean(ds)) if ds else float("nan")


MISS_MM = 50.0   # a landmark not predicted at all counts as this radial error in the val selection


LANDMARK_SIGMA_MM = 3.0   # landmark heatmaps (user, 23/10: landmarks are a regression, not a segmentation)


def heat_spec(lm, start) -> torch.Tensor:
    """A tile's landmarks for heatmaps(): [n, 6] = their voxels relative to the tile's corner `start` (NaN if absent)
    and the grid spacing (mm). The tiles carry this, not the maps (27/10: the dense maps of whole fetal scans, built on
    the CPU by the one drawing thread, were ~5 s of every iteration)."""
    xyz, sp = lm
    rel = xyz.float() - torch.as_tensor(np.asarray(start), dtype=torch.float)
    return torch.cat([rel, torch.as_tensor(np.asarray(sp), dtype=torch.float).expand(len(rel), 3)], 1)


def heatmaps(spec: torch.Tensor, P) -> torch.Tensor:
    """heat_spec [..., n, 6] -> [..., n, *P] Gaussian heatmaps on spec's device; exactly 1 at each landmark's voxel."""
    ax = [(torch.arange(p, device=spec.device) - spec[..., a:a + 1]) * spec[..., 3 + a:4 + a] for a, p in enumerate(P)]
    g = [torch.exp(-0.5 * (d / LANDMARK_SIGMA_MM) ** 2).nan_to_num(0.0) for d in ax]
    return (g[0][..., :, None, None] * g[1][..., None, :, None] * g[2][..., None, None, :]).half()


def focal_heatmap(logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss of heatmap logits [B, n, *P] against targets h (1 at the landmark)."""
    pos = h > 0.999
    lp, ln = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = lp.exp()
    neg = (p ** 2 * (1 - h) ** 4 * ln)[~pos]
    if not pos.any():   # no landmark in the tile: the mean, not a sum over millions of voxels divided by 1
        return -neg.mean()
    return -(((1 - p) ** 2 * lp)[pos].sum() + neg.sum()) / pos.sum()


def landmark_step(net, x, y, c, amp, dev) -> torch.Tensor:
    """Landmark tasks: channels 1.. of the decoder are heatmaps (sigmoid), y the tiles' target heatmaps; one scan
    at a time, every model's decoder as it is."""
    parts = list(zip(x, y, c if c is not None else [None] * len(x))) if isinstance(x, list) else \
        [(x[i:i + 1], y[i:i + 1], None if c is None else c[i:i + 1]) for i in range(len(x))]
    parts = [(a, heatmaps(b.to(dev), a.shape[2:]), u) for a, b, u in parts]   # the maps on the GPU (heat_spec)
    if getattr(net, "cascade", False):   # ours: every tile in one call, the cascade's levels read for all at once (24/10)
        with amp():
            loss = net.landmark_sums([a.to(dev).float() for a, _, _ in parts], [b for _, b, _ in parts],
                                     [u.to(dev) for _, _, u in parts], parts[0][1].shape[1] + 1)
        loss.backward()
        return loss.detach()
    tot = 0.0
    for xi, yi, ci in parts:   # the other models: one scan at a time
        with amp():
            lg = call(net, xi.to(dev).float(), ci)
        loss = focal_heatmap(lg[:, 1:].float(), yi.to(dev).float()) / len(parts)
        loss.backward()
        tot += float(loss)
    return torch.tensor(tot)


def landmark(p: torch.Tensor) -> torch.Tensor:
    """One point per landmark, always (user, 23/10): the centroid of the voxels at the maximum of the class's
    probability map p [*S] (ties of the uint8 saves included)."""
    return (p >= p.max()).nonzero().float().mean(0)


def mre_on(pred: torch.Tensor, y: torch.Tensor, spacing, k: int) -> float:
    """Landmarks: mean radial error (mm) between the GT centroid of each class present in the GT and the predicted point,
    on the model grid (canonical: axis permutations / flips of the native voxels, so distances are the world's).
    pred: class probabilities [k, *S] -> landmark() (every landmark gets a point); labels [*S] (the cascade's val
    pass) -> centroid of the class, a class not predicted counts MISS_MM."""
    sp = torch.as_tensor(spacing, dtype=torch.float, device=pred.device)
    err = []
    for c in range(1, k):
        g = (y == c).nonzero().float()
        if not len(g):
            continue
        if pred.dim() == y.dim() + 1:
            err.append(float(((g.mean(0) - landmark(pred[c])) * sp).norm()))
            continue
        q = (pred == c).nonzero().float()
        err.append(float(((g.mean(0) - q.mean(0)) * sp).norm()) if len(q) else MISS_MM)
    return float(np.mean(err)) if err else float("nan")


@torch.no_grad()
def dice_full_res(probs: torch.Tensor, grid_from_gt: np.ndarray, lab: np.ndarray, k: int,
                  dev: str | torch.device | None = None) -> tuple[float, list]:
    """Probabilities [k, *grid] (on any device) -> every GT voxel (trilinear, border clamped), argmax, Dice per class;
    slab-wise, moving to `dev` only the part of the grid each GT slab samples."""
    dev = dev or probs.device
    if probs[0].numel() == 0:  # the pipeline cropped the volume to nothing (no foreground): background everywhere
        probs = F.one_hot(torch.zeros(1, 1, 1, dtype=torch.long), k).permute(3, 0, 1, 2).float()
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
        keep = t != 255   # ignored voxels (geom.gt_labels `ignore`): neither scored nor counted as predictions
        pred, t = pred[keep], t[keep]
        tp += torch.bincount(t[pred == t], minlength=k)[:k].float()
        np_ += torch.bincount(pred, minlength=k)[:k].float()
        nt += torch.bincount(t, minlength=k)[:k].float()
    per = [(2 * tp[c] / (np_[c] + nt[c])).item() if (np_[c] + nt[c]) > 0 else float("nan") for c in range(1, k)]
    return (float(np.nanmean(per)) if not all(math.isnan(d) for d in per) else float("nan")), per


class Patches:
    """Random training patches from volumes held in RAM; 1/3 centred on a voxel of a random present class."""

    def __init__(self, vols: list[dict], patch, k: int, rng: np.random.Generator):
        self.vols, self.ctx, self.patch, self.rng, self.align, self.lm = [], [], [], rng, [], []
        for v in vols:
            self.lm.append(v.get("lm"))
            pv = tile_of(patch, v.get("align"), k)
            x, y = pad_to(v["x"], v["y"], pv) if pv else (v["x"], v["y"])
            self.ctx.append(v.get("ctx"))
            self.align.append(v.get("align"))
            self.patch.append(pv)
            fg = {}
            for c in range(1, k):
                idx = (y == c).nonzero()
                if len(idx):
                    fg[c] = idx[torch.from_numpy(rng.choice(len(idx), min(len(idx), 2000), replace=False))]
            self.vols.append((x, y, fg))

    def sample(self, n: int) -> tuple:
        """Batches of n patches; tiles of different shapes (cm-sized tiles of scans with different spacings) come as
        lists of [1, ...] tensors."""
        xs, ys, cs = [], [], []
        for _ in range(n):
            j = self.rng.integers(len(self.vols))
            x, y, fg = self.vols[j]
            cs.append(self.ctx[j])
            S = np.array(y.shape)
            P = S if self.patch[j] is None else np.array(self.patch[j])
            if fg and self.rng.random() < 1 / 3:
                vox = fg[list(fg)[self.rng.integers(len(fg))]]
                c = vox[self.rng.integers(len(vox))].numpy()
                start = np.clip(c - P // 2, 0, S - P)
            else:
                start = np.array([self.rng.integers(s - p + 1) for s, p in zip(S, P)])
            if self.align[j] is not None:   # on the volume's 16 mm grid (token cache)
                start = (start // np.array(self.align[j])) * np.array(self.align[j])
                cs[-1] = torch.cat([cs[-1], torch.tensor(start, dtype=torch.float)])
            sl = tuple(slice(int(a), int(a + p)) for a, p in zip(start, P))
            xs.append(x[(slice(None), *sl)].float())  # the cache mixes fp16 and fp32 volumes
            ys.append(y[sl] if self.lm[j] is None else heat_spec(self.lm[j], start))
        if len({t.shape for t in xs}) > 1:
            return [t[None] for t in xs], [t[None] for t in ys], None if cs[0] is None else [t[None] for t in cs]
        return torch.stack(xs), torch.stack(ys), None if cs[0] is None else torch.stack(cs)


def batch_sums(logits: torch.Tensor, y: torch.Tensor, k: int) -> tuple:
    """(sum of the cross-entropy, labelled voxels, per class: soft intersection, predicted sum, target sum)."""
    y = y.long()
    ce = F.cross_entropy(logits.float(), y, ignore_index=255, reduction="sum")
    ok = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * ok
    yc = torch.where(y == 255, 0, y)
    inter = p.new_zeros(k).scatter_add(0, yc.flatten(), p.gather(1, yc.unsqueeze(1)).flatten())
    return ce, ok.sum(), inter, p.sum((0, 2, 3, 4)), p.new_zeros(k).scatter_add(0, yc.flatten(), ok.flatten())


def micro_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """The gradient of loss_fn on the whole batch, one scan at a time (volumes of several shapes: whole-volume
    models, patch None; or a wrapper's `seg_microbatch` with many classes). The soft Dice sums over the batch before dividing, so a first
    pass without gradient gives the batch's sums; then each scan backpropagates a loss linear in its own sums with
    the batch's derivatives as coefficients: the same gradient as the batch of 2, one more forward pass."""
    parts = list(zip(x, y, c if c is not None else [None] * len(x))) if isinstance(x, list) else \
        [(x[i:i + 1], y[i:i + 1], None if c is None else c[i:i + 1]) for i in range(len(x))]
    sums = []
    for xi, yi, ci in parts:
        with torch.no_grad(), amp():
            sums.append(batch_sums(call(net, xi.to(dev).float(), ci), yi.to(dev), k))
    n = sum(s[1] for s in sums)
    inter, psum, tsum = (sum(s[j] for s in sums) for j in (2, 3, 4))
    den = psum + tsum + 1e-5
    a = -2 / den / (k - 1)
    b = (2 * inter + 1e-5) / den ** 2 / (k - 1)
    a[0] = b[0] = 0
    for xi, yi, ci in parts:
        with amp():
            logits = call(net, xi.to(dev).float(), ci)
        ce, _, i_, p_, _ = batch_sums(logits, yi.to(dev), k)
        (ce / n + (a * i_).sum() + (b * p_).sum()).backward()
    dice = (2 * inter[1:] + 1e-5) / den[1:]
    return sum(s[0] for s in sums) / n + 1 - dice.mean()


def sums_loss(t: list) -> torch.Tensor:
    """SegNet.cascade_sums' sums -> CE per voxel + 1 - mean soft Dice of the classes > 0 (loss_fn's)."""
    ce, n, inter, psum, tsum = t
    return ce / n.clamp_min(1) + 1 - ((2 * inter[1:] + 1e-5) / (psum[1:] + tsum[1:] + 1e-5)).mean()


def cascade_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """Ours (fm/sfc.SegNet, the Dmix cascade): the point head's loss plus head_w x each level head's, one backward; tiles
    of several shapes in one call (24/10: the levels read for all of them at once)."""
    if isinstance(x, list):
        x, y, c = [t.to(dev).float() for t in x], [u.to(dev) for u in y], [v.to(dev) for v in c]
    else:
        x, y, c = x.to(dev).float(), y.to(dev), c.to(dev)
    with amp():
        fin, aux = net.cascade_sums(x, y, c, k)
    loss = sums_loss(fin) + net.head_w * sum(sums_loss(a) for a in aux)
    if getattr(net, "_rloss", None) is not None:   # cascade_rank: the ranker's ListNet (its own parameters)
        loss, net._rloss = loss + net._rloss, None
    loss.backward()
    return loss.detach()


def loss_fn(logits: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    y = y.long()
    ce = F.cross_entropy(logits.float(), y, ignore_index=255)
    ok = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * ok
    # soft Dice per class without a one-hot target (the same sums: with ~50 classes on a 160^3 batch of 2 the int64
    # one-hot and its float copies took ~6 GB, GPU OOM for decoder F on TotalSeg, 18/10)
    yc = torch.where(y == 255, 0, y)
    inter = p.new_zeros(k).scatter_add(0, yc.flatten(), p.gather(1, yc.unsqueeze(1)).flatten())   # p is 0 off `ok`
    tsum = p.new_zeros(k).scatter_add(0, yc.flatten(), ok.flatten())
    dice = (2 * inter[1:] + 1e-5) / (p.sum((0, 2, 3, 4))[1:] + tsum[1:] + 1e-5)
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
    LM = job.get("select") == "mre"   # landmark task: heatmap regression (landmark_step, sigmoid maps, MRE selection)
    net.landmark = LM
    DET = job.get("select") == "det"   # point detection (cmbpts, user 28/10): component centroids matched within DET_MM
    # the true points: the task's points csv (id, i, j, k: GT voxels) when given, else the label's components' centroids
    # (2 mm spheres closer than ~4 mm merge: ~3 % of the cmbpts lesions, Ettore 28/10)
    PTS = pd.read_csv(job["items"][0]["points"]).groupby("id")[["i", "j", "k"]] if DET and job["items"][0].get("points") else None
    net.lm_prior = float(os.environ.get("SFC_LM_PRIOR", "0"))   # landmark_predict: + the level heads' log-sigmoid (diagnosis)
    if getattr(net, "cascade", False):   # our decoder trains on large tiles (whole scans mostly): 16 cells of 16 mm per
        global TILE_CELLS, TILE_MAX         # axis, capped by voxels only (no dense logits: points)
        TILE_CELLS, TILE_MAX = 16, 3e8 * k
    if hasattr(net, "tile_max"):   # fm/dense_dec: dense activations, tiles of ~160^3 voxels on the token grid
        TILE_MAX = net.tile_max(k)
    # token cache (SegNet args.token_cache): tiles start on each volume's 16 mm grid and pass (volume, corner) in ctx
    tcache = getattr(net, "cache", None)
    vid = {it["id"]: n for n, it in enumerate(job["items"])}
    if tcache is not None:
        tcache.ids = [it["id"] for it in job["items"]]
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
        ctx = r.get("ctx")
        if ctx is not None and getattr(w, "seg_extent", False):   # + the scan's shape: the net can skip the padding
            ctx = torch.cat([ctx.float(), torch.tensor(r["x"].shape[1:], dtype=torch.float)])
        if tcache is not None:   # + the volume's index in the token cache; tiles append their corner
            ctx = torch.cat([ctx, torch.tensor([float(vid[it["id"]])])])
            if hasattr(tcache, "ensure"):   # LiveCache: its levels read and written now if the folder lacks them
                tcache.ensure(it["id"], r["x"], r["ctx"])
        v = {"x": r["x"], "y": r["y"], "grid_from_gt": r["grid_from_gt"], "id": it["id"], "ctx": ctx,
             "align": None if tcache is None else [4 * a for a in tcache.meta(it["id"])["k4"]]}
        if DET:
            v["sp"], v["pts"] = grid_spacing(r["grid_from_gt"], it["seg"]), true_points(PTS, it["id"], r["grid_from_gt"])
        if LM:   # model-grid spacing (mm: GT zooms through the grid mapping) and each landmark's voxel on the grid
            sp = grid_spacing(r["grid_from_gt"], it["seg"])
            y = r["y"][0] if r["y"].dim() == 4 else r["y"]
            xyz = torch.stack([(y == c).nonzero().float().mean(0).round() for c in range(1, k)])   # NaN if absent
            v["sp"], v["lm"] = sp, (xyz, sp)
        return v

    split = {s: [it for it in job["items"] if it["split"] == s and (cache / f"{it['id']}.pt").exists()]
             for s in ("train", "val", "test")}
    # on disk as the run goes, so a killed job resumes: the training state every 10 min (.last.pt), the best state
    # until the test is written (.best.pt), the test volumes done (.test.json)
    out_p = Path(job["out"])
    if hasattr(net, "cube_dir"):   # fm/dense_dec cubes: the training scans' cubes kept next to the result (resumes reload them)
        net.cube_dir = out_p.with_suffix(".cubes")
    ckpt, last, part = (out_p.with_suffix(s) for s in (".best.pt", ".last.pt", ".test.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # + the wrapper's whole config and the tile rule: a run of another decoder in the same folder is never resumed
    # (20/10: the token-cache runs resumed the cancelled tile runs' states, same iters / lr / seed)
    key = {n: job[n] for n in ("init", "iters", "val_every", "batch", "lr", "seed")} | {"train": len(split["train"])} | \
        {"cfg": json.dumps(job["cfg"], sort_keys=True, default=str), "tile_cells": TILE_CELLS}
    saved = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.exists() else {}
    iters, every = int(job["iters"]), int(job["val_every"])

    def trained() -> dict:  # what training changes (frozen parameters stay as built)
        return {n: t.detach().to("cpu", copy=True) for n, t in net.state_dict().items() if n not in frozen_names}

    def load_trained(state: dict) -> None:
        # keys checked here: some repos' nets (FOMO26 / asparagus) override load_state_dict and return None
        names = set(net.state_dict())
        unexpected, missing = set(state) - names, names - set(state)
        assert not unexpected and missing <= frozen_names, (unexpected, missing)
        net.load_state_dict(state, strict=False)

    if saved.get("key") == key:
        best, curve = saved["best"], saved["curve"]
        print(f"[seg] best state from {ckpt} (iter {best['iter']}), training skipped", flush=True)
    else:
        train_v = [load(it) for it in split["train"]]
        val_v = [load(it) for it in split["val"]]
        data = Patches(train_v, patch, k, rng)
        del train_v
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=job["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / iters) ** 0.9)
        best, curve, start, t0 = {"val": -1.0}, [], 1, time.time()
        state = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else {}
        if state.get("key") == key:
            load_trained(state["net"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            rng.bit_generator.state = state["rng"]  # after Patches, which draws from it as in the first job
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"] is not None and dev.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"])
            best, curve, start, t0 = state["best"], state["curve"], state["it"] + 1, t0 - state["min"] * 60
            print(f"[seg] resumed from {last} at iter {state['it']} ({state['min']:.1f} min done)", flush=True)
            if hasattr(net, "on_val"):   # fm/dense_dec cubes: the training scans' cubes live in RAM, placed again
                net.eval()
                net.on_val(0, every, [(xv, cv) for (xv, _, _), cv in zip(data.vols, data.ctx)],
                           lambda xv, cv: sliding(net, xv.to(dev).float(), patch, k, amp, cv), resume=True)
        t_save = time.time()

        def save_last(it: int) -> None:
            torch.save({"key": key, "it": it, "net": trained(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                        "best": best, "curve": curve, "min": (time.time() - t0) / 60}, last.with_suffix(".tmp"))
            os.replace(last.with_suffix(".tmp"), last)

        def set_mode():
            net.train()
            for m in frozen_mods:
                m.eval()

        set_mode()
        # the next AHEAD batches drawn in advance, their cached tokens read while the current ones train: a TotalSeg
        # tile's fine tokens are ~1600 small Lustre reads per level (~3 s against ~0.7 s of GPU, 21/10)
        def draw():
            b = data.sample(int(job["batch"]))
            if tcache is not None:
                tcache.next_batch(keep=AHEAD)
                for xi, ci in (zip(b[0], b[2]) if isinstance(b[0], list) else [(b[0], b[2])]):
                    tcache.prefetch(ci, tuple(xi.shape[2:]))
            return b
        # drawn on one thread while the GPU trains (24/10, fetal: tiles + heatmaps took 0.9 of 2.4 s per iteration in
        # series with the step); one worker: the same draws in the same order
        from concurrent.futures import ThreadPoolExecutor
        drawer = ThreadPoolExecutor(1)
        queue = [drawer.submit(draw) for _ in range(min(AHEAD, iters - start + 1))]
        tm = {"data": 0.0, "step": 0.0, "val": 0.0}   # seconds per part since the last val line (24/10: where the time goes)
        # SFC_SEG_PROFILE="w,n" (27/10, where the step's time goes): after w iterations, n under cProfile + torch.profiler,
        # the tables to <out>.prof.txt, then the run stops (no val, no test)
        pw, pn = (int(v) for v in os.environ.get("SFC_SEG_PROFILE", "0,0").split(","))
        for it in range(start, iters + 1):
            if pn and it in (start + pw, start + pw + pn):
                torch.cuda.synchronize()
                if it == start + pw:
                    import cProfile
                    prof = (cProfile.Profile(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                                  torch.profiler.ProfilerActivity.CUDA]), time.time())
                    prof[1].__enter__()
                    prof[0].enable()
                else:
                    import io
                    import pstats
                    prof[0].disable()
                    prof[1].__exit__(None, None, None)
                    wall = (time.time() - prof[2]) / pn
                    ev = prof[1].key_averages()
                    gpu = sum(e.self_device_time_total for e in ev) / 1e6 / pn
                    buf = io.StringIO()
                    pstats.Stats(prof[0], stream=buf).sort_stats("cumulative").print_stats(80)
                    head = f"batch {job['batch']}: {wall:.3f} s per iteration, GPU kernels {gpu:.3f} s ({100 * gpu / wall:.0f} %)"
                    out_p.with_suffix(".prof.txt").write_text(head + "\n\n" + ev.table(sort_by="self_device_time_total", row_limit=40)
                                                              + "\n\n" + buf.getvalue())
                    print(f"[seg] profile: {head} -> {out_p.with_suffix('.prof.txt')}", flush=True)
                    return
            ta = time.time()
            x, y, c = queue.pop(0).result()
            if it + len(queue) < iters:
                queue.append(drawer.submit(draw))
            tb = time.time()
            opt.zero_grad(set_to_none=True)
            if LM:
                loss = landmark_step(net, x, y, c, amp, dev)
                logits = torch.zeros(1, device=dev)
            elif getattr(net, "cascade", False):
                loss = cascade_step(net, x, y, c, k, amp, dev)
                logits = torch.zeros(1, device=dev)
            elif isinstance(x, list) or (getattr(w, "seg_microbatch", False) and k > 10):   # lists: tiles of several shapes
                loss = micro_step(net, x, y, c, k, amp, dev)
                logits = torch.zeros(1, device=dev)
            else:
                with amp():
                    logits = call(net, x.to(dev, non_blocking=True).float(), c)
                loss = loss_fn(logits, y.to(dev), k)
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 12)
            if not torch.isfinite(gn):   # a cause to fix, never a step to skip (user, 18/10): stop and say which batch
                raise FloatingPointError(f"iter {it}: loss {float(loss)}, grad norm {float(gn)}, logits finite "
                                         f"{bool(torch.isfinite(logits).all())}, input finite "
                                         f"{all(bool(torch.isfinite(t).all()) for t in (x if isinstance(x, list) else [x]))}, "
                                         f"context {None if c is None else [t.tolist() for t in c]}")
            opt.step()
            sched.step()
            tc = time.time()
            tm["data"], tm["step"] = tm["data"] + tb - ta, tm["step"] + tc - tb
            if it % every == 0 or it == iters:
                net.eval()
                if getattr(net, "cascade", False) and not LM:   # one pass: the levels' thresholds on val and the val predictions
                    preds = net.cascade_val(val_v, k, amp)
                    if getattr(net, "crank", False):   # cascade_rank: the same val pass with the ranker's order (logged only)
                        net.cuse = "rank"
                        try:
                            rds = [dice_on(p, v["y"].to(dev), k) for p, v in zip(net.cascade_val(val_v, k, amp), val_v)]
                        finally:
                            net.cuse = "mixed"
                else:   # probabilities kept for the landmark tasks (a point per landmark from each class's map)
                    preds = [sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]).to(dev)
                             for v in val_v]
                    if job.get("select") != "mre":
                        preds = [p.argmax(0) for p in preds]
                ds = [dice_on(p.argmax(0) if p.dim() == 4 else p, v["y"].to(dev), k) for p, v in zip(preds, val_v)]
                val = float(np.nanmean(ds))
                rec = {"iter": it, "loss": float(loss), "val_dice": val, "min": (time.time() - t0) / 60}
                score, msg = val, f"val dice {val:.4f}"
                if getattr(net, "crank", False) and getattr(net, "cascade", False) and not LM:
                    rec |= {"val_dice_vol": [float(d) for d in ds], "rank_dice_vol": [float(d) for d in rds],
                            "rank_dice": float(np.nanmean(rds))}
                    msg += f" rank dice {rec['rank_dice']:.4f}"
                if job.get("select") == "mre":   # landmarks (user, 22/10): chosen on the val radial error, not the Dice
                    mre = float(np.mean([mre_on(p, v["y"].to(dev), v["sp"] if "sp" in v else v["ctx"][:3], k) for p, v in zip(preds, val_v)]))
                    rec["val_mre"], score, msg = mre, -mre, msg + f" val MRE {mre:.2f} mm"
                if job.get("select") in ("f1", "det"):   # small lesions (CMB, user 23/10): chosen on the val lesion-wise F1
                    md = "det" if job["select"] == "det" else True   # det: detection F1 (centroids within DET_MM)
                    f1v = float(np.nanmean([_metric([_score((p == c).cpu().numpy(), (v["y"] == c).numpy(), md,
                                                            sp=v.get("sp"), pts=v.get("pts")) for p, v in zip(preds, val_v)],
                                                    True)
                                            for c in range(1, k)]))
                    rec["val_f1"], score, msg = f1v, f1v, msg + f" val F1 {f1v:.3f}"
                del preds
                curve.append(rec)
                tm["val"] = time.time() - tc
                print(f"[seg] it {it} loss {float(loss):.3f} {msg} ({(time.time() - t0) / 60:.1f} min; s: data {tm['data']:.0f}"
                      f" step {tm['step']:.0f} val {tm['val']:.0f})", flush=True)
                tm = dict.fromkeys(tm, 0.0)
                if "state" not in best or score > best.get("score", best["val"]):
                    best = {"val": val, "iter": it, "score": score, "state": trained()} | \
                        ({"val_mre": rec["val_mre"]} if "val_mre" in rec else {}) | \
                        ({"val_f1": rec["val_f1"]} if "val_f1" in rec else {})
                if hasattr(net, "on_val"):   # fm/dense_dec cubes: the next fine level placed on the training scans
                    net.on_val(it, every, [(xv, cv) for (xv, _, _), cv in zip(data.vols, data.ctx)],
                               lambda xv, cv: sliding(net, xv.to(dev).float(), patch, k, amp, cv))
                set_mode()
                pat = int(job.get("patience") or 0)   # screening: once learning (> 0), no new best for `pat` validations,
                # never before 60 % of the budget (CMB p20 d4 Dmix, 24/10: F1 .018 / .024 then 0, learns after 7500)
                if pat and it >= 0.6 * iters and (job.get("select") == "mre" or best["score"] > 0) and \
                        sum(r["iter"] > best["iter"] for r in curve) >= pat:
                    print(f"[seg] no new val best since it {best['iter']}: stopped (screening)", flush=True)
                    break
            if time.time() - t_save > SAVE_S and it < iters:
                save_last(it)
                t_save = time.time()
        del val_v, data
        torch.save({"key": key, "best": best, "curve": curve}, ckpt)
        last.unlink(missing_ok=True)
    if LM and getattr(net, "ctopk_eval", None) and "lm_topk" not in best:
        # landmark cascade (user, 24/10): the cells per landmark scaled on the val MRE of the best state, once (x1 is the
        # selection's own pass; x0.5 and x2 two more), kept for the test
        load_trained(best["state"])
        net.eval()
        val_v, t0k = [load(it) for it in split["val"]], list(net.ctopk_eval)
        by = {1.0: best["val_mre"]}
        for f in (0.5, 2.0):
            net.ctopk_eval = [max(1, round(t * f)) for t in t0k]
            by[f] = float(np.mean([mre_on(sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]).to(dev),
                                          v["y"].to(dev), v["sp"], k) for v in val_v]))
        f = min(by, key=lambda t: (by[t], t))
        best |= {"lm_topk": [max(1, round(t * f)) for t in t0k], "val_mre": by[f],
                 "mre_by_topk": {"x" + str(t): m for t, m in sorted(by.items())}}
        net.ctopk_eval = best["lm_topk"]
        print(f"[seg] landmark cells per level {best['lm_topk']}: val MRE {by[f]:.2f} mm "
              f"({' '.join(f'x{t}:{m:.2f}' for t, m in sorted(by.items()))})", flush=True)
        del val_v
        torch.save({"key": key, "best": best, "curve": curve}, ckpt)
    if job.get("val_only"):  # pretraining tuning: the best val Dice, no test volume read
        if job.get("save_pred"):   # the best state's val predictions kept anyway (user, 24/10: lesion counts on val)
            load_trained(best["state"])
            net.eval()
            if "lm_topk" in best:
                net.ctopk_eval = best["lm_topk"]
            pv = out_p.with_suffix(".pred_val")
            for it in split["val"]:
                if not (pv / f"{it['id']}.npz").exists():
                    v = load(it)
                    save_pred(pv, it["id"], sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]),
                              v["grid_from_gt"])
                    del v
        best.pop("state")
        out_p.write_text(json.dumps({"model": job["cfg"]["model"], "init": job["init"], "iters": iters,
                                     "best": best, "curve": curve, "val_dice": best["val"],
                                     "n": {s: len(v) for s, v in split.items() if s != "test"}}, indent=1))
        provenance(job, split, out_p, ckpt, {"best": best})
        print(f"[seg] best val dice {best['val']:.4f}" + (f" val MRE {best['val_mre']:.2f} mm" if 'val_mre' in best else "") +
              f" (iter {best['iter']}) -> {job['out']}", flush=True)
        return
    load_trained(best.pop("state"))
    net.eval()
    if os.environ.get("SFC_CASCADE_ORACLE") and getattr(net, "cascade", False) and not LM:
        # diagnosis (user 27/10): the val Dice / lesion F1 with the cascade as trained and with an oracle gate (every truly
        # mixed cell read down, the thresholds as calibrated) -> how much the gate loses; nothing written but the log
        val_v = [load(it) for it in split["val"]]
        for net.oracle in (False, True):
            preds = net.cascade_val(val_v, k, amp)
            dv = float(np.nanmean([dice_on(p, v["y"].to(dev), k) for p, v in zip(preds, val_v)]))
            pc = [_metric([_score((p == c).cpu().numpy(), (v["y"] == c).numpy(), False) for p, v in zip(preds, val_v)], False)
                  for c in range(1, k)]
            f1 = [_metric([_score((p == c).cpu().numpy(), (v["y"] == c).numpy(), True) for p, v in zip(preds, val_v)], True)
                  for c in range(1, k)]
            print(f"[seg] oracle gate {net.oracle}: val Dice {dv:.4f}, pooled per class {np.round(pc, 4).tolist()}, "
                  f"lesion F1 per class {np.round(f1, 4).tolist()}, refined {net.last_refined:.4f}", flush=True)
        return
    if "lm_topk" in best:   # the val-chosen cells per landmark for the test and the saved val predictions
        net.ctopk_eval = best["lm_topk"]
    rule, f1 = {}, "det" if DET else job.get("select") == "f1"
    pv = out_p.with_suffix(".pred_val")
    if not LM:   # val predictions of the best state -> the post-processing rule (chosen on val only)
        vals = []
        for it in split["val"]:
            f = pv / f"{it['id']}.npz"
            if not f.exists():
                v = load(it)
                save_pred(pv, it["id"], sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]),
                          v["grid_from_gt"])
                del v
            z = np.load(f)
            y = np.asarray(torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)["y"])
            y = y[0] if y.ndim == 4 else y
            vals.append((z["pred"], z["prob"] if len(z["prob"]) == k else None, np.where(y == 255, 0, y),
                         grid_spacing(z["grid_from_gt"], it["seg"]) if DET else None,
                         true_points(PTS, it["id"], z["grid_from_gt"]) if DET else None))
        sizes = {}   # train components' sizes (voxels on the model grid): 1st / 5th percentile per class
        for it in split["train"][:20]:
            y = np.asarray(torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)["y"])
            y = y[0] if y.ndim == 4 else y
            for c in range(1, k):
                cc, n = _components(y == c)
                if n:
                    sizes.setdefault(c, []).extend(np.bincount(cc.ravel())[1:].tolist())
        sizes = {c: [int(np.percentile(v, 1)), int(np.percentile(v, 5))] for c, v in sizes.items()}
        rule = choose_postproc(vals, k, sizes, f1)
        del vals
    tkey = key | {"best_iter": best["iter"], "postproc": {str(c): r for c, r in rule.items()}} | \
        ({"lm_topk": best["lm_topk"]} if "lm_topk" in best else {})   # (only when chosen: other runs' partial tests resume)
    done = json.loads(part.read_text()) if part.exists() else {}
    res = done["res"] if done.get("key") == tkey else {"per_volume": [], "per_class": [], "test_ids": []}
    if res["test_ids"]:
        print(f"[seg] test resumed from {part}: {len(res['test_ids'])} volumes done", flush=True)
    for it in split["test"]:
        if it["id"] in res["test_ids"]:
            continue
        t1 = time.time()
        v = load(it)
        probs = sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"])
        lab, _, _ = gt_labels(it["seg"], it.get("classes"), it.get("ignore"))
        pl = None
        if rule:   # the val-chosen post-processing, then scored as the argmax of its one-hot
            q = (probs.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy() if k <= 8 else None
            pl = postproc(probs.argmax(0).to(torch.uint8).cpu().numpy(), q, rule)
            probs_s = F.one_hot(torch.from_numpy(pl).long(), k).permute(3, 0, 1, 2).half()
        d, per = dice_full_res(probs_s if rule else probs, v["grid_from_gt"], lab, k, dev)
        if DET:   # detection sums (tp, points, false detections) per class on the model grid, as the val selection
            pm = pl if pl is not None else probs.argmax(0).cpu().numpy()
            yy = (v["y"][0] if v["y"].dim() == 4 else v["y"]).numpy()
            res.setdefault("det", []).append([_score(pm == c, yy == c, "det", sp=v["sp"], pts=v["pts"]) for c in range(1, k)])
        save_pred(out_p.with_suffix(".pred"), it["id"], probs, v["grid_from_gt"], lab=pl)
        res["per_volume"].append(d)
        res["per_class"].append(per)
        res["test_ids"].append(it["id"])
        part.write_text(json.dumps({"key": tkey, "res": res}))
        print(f"[seg] test {len(res['test_ids'])}/{len(split['test'])} {it['id']}: dice {d:.4f} grid "
              f"{tuple(v['x'].shape[1:])} {time.time() - t1:.0f}s, peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} GB", flush=True)
        del v, probs, lab
    out = {"model": job["cfg"]["model"], "init": job["init"], "lr": job["lr"], "iters": iters, "patch": patch,
           "frozen_prefixes": frozen, "params_frozen": n_frozen, "params_trained": n_train, "best": best,
           "curve": curve, "n": {s: len(v) for s, v in split.items()},
           "test_dice": float(np.nanmean(res["per_volume"])), "postproc": {str(c): r for c, r in rule.items()}, **res}
    if DET:
        out["test_det_f1"] = float(np.nanmean([_metric([d[c] for d in res["det"]], True) for c in range(k - 1)]))
    out_p.write_text(json.dumps(out, indent=1))
    for it in split["val"]:   # the val predictions of the best state (landmarks: not done above)
        if (pv / f"{it['id']}.npz").exists():
            continue
        v = load(it)
        save_pred(pv, it["id"], sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]), v["grid_from_gt"])
        del v
    provenance(job, split, out_p, ckpt, {"best": best, "test_dice": out["test_dice"], "postproc": out["postproc"]})
    part.unlink(missing_ok=True)   # the decoder checkpoint (.best.pt) is kept (user, 23/10)
    print(f"[seg] test dice {out['test_dice']:.4f} on {len(res['per_volume'])} volumes -> {job['out']}", flush=True)


TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)   # post-processing: probability thresholds tried per class (k <= 8)


def _components(m: np.ndarray):
    from scipy import ndimage
    return ndimage.label(m, np.ones((3, 3, 3)))


def _pp_class(lab: np.ndarray, q: np.ndarray | None, c: int, tau, smin: int, largest: bool, comp=None) -> np.ndarray:
    """Class c's mask after its rule: threshold tau on its probability (uint8 q) instead of the argmax, components under
    smin voxels removed, only the largest kept. comp: the mask's components if already computed."""
    m = (q[c] >= round(tau * 255)) if tau is not None else lab == c
    if smin or largest:
        cc, n = comp if comp is not None else _components(m)
        if n:
            sz = np.bincount(cc.ravel())[1:]
            keep = (sz >= smin) if not largest else (np.arange(n) == sz.argmax())
            m = keep[cc - 1] & (cc > 0)
    return m


def postproc(lab: np.ndarray, q: np.ndarray | None, rule: dict) -> np.ndarray:
    """Labels [*S] (argmax) -> labels with each class's val-chosen rule applied (classes in order)."""
    out = lab.copy()
    for c, (tau, smin, largest) in sorted(rule.items()):
        m = _pp_class(lab, q, c, tau, smin, largest)
        out[(out == c) & ~m] = 0
        out[m] = c
    return out


DET_MM = 5.0   # detection (select: det): a predicted centroid within this distance of a true point is a hit


def grid_spacing(grid_from_gt, seg) -> np.ndarray:
    """The model grid's spacing (mm): the GT's zooms through the grid mapping."""
    A = np.linalg.inv(np.asarray(grid_from_gt, dtype=float)[:3, :3])
    return np.linalg.norm(A * np.asarray(nib.load(seg).header.get_zooms()[:3], dtype=float)[:, None], axis=0)


def true_points(pts, i: str, grid_from_gt) -> np.ndarray | None:
    """The task's true points of volume i (GT voxels, points csv grouped by id) on the model grid [n, 3]; None without
    a points csv (the label's components then)."""
    if pts is None:
        return None
    if i not in pts.groups:
        return np.zeros((0, 3))
    M = np.asarray(grid_from_gt, dtype=float)
    return pts.get_group(i).to_numpy(float) @ M[:3, :3].T + M[:3, 3]


def _score(m: np.ndarray, t: np.ndarray, f1, gt=None, comp=None, sp=None, pts=None) -> tuple:
    """A class on one volume -> sums: Dice (tp, pred, true), lesion F1 (hit lesions, lesions, false components) or,
    f1 == "det", detection (true points matched, points, unmatched predictions: the centroids of the components of
    m and t, one to one, greedy by distance, within DET_MM on the grid of spacing sp); gt / comp: the true / predicted
    mask's components if already computed; pts: the true points [n, 3] on the grid instead of t's components."""
    if f1 == "det":
        from scipy import ndimage
        if pts is None:
            gl, ng = gt if gt is not None else _components(t)
        else:
            ng = len(pts)
        pl, npred = comp if comp is not None else _components(m)
        if not ng or not npred:
            return 0.0, float(ng), float(npred)
        a = (np.asarray(ndimage.center_of_mass(t, gl, range(1, ng + 1))) if pts is None else pts) * sp
        b = np.asarray(ndimage.center_of_mass(m, pl, range(1, npred + 1))) * sp
        d = np.linalg.norm(a[:, None] - b[None], axis=-1)
        ui, uj, tp = set(), set(), 0
        for i, j in zip(*np.unravel_index(np.argsort(d, None), d.shape)):
            if d[i, j] > DET_MM:
                break
            if i not in ui and j not in uj:
                ui.add(i), uj.add(j)
                tp += 1
        return float(tp), float(ng), float(npred - tp)
    if not f1:
        return float((m & t).sum()), float(m.sum()), float(t.sum())
    gl, ng = gt if gt is not None else _components(t)
    pl, npred = _components(m)
    hit = len(set(np.unique(gl[m])) - {0})
    return float(hit), float(ng), float(npred - len(set(np.unique(pl[t])) - {0}))


def _metric(sums, f1: bool) -> float:
    a, b, c = (sum(x[i] for x in sums) for i in range(3))
    if not f1:
        return 2 * a / (b + c) if b + c else float("nan")
    p, r = a / max(a + c, 1), a / max(b, 1)
    return 2 * p * r / max(p + r, 1e-9)


def choose_postproc(vals: list, k: int, sizes: dict, f1: bool) -> dict:
    """Per foreground class, on the val volumes [(argmax, uint8 probs or None, labels)], the rule (tau, min size,
    largest only) that improves the task's metric (Dice, or lesion F1); no rule if nothing does (user, 23/10:
    nnU-Net-style post-processing chosen on val, the same for every task)."""
    rule = {}
    for c in range(1, k):
        t = [(y == c) for _, _, y, *_ in vals]
        if not any(x.any() for x in t):
            continue
        gts = [_components(tt) if f1 else None for tt in t]
        base = _metric([_score(lab == c, tt, f1, g, sp=sp, pts=pp) for (lab, _, _, sp, pp), tt, g in zip(vals, t, gts)], f1)
        best, br = base, None
        taus = (None,) + (TAUS if vals[0][1] is not None else ())
        for tau in taus:   # the thresholded masks' components once per tau, the filters on top of them
            comps = [_components((q[c] >= round(tau * 255)) if tau is not None else lab == c) for lab, q, _ in vals]
            for smin in sorted({0, *sizes.get(c, ())}) if f1 != "det" else (0,):   # det: the threshold only
                for largest in (False, True) if f1 != "det" else (False,):
                    if tau is None and not smin and not largest:
                        continue
                    sc = _metric([_score(_pp_class(lab, q, c, tau, smin, largest, cp), tt, f1, g,
                                         cp if f1 == "det" else None, sp, pp)
                                  for (lab, q, _, sp, pp), tt, g, cp in zip(vals, t, gts, comps)], f1)
                    if sc > best + 1e-4:
                        best, br = sc, (tau, int(smin), largest)
        if br is not None:
            rule[c] = br
            print(f"[seg] post-processing class {c}: tau {br[0]}, min {br[1]} voxels, largest {br[2]}: val "
                  f"{base:.4f} -> {best:.4f}", flush=True)
    return rule


def save_pred(d: Path, i: str, probs: torch.Tensor, grid_from_gt, lab=None) -> None:
    """Prediction of one volume on the model grid (user, 23/10: always kept, val and test, to choose thresholds and
    filters after the run): argmax, class probabilities as uint8 (p * 255; all classes up to 8, else the argmax's
    probability only) and the GT mapping (LFC landmark error, CMB lesion F1)."""
    d.mkdir(parents=True, exist_ok=True)
    q = (probs.float().clamp(0, 1) * 255).round().to(torch.uint8)
    np.savez_compressed(d / f"{i}.npz", pred=(probs.argmax(0).to(torch.uint8).cpu().numpy() if lab is None else lab),
                        prob=(q if len(q) <= 8 else q.max(0).values[None]).cpu().numpy(),
                        grid_from_gt=np.asarray(grid_from_gt))


def provenance(job: dict, split: dict, out_p: Path, ckpt: Path, extra: dict) -> None:
    """<out>.provenance.json: where the run's files come from (code commit, encoder checkpoint, job, split ids)."""
    repo = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                                timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "src"], capture_output=True, check=False,
                                    text=True, timeout=30).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "?", None
    out_p.with_suffix(".provenance.json").write_text(json.dumps({
        "result": str(out_p), "decoder_checkpoint": str(ckpt), "pred_test": str(out_p.with_suffix(".pred")),
        "pred_val": str(out_p.with_suffix(".pred_val")), "commit": commit, "uncommitted_src_changes": dirty,
        "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "encoder_checkpoint": job["cfg"].get("checkpoint"),
        "job": {k: v for k, v in job.items() if k != "items"},
        "split_ids": {s: [it["id"] for it in v] for s, v in split.items()}} | extra, indent=1, default=str))


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    {"prep": prep, "train": train}[job["stage"]](job)


if __name__ == "__main__":
    main(sys.argv[1])
