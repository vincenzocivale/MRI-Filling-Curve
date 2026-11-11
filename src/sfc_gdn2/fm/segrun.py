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
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from ..bench.geom import gt_labels
from .worker import build, fail, prepared

# token-cache runs (20/10): tiles of TILE_CELLS 16 mm cells per axis (16 cm, as the resampled models' 160 voxels at
# ~1 mm) instead of 160 native voxels (67 cm of TotalSeg's 4.2 mm voxels, 7 cm of SOOP's 0.45 mm)
TILE_CELLS = 10
LES_MM = 0.0   # tile_rule: half the p95 foreground extent; a lesion-centred tile holds it whole (Patches.sample)
AHEAD = 6   # batches whose cached tokens are read in advance
TILE_MAX = 1e9   # voxels x classes of one tile
SLIDE = [int(os.environ.get("SFC_SLIDE_BATCH", "4"))]   # sliding: tiles per forward (halved on OOM, kept halved)
# (05/11, Francesco: SOOP tests took ~6 h) tiles whose input is all zero (air around skull-stripped scans) are not run;
# voxels no run tile covers are background. nnU-Net's crop to the nonzero box, with the tiles where they were (the
# token caches' corners unchanged). SFC_SKIP_EMPTY=0: every tile, as before.
SKIP_EMPTY = os.environ.get("SFC_SKIP_EMPTY", "1") == "1"
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


def tile_rule(job: dict, vols: list, lm: bool, k: int = 2) -> None:
    """The tile and the batch from the task's fingerprint (user 31/10: "la dimensione della patch conta molto", a rule):
    on up to 40 training scans the foreground's components (26-connected), their largest extent in mm (p95, E); the
    tile 2 E per axis in 16 mm cells (TILE_CELLS), at least 48 voxels on the finest axis, at most TILE_CELLS; the batch
    holds the voxels of 4 tiles of 160^3 (8: out of memory with the many tiles' fields), between 8 and 64 tiles (landmarks: 8, their step runs one tile at a
    time). The tokens give the global context; the tile only the local one."""
    from scipy import ndimage
    global TILE_CELLS, LES_MM
    ext, al, tv = [], [], []
    for v in vols:
        y = v["y"].reshape(v["x"].shape[1:]).numpy()
        sp = np.asarray([float(a) for a in v["ctx"][:3]])
        lab, _ = ndimage.label((y > 0) & (y < 255), np.ones((3, 3, 3)))
        ext += [float((np.array([q.stop - q.start for q in sl]) * sp).max()) for sl in ndimage.find_objects(lab)]
        al.append(v["align"])
    E = float(np.percentile(ext, 95)) if ext else 160.0
    n = min(TILE_CELLS, max(math.ceil(2 * E / 16), math.ceil(int(os.environ.get("SFC_TILE_MIN", "48")) / float(np.median([max(a) for a in al])))))   # v14: 48 -> 96 voxels
    for v in vols:
        tv.append(math.prod(min(n * a, s) for a, s in zip(v["align"], v["x"].shape[1:])))
    # the batch from the largest tile, not the median (31/10: HIE, cmbpts, bladder out of memory when a batch drew the
    # scans whose tiles are largest); 4 x 160^3 voxels (8 x: ms_seg out of memory)
    # classes scale the logits' memory (31/10: TotalSeg, ~50 classes, out of memory at 8 tiles): the budget for 14
    # classes (Willis, 8 x 160^3 fitted), fewer tiles above, at least 2
    b = 8 if lm else int(min(64, max(2 if k > 14 else 8, 4 * 160 ** 3 * min(1.0, 14 / k) // min(max(tv), 160 ** 3))))
    b = int(os.environ.get("SFC_BATCH", b))   # user 31/10: the baselines' schedule (batch 2) with the rule's tile
    print(f"[seg] tile rule: foreground extent p95 {E:.1f} mm -> {n} cells of 16 mm per axis (median tile "
          f"{float(np.median(tv)) ** (1 / 3):.0f}^3 voxels), batch {b} (was {TILE_CELLS} cells, {job['batch']})", flush=True)
    TILE_CELLS, job["batch"], LES_MM = n, b, E / 2


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
    """[C, *S] -> class probabilities [n_out, *S] (Gaussian-weighted tiles, step 0.5 of the patch)."""
    lm = getattr(net, "landmark", False)   # landmark heatmaps: sigmoid per channel, the generic tile path
    S = tuple(x.shape[1:])
    patch = S if patch is None else tuple(min(p, s) for p, s in zip(patch, S)) if getattr(net, "any_tile", False) \
        else tuple(patch)
    snap = getattr(net, "snap", None)   # fm/hier_dec AlignUNet: tiles, corners and the padded volume on its bottleneck grid
    if getattr(net, "fixed_patch", None):   # nnU-Net's patch (cut to the scan, on the grid)
        patch = tuple(min(p, max(-(-s // b) * b, 2 * b)) for p, s, b in zip(net.fixed_patch, S, snap))
    elif snap:
        patch = tuple(max(b, round(p / b) * b) for p, b in zip(patch, snap))
    xp, _ = pad_to(x, None, [-(-max(s, p) // b) * b for s, p, b in zip(S, patch, snap)] if snap else patch)
    P = tuple(xp.shape[1:])
    starts = [np.unique(np.round(np.linspace(0, s - p, math.ceil((s - p) / (p / 2)) + 1)).astype(int)) if s > p else [0]
              for s, p in zip(P, patch)]
    if snap:   # the last start s - p is on the grid (both multiples): coverage kept
        starts = [np.unique(np.asarray(st) // b * b) for st, b in zip(starts, snap)]   # (an axis shorter than the tile: [0])
    g = gaussian(patch, x.device)
    # fp16 accumulator on the device if it takes <= 1/4 of the free memory, else on the CPU (nnU-Net's predictor does
    # both; "fits at all" left the tiles' forward no room: nnF CNN OOM on whole-body TotalSeg grids)
    # (07/11) free = the device's + what PyTorch's allocator holds unused (reserved, not allocated): a larger tile batch
    # left more of it reserved and the accumulator went to the CPU (TotalSeg val ~2-3x slower at SFC_SLIDE_BATCH 8-16)
    free = torch.cuda.mem_get_info(x.device)[0] + torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(x.device) \
        if x.is_cuda else 0
    fits = x.is_cuda and n_out * math.prod(P) * 2 <= free / 4
    out = torch.zeros(n_out, *P, dtype=torch.half, device=x.device if fits else "cpu")
    wsum = torch.zeros(P, device=x.device)
    tiles, i, bs = list(product(*starts)), 0, SLIDE[0]   # tiles in batches (28/10), halved on OOM
    if SKIP_EMPTY:
        nz = (xp != 0).any(0)
        keep = [t for t in tiles if bool(nz[t[0]:t[0] + patch[0], t[1]:t[1] + patch[1], t[2]:t[2] + patch[2]].any())]
        tiles = keep or tiles   # (an all-zero volume: every tile, as before)
        del nz
    corner = ctx is not None and getattr(net, "tile_corner", False)
    while i < len(tiles):
        part = tiles[i: i + bs]
        sls = [(slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2])) for a, b, c in part]
        try:
            with amp():
                cx = None if ctx is None else torch.stack(
                    [torch.cat([ctx.float(), torch.tensor(t, dtype=torch.float)]) if corner else ctx for t in part])
                lg = call(net, torch.stack([xp[(slice(None), *sl)] for sl in sls]), cx)   # fm/hier_dec: tokens by corner
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs //= 2
            SLIDE[0] = bs   # (07/11) kept for the next volumes: each started again at 16 and failed twice (TotalSeg
            # val ~25 s per scan against ~6 at 4)
            torch.cuda.empty_cache()
            continue
        for sl, lt in zip(sls, lg):
            out[(slice(None), *sl)] += ((lt.float().sigmoid() if lm else lt.float().softmax(0)) * g).to(out)
            wsum[sl] += g
        i += len(part)
    wsum = wsum.to(out.device)
    for c in range(n_out):  # one class at a time: dividing the whole fp16 accumulator by an fp32 tensor made an fp32
        out[c].div_(wsum)   # copy of it (whole-body nnF grids: 60 GB accumulator, RAM OOM at 240 GB)
    if SKIP_EMPTY and bool((wsum == 0).any()):   # no tile covered them: background (heatmaps: 0)
        empty = wsum == 0
        for c in range(n_out):
            out[c][empty] = 1.0 if c == 0 and not lm else 0.0
    return out[(slice(None),) + tuple(slice(0, s) for s in S)]


def dice_on(pred: torch.Tensor, y: torch.Tensor, k: int) -> float:
    ok = y != 255
    p, t = pred[ok], y[ok].long()
    ds = [2 * ((p == c) & (t == c)).sum().item() / ((p == c).sum() + (t == c).sum()).item()
          for c in range(1, k) if (t == c).any() or (p == c).any()]
    return float(np.mean(ds)) if ds else float("nan")


MISS_MM = 50.0   # a landmark not predicted at all counts as this radial error in the val selection


TRAIN_EVAL = 6   # training scans scored at each validation (user 30/10)
TRAIN_LOG = int(os.environ.get("SFC_TRAIN_LOG", "125"))  # iterations between the running training lines (loss, the batches' soft Dice; user 31/10)
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


def sel_target(net, y) -> None:
    """fm/hier_dec: the labels of the tiles the next forward trains its P(mixed) selector on."""
    if hasattr(net, "sel_y"):
        net.sel_y = y


def pop_aux(net):
    """The auxiliary loss the last forward left (hier_dec's selector, deep supervision), 0 otherwise."""
    a = getattr(net, "aux", 0.0)
    if hasattr(net, "aux"):
        net.aux = 0.0
    return a


def landmark_step(net, x, y, c, amp, dev) -> torch.Tensor:
    """Landmark tasks: channels 1.. of the decoder are heatmaps (sigmoid), y the tiles' target heatmaps; one scan
    at a time, every model's decoder as it is."""
    parts = list(zip(x, y, c if c is not None else [None] * len(x))) if isinstance(x, list) else \
        [(x[i:i + 1], y[i:i + 1], None if c is None else c[i:i + 1]) for i in range(len(x))]
    parts = [(a, heatmaps(b.to(dev), a.shape[2:]), u) for a, b, u in parts]   # the maps on the GPU (heat_spec)
    tot = 0.0
    for xi, yi, ci in parts:   # one scan at a time
        sel_target(net, (yi.amax(1) > 0.1).to(torch.uint8))
        with amp():
            lg = call(net, xi.to(dev).float(), ci)
        loss = (focal_heatmap(lg[:, 1:].float(), yi.to(dev).float()) + pop_aux(net)) / len(parts)
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
    pred: class probabilities [k, *S] -> landmark() (every landmark gets a point)."""
    sp = torch.as_tensor(spacing, dtype=torch.float, device=pred.device)
    err = []
    for c in range(1, k):
        g = (y == c).nonzero().float()
        if not len(g):
            continue
        err.append(float(((g.mean(0) - landmark(pred[c])) * sp).norm()))
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
    """Random training patches from volumes held in RAM; 1/3 centred on a voxel of a random present class.
    SFC_FT_SCANS=n (fine-tuning, 04/11): each batch's tiles from n scans drawn for the batch, so the encoder reads n
    whole scans per step with gradient (WMH: the planner's 40 small tiles came from ~40 scans, out of memory)."""

    def __init__(self, vols: list[dict], patch, k: int, rng: np.random.Generator, fit: bool = False, snap=None, fixed=None):
        self.fit, self.snap = fit, snap   # snap (fm/hier_dec AlignUNet): tiles and corners on the bottleneck's grid
        # fixed (AlignUNet SFC_NNUNET_PLAN): nnU-Net's label-free patch for every scan, cut to the scan (its grid)
        self.vols, self.ctx, self.patch, self.rng, self.align, self.lm = [], [], [], rng, [], []
        for v in vols:
            self.lm.append(v.get("lm"))
            pv = tile_of(patch, v.get("align"), k)
            if fixed:
                pv = [min(p, max(-(-z // b) * b, 2 * b)) for p, z, b in zip(fixed, v["x"].shape[1:], snap)]   # >= 2 bottleneck
                # cells per axis (a scan smaller than one: zero-padded; 1 x 1 x 1 broke InstanceNorm in training)
            elif pv and getattr(self, "fit", False) and v.get("align"):   # any_tile: never larger than the volume (its grid)
                pv = [min(p, -(-s // a) * a) for p, s, a in zip(pv, v["x"].shape[1:], v["align"])]
            if pv and snap and not fixed:   # the nearest multiple (the voxel budget's tile kept: Willis 156x156x84 -> 160x160x80)
                pv = [max(b, round(p / b) * b) for p, b in zip(pv, snap)]
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
        m = int(os.environ.get("SFC_FT_SCANS", "0"))
        pool = self.rng.choice(len(self.vols), min(m, len(self.vols)), replace=False) if m else None
        for _ in range(n):
            j = self.rng.integers(len(self.vols)) if pool is None else int(pool[self.rng.integers(len(pool))])
            x, y, fg = self.vols[j]
            cs.append(self.ctx[j])
            S = np.array(y.shape)
            P = S if self.patch[j] is None else np.array(self.patch[j])
            if fg and self.rng.random() < 1 / 3:
                vox = fg[list(fg)[self.rng.integers(len(fg))]]
                c = vox[self.rng.integers(len(vox))].numpy()
                start = np.clip(c - P // 2, 0, S - P)
                if self.align[j] is not None:   # a grid start holding the lesion whole (LES_MM around the voxel), anywhere
                    a = np.array(self.align[j])  # in the tile (v13's middle half taught fetal landmarks a centre bias)
                    m = np.ceil(LES_MM * a / 16).astype(int)   # 16 mm = one cell of `a` voxels
                    if self.lm[j] is None and os.environ.get("SFC_SAMPLE_MID", "1") == "1":   # segmentation: the lesion
                        m = np.maximum(m, P // 4)   # in the tile's middle half (v13); landmarks anywhere (user 31/10)
                    lo, hi = np.maximum(0, c - P + 1 + m), np.minimum(S - P, c - m)
                    lo, hi = -(-lo // a), hi // a
                    start = np.where(hi >= lo, np.array([self.rng.integers(x, y + 1) if y >= x else 0
                                                         for x, y in zip(lo, hi)]) * a, start)
            else:
                start = np.array([self.rng.integers(s - p + 1) for s, p in zip(S, P)])
            if self.align[j] is not None:   # on the volume's 16 mm grid (token cache)
                a = np.array(self.snap or self.align[j])
                start = (start // a) * a
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
    inter = _binsum(yc.flatten(), p.gather(1, yc.unsqueeze(1)).flatten(), k)
    return ce, ok.sum(), inter, p.sum((0, 2, 3, 4)), _binsum(yc.flatten(), ok.flatten(), k)


def micro_step(net, x, y, c, k: int, amp, dev, m: int = 1) -> torch.Tensor:
    """The gradient of loss_fn on the whole batch, one scan at a time (volumes of several shapes: whole-volume
    models, patch None; or a wrapper's `seg_microbatch` with many classes). The soft Dice sums over the batch before dividing, so a first
    pass without gradient gives the batch's sums; then each scan backpropagates a loss linear in its own sums with
    the batch's derivatives as coefficients: the same gradient as the batch of 2, one more forward pass."""
    parts = list(zip(x, y, c if c is not None else [None] * len(x))) if isinstance(x, list) else \
        [(x[i:i + m], y[i:i + m], None if c is None else c[i:i + m]) for i in range(0, len(x), m)]
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
        sel_target(net, yi)
        with amp():
            logits = call(net, xi.to(dev).float(), ci)
        ce, _, i_, p_, _ = batch_sums(logits, yi.to(dev), k)
        (ce / n + (a * i_).sum() + (b * p_).sum() + pop_aux(net) / len(parts)).backward()
    dice = (2 * inter[1:] + 1e-5) / den[1:]
    return sum(s[0] for s in sums) / n + 1 - dice.mean()


def group_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """A decoder reading any tile shape (any_tile, fm/hier_dec): the batch's tiles of one shape in one call, one call per
    shape, loss_fn's value on the whole batch from the calls' summed sums (the soft Dice over the batch), one backward."""
    by = {}
    for i, t in enumerate(x):
        by.setdefault(tuple(t.shape), []).append(i)
    sums, aux = [], 0.0
    for idx in by.values():
        yi = torch.cat([y[i] for i in idx])
        sel_target(net, yi)
        with amp():
            lg = call(net, torch.cat([x[i] for i in idx]).to(dev).float(), None if c is None else torch.cat([c[i] for i in idx]))
        sums.append(batch_sums(lg, yi.to(dev), k))
        aux = aux + pop_aux(net) * len(idx) / len(x)
    tt = [sum(s_[j] for s_ in sums) for j in range(5)]
    group_step.sums = [t.detach() for t in tt[2:]]   # the batch's soft Dice sums, for the running training log
    loss = sums_loss(tt) + aux
    loss.backward()
    return loss.detach()


def sums_loss(t: list) -> torch.Tensor:
    """batch_sums' sums (summed over calls) -> CE per voxel + 1 - mean soft Dice of the classes > 0 (loss_fn's)."""
    ce, n, inter, psum, tsum = t
    return ce / n.clamp_min(1) + 1 - ((2 * inter[1:] + 1e-5) / (psum[1:] + tsum[1:] + 1e-5)).mean()


def _binsum(idx: torch.Tensor, w: torch.Tensor, k: int, chunks: int = 4096) -> torch.Tensor:
    """[k] sums of w per class idx (flat): scatter_add into chunks x k bins, then the chunks summed. (08/11) one
    scatter_add into k bins serialises millions of atomics on a few addresses: 13-15 % of an AlignUNet step on Willis
    (10.7 ms -> 0.8 ms per call at 51 classes; the same sums up to fp32 reassociation, rel. 3e-5)."""
    c = torch.arange(idx.numel(), device=idx.device) // -(-idx.numel() // chunks)
    return w.new_zeros(chunks * k).scatter_add(0, c * k + idx, w).view(chunks, k).sum(0)


def loss_fn(logits: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    y = y.long()
    ce = F.cross_entropy(logits.float(), y, ignore_index=255)
    ok = (y != 255).unsqueeze(1).float()
    p = logits.float().softmax(1) * ok
    # soft Dice per class without a one-hot target (the same sums: with ~50 classes on a 160^3 batch of 2 the int64
    # one-hot and its float copies took ~6 GB, GPU OOM for decoder F on TotalSeg, 18/10)
    yc = torch.where(y == 255, 0, y)
    inter = _binsum(yc.flatten(), p.gather(1, yc.unsqueeze(1)).flatten(), k)   # p is 0 off `ok`
    tsum = _binsum(yc.flatten(), ok.flatten(), k)
    dice = (2 * inter[1:] + 1e-5) / (p.sum((0, 2, 3, 4))[1:] + tsum[1:] + 1e-5)
    return ce + 1 - dice.mean()


def vox_budget(job: dict, vols: list, patch, k: int, lm: bool = False) -> None:
    """B' (user 31/10): every model the same budget of SFC_VOX_BUDGET voxels per step (2 x 160^3: a baseline's 160^3
    patch at batch 2), filled with its own tile: the batch = budget // the largest training tile, between 2 (nnU-Net's
    minimum) and 64. Ours (token grid): a tile of at most budget / 2 voxels (fewer 16 mm cells), above 14 classes fewer
    tiles (logits' memory: TotalSeg 2). Baselines: their patch, the whole volume when they have none (MedicalNet)."""
    global TILE_CELLS
    V = int(os.environ["SFC_VOX_BUDGET"])
    thr = float(os.environ.get("SFC_PLANNER_THR", "0"))
    if thr and vols and vols[0].get("align"):
        # planner (user 01/11): the largest tile (most context) whose steps still see thr x V lesion voxels on average.
        # From the training labels: L = size-biased mean component (the one a foreground-centred tile lands on), f = the
        # foreground share, E = p95 component extent. E[lesion voxels per step] = B/3 min(L, V/B) + 2/3 f V (1/3 of the
        # tiles foreground-centred, segrun's sampling); B the smallest in [2, 64] reaching thr V, the tile at least 2 E.
        from scipy import ndimage
        sz, ext, fg, vx = [], [], [], []
        for v in vols:
            y = v["y"].reshape(v["x"].shape[1:]).numpy()
            sp = np.asarray([float(a) for a in v["ctx"][:3]])
            m = (y > 0) & (y < 255)
            lab, n = ndimage.label(m, np.ones((3, 3, 3)))
            if n:
                sz += np.bincount(lab.ravel())[1:].tolist()
                ext += [float((np.array([q.stop - q.start for q in sl]) * sp).max()) for sl in ndimage.find_objects(lab)]
            fg.append(float(m.mean())), vx.append(float(np.prod(sp)) ** (1 / 3))
        sz = np.asarray(sz, float)
        L = float((sz ** 2).sum() / max(sz.sum(), 1.0))
        f, E, h = float(np.mean(fg)), float(np.percentile(ext, 95)) if ext else 160.0, float(np.median(vx))
        tmin = (2 * E / h) ** 3
        B = next((b for b in range(2, 65) if V / b >= tmin and b / 3 * min(L, V / b) + 2 / 3 * f * V >= thr * V), 64)
        B = max(2, min(B, int(V // max(tmin, 1.0))))
        if lm:   # landmark localisation (one named point per class per scan, heatmap regression): the target is the whole
            B = 2   # constellation, the largest tile, then refilled below (01/11: the lesion rule gave fetal 64 tiles of 28^3,
            # val MRE 10 mm). Instance detection (cmbpts: 2 mm spheres around point annotations) keeps the lesion rule.
        mean_tv = lambda n: float(np.mean([math.prod(min(n * a, s) for a, s in zip(v["align"], v["x"].shape[1:])) for v in vols]))
        max_tv = lambda n: max(math.prod(min(n * a, s) for a, s in zip(v["align"], v["x"].shape[1:])) for v in vols)
        # (05/11) the cells from the mean tile only: the largest scan used to set them for every scan (TotalSeg's 0.33 mm
        # scans held all its tiles at 3 cells = 48 mm); a scan too large is cut by tile_of (voxels x classes <= TILE_MAX)
        TILE_CELLS = max([1] + [n for n in range(1, 21) if mean_tv(n) <= V / B])
        t = mean_tv(TILE_CELLS)
        # the tile rounds down to whole 16 mm cells: refill the budget with that tile (01/11: Willis kept B = 2 on 127^3
        # tiles, 49 % of the budget; B' had 4)
        B = min(64, max(B, int(V // t)))
        b = B   # (05/11) no cut above 14 classes (TotalSeg: 58 -> 2 tiles, 4 % of the budget); memory: SFC_SEG_CHUNK
        print(f"[seg] planner thr {thr}: L {L:.0f} voxels, f {f:.4f}, E95 {E:.1f} mm -> {b} tiles of {TILE_CELLS} cells "
              f"(mean {t ** (1 / 3):.0f}^3, largest {max_tv(TILE_CELLS) ** (1 / 3):.0f}^3), {b * t / V:.0%} of the budget on average; "
              f"was {job['batch']}", flush=True)
        job["batch"] = b
        return
    if vols and vols[0].get("align"):
        tv = lambda n: max(math.prod(min(n * a, s) for a, s in zip(v["align"], v["x"].shape[1:])) for v in vols)
        while TILE_CELLS > 1 and tv(TILE_CELLS) > V // 2:
            TILE_CELLS -= 1
        t = tv(TILE_CELLS)
        b = min(64, max(2, V // t))
        b = max(2, int(b * min(1.0, 14 / k)))
    else:
        t = math.prod(patch) if patch else max(math.prod(v["x"].shape[1:]) for v in vols)
        b = min(64, max(2, V // t))
    print(f"[seg] voxel budget {V}: largest tile {t} voxels ({t ** (1 / 3):.0f}^3), {TILE_CELLS} cells, batch {b} "
          f"({b * t / V:.0%} of the budget; was {job['batch']})", flush=True)
    job["batch"] = b


FT_SCALES = (16.0, 8.0, 4.0, 2.0)   # (08/11) the encoder's reading scales (mm), per-scale PEFT


class Scale:
    """(08/11, Francesco: per-scale PEFT) the scale the encoder reads now: patch_embed.patch_mm, set around every read
    (fm/hier_dec._read_snake, token_cache.read / read_grad) -> its index in FT_SCALES. Read when a weight is built in
    the forward, so nothing may be recomputed in the backward (checkpointing off: SFC_FT_FINE_CKPT=0, SFC_FT_CKPT=0),
    else the backward would rebuild it at the scale restored after the read. Test-time ablations on the same
    checkpoint: SFC_FT_SCALE_AS="2:16" a scale read with another's adapters and offsets; SFC_FT_OFFSETS_OFF=1 the
    offsets (gates, decay) zeroed; SFC_FT_LORA_OFF="4,2" those scales' LoRA off."""

    def __init__(self, pe):
        self.pe = pe
        self.As = {float(a): float(b) for a, b in (q.split(":") for q in os.environ.get("SFC_FT_SCALE_AS", "").split(",") if q)}
        self.lora_off = {float(q) for q in os.environ.get("SFC_FT_LORA_OFF", "").split(",") if q}
        self.off = os.environ.get("SFC_FT_OFFSETS_OFF") == "1"

    def __call__(self) -> tuple[int, float]:   # (index of the adapters used, the scale read)
        mm = float(self.pe.patch_mm)
        a = self.As.get(mm, mm)   # (08/11) AlignUNet reads cells of any edge (9.6, 2.4 mm ...): the nearest grain's adapters
        return min(range(len(FT_SCALES)), key=lambda i: abs(math.log(FT_SCALES[i] / a))), mm


class LoRA(torch.nn.Module):
    """A weight's parametrization W -> W + 2 B A (LoRA, Hu et al. 2022; alpha = 2r, Biderman et al. 2024), B = 0: the
    pretrained net exactly at the start. A conv kernel is an [out, in k^3] matrix (loralib's ConvLoRA, Aleem et al.
    2024 on nnU-Net-style CNNs). A parametrization, not a wrapper: code reading `.weight` itself (timm's Eva:
    F.linear(x, self.qkv.weight)) gets the adapted weight too. `scale` (08/11): one (A, B) per reading scale."""

    def __init__(self, w: torch.Tensor, r: int, scale: Scale | None = None):
        super().__init__()
        n, S = w[0].numel(), (len(FT_SCALES),) if scale else ()
        self.A = torch.nn.Parameter(torch.empty(*S, r, n, device=w.device).uniform_(-n ** -0.5, n ** -0.5))  # loralib's init
        self.B = torch.nn.Parameter(torch.zeros(*S, w.shape[0], r, device=w.device))
        self.scale = scale

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            return w + (2 * self.B @ self.A).view_as(w).to(w.dtype)
        i, mm = self.scale()
        return w if mm in self.scale.lora_off else w + (2 * self.B[i] @ self.A[i]).view_as(w).to(w.dtype)


class Offset(torch.nn.Module):
    """(08/11) a per-scale additive offset, 0 at the start: as a parametrization of GDN-2's decay (A_log, dt_bias: the
    bias of the decay's input) and as an output hook of the bias-free gate projections (erase b_proj, write w_proj) and
    of the attention's output gate (g). LoRA had spent its rank on them (sfc_runs/ft/lora_bias.log: erase of blocks 0-1
    ~95 % constant, ~12 logits). scale None (11/11, finetune.offsets without scales): one offset for every scale."""

    def __init__(self, shape, scale: Scale | None, device):
        super().__init__()
        S = (len(FT_SCALES),) if scale else ()
        self.d, self.scale = torch.nn.Parameter(torch.zeros(*S, *shape, device=device)), scale
        self.off = scale.off if scale else os.environ.get("SFC_FT_OFFSETS_OFF") == "1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.off:
            return x
        return x + (self.d if self.scale is None else self.d[self.scale()[0]]).to(x.dtype)


def peft(net, pre: set, r: int, full: tuple, where: tuple, norms: bool = True, scales: bool = False,
         full_re: str = "", offsets: bool = False) -> tuple[set, set]:
    """PEFT of the pretrained part (06/11, Francesco: LoRA after P2's fine-tuning; notes/LORA.md), `pre` its parameter
    names: LoRA of rank r (0: none) on every Linear / Conv2d / Conv3d weight (groups 1) under a `where` prefix; trained
    whole: the 1-D parameters (norms, biases: BitFit; not with norms False) and everything under a `full` prefix (ours: the patch embedding,
    surgical fine-tuning for an input-level shift, Lee et al. 2023); the rest frozen, GDN-2's decays (A_log, dt_bias)
    too (LoRA fails on SSM modules, Galim et al. 2025). scales (08/11, Francesco): one LoRA of rank r per reading scale
    (FT_SCALES) and per-scale Offsets on erase / write / decay / the attention gate. offsets (11/11, Francesco): the
    same Offsets, one for every scale, with a plain LoRA: the bias-free gates get their bias (LoRA spent rank 1 on it,
    notes/GATE_IDEA.md) and the decays (A_log, dt_bias) are no longer frozen (as offsets: own lr, no weight decay).
    -> (trained, frozen) names of the pretrained part."""
    from torch.nn.utils import parametrize
    sc = None
    if scales:
        bad = {k: os.environ.get(k, "1") for k in ("SFC_FT_FINE_CKPT", "SFC_FT_CKPT")}
        assert set(bad.values()) == {"0"}, f"per-scale PEFT needs every recomputation off: {bad}"
        net.encoder.grad_checkpoint = False
        sc = Scale(net.encoder.patch_embed)
    for name, m in list(net.named_modules()):
        if (r and isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)) and getattr(m, "groups", 1) == 1
                and f"{name}.weight" in pre and name.startswith(where) and not name.startswith(full)):
            parametrize.register_parametrization(m, "weight", LoRA(m.weight, r, sc))
    if sc is not None or offsets:
        for name, m in list(net.encoder.named_modules()):
            gates = [m.b_proj, m.w_proj] if hasattr(m, "A_log") else [m.g] if hasattr(m, "qkv") else []
            for lin in gates:
                lin.offset = Offset((lin.out_features,), sc, lin.weight.device)
                lin.register_forward_hook(lambda mod, inp, out: mod.offset(out))
            if hasattr(m, "A_log"):
                for a in ("A_log", "dt_bias"):
                    parametrize.register_parametrization(m, a, Offset(getattr(m, a).shape, sc, getattr(m, a).device))
    p = dict(net.named_parameters())
    keep = {n for n in pre & p.keys()
            if n.startswith(full) or norms and p[n].ndim <= 1 and n.rsplit(".", 1)[-1] not in ("A_log", "dt_bias")
            or full_re and re.search(full_re, n)}   # (11/11) e.g. GDN-2's depthwise short convs (no LoRA: groups > 1)
    lora = {n for n in p if ".parametrizations." in n and not n.endswith(".original") or ".offset." in n}
    orig = {n for n in p if n.endswith(".original")}
    return lora | keep, orig | (pre & p.keys()) - keep


def train(job: dict) -> None:
    dev = torch.device(job["device"])
    if gb := os.environ.get("SFC_GPU_GB"):   # (04/11) a smaller GPU's memory, emulated: fine-tuning must fit 12 GB
        i = dev.index or 0
        torch.cuda.set_per_process_memory_fraction(float(gb) * 2**30 / torch.cuda.get_device_properties(i).total_memory, i)
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
    # fine-tuning (04/11, P2): job["finetune"] = {"enc_lr": encoder lr / decoder lr, "warm": fraction of the iterations
    # with the encoder frozen (decoder warm-up), "init": optional, a frozen run's .best.pt (its decoder)}: nothing frozen, the encoder in its own lr
    # group; ours: the 16 / 8 mm tokens read now with gradient (token_cache.GradCache), not from the token cache
    FT = job.get("finetune")
    enc_names = frozen_names
    if FT:
        frozen_names = set()
        if FT.get("init"):
            st = torch.load(FT["init"], map_location="cpu", weights_only=False)["best"]["state"]
            bad = set(st) - set(net.state_dict())
            assert not bad, sorted(bad)[:4]
            net.load_state_dict(st, strict=False)
            print(f"[seg] finetune: decoder from {FT['init']} ({len(st)} tensors)", flush=True)
        if "lora" in FT:   # (06/11) PEFT: FT["lora"] rank (0: norms + FT["full"] only), FT["full"] / FT["lora_in"] prefixes
            tup = lambda v: (v,) if isinstance(v, str) else tuple(v)   # a YAML string or list of prefixes
            enc_names, frozen_names = peft(net, enc_names, int(FT["lora"]), tup(FT.get("full", ())),
                                           tup(FT.get("lora_in", "")), bool(FT.get("norms", 1)), bool(FT.get("scales", 0)),
                                           str(FT.get("full_re", "")), bool(FT.get("offsets", 0)))
            print(f"[seg] finetune: LoRA r {FT['lora']} on {sum('.original' in n for n in frozen_names)} weights, "
                  f"{len(enc_names)} pretrained tensors trained, {len(frozen_names)} frozen", flush=True)
        if enc_names:   # (06/11) none (lora 0, norms 0, no full): the control, the decoder alone from `init` on the
            net.finetune = True   # cached tokens, the frozen part in eval mode as in the table's run
            if hasattr(net, "encoder"):
                net.encoder.grad_checkpoint = not FT.get("scales")   # (08/11) per-scale PEFT: no recomputation
            if hasattr(net, "cache"):
                from .token_cache import GradCache
                old = net.cache
                net.cache = GradCache(old.root, net)
                net.cache.levels = old.levels
    LM = job.get("select") == "mre"   # landmark task: heatmap regression (landmark_step, sigmoid maps, MRE selection)
    net.landmark = LM
    DET = job.get("select") == "det"   # point detection (cmbpts, user 28/10): component centroids matched within DET_MM
    # the true points: the task's points csv (id, i, j, k: GT voxels) when given, else the label's components' centroids
    # (2 mm spheres closer than ~4 mm merge: ~3 % of the cmbpts lesions, Ettore 28/10)
    PTS = read_points(job["items"][0]["points"]) if DET and job["items"][0].get("points") else None
    if hasattr(net, "tile_max"):   # fm/hier_dec: dense activations, tiles of ~160^3 voxels on the token grid
        global TILE_MAX
        TILE_MAX = net.tile_max(k)
    # token cache (args.token_cache, fm/hier_dec): tiles start on each volume's 16 mm grid and pass (volume, corner) in ctx
    tcache = getattr(net, "cache", None)
    vid = {it["id"]: n for n, it in enumerate(job["items"])}
    if tcache is not None:
        tcache.ids = [it["id"] for it in job["items"]]
    del ref, pre, ref_p
    if os.environ.get("SFC_CL3D") == "1":   # (06/11) 3D conv weights NDHWC: cuDNN's bf16 kernels without layout
        for m in net.modules():              # conversions (~18 % of the GPU time of the TotalSeg fine-tuning profile)
            if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)) and "weight" in m._parameters:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    n_frozen = n_train = 0
    for n, p in net.named_parameters():
        p.requires_grad_(n not in frozen_names)
        n_frozen += p.numel() * (not p.requires_grad)
        n_train += p.numel() * p.requires_grad
    frozen_mods = [] if FT and enc_names else [m for n, m in net.named_modules() if n and is_frozen(n)]
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
        if hasattr(net, "vol_stats"):   # fm/hier_dec HierUNet: z-score per scan on its nonzero voxels (nnU-Net's)
            nz = r["x"][r["x"] != 0].float()
            net.vol_stats[vid[it["id"]]] = (float(nz.mean()), float(nz.std()) + 1e-6)
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
    if hasattr(net, "plan"):   # fm/hier_dec HierUNet: the task's fingerprint (training scans) -> strides
        net.plan([it["id"] for it in split["train"]])
    # on disk as the run goes, so a killed job resumes: the training state every 10 min (.last.pt), the best state
    # until the test is written (.best.pt), the test volumes done (.test.json)
    out_p = Path(job["out"])
    ckpt, last, part = (out_p.with_suffix(s) for s in (".best.pt", ".last.pt", ".test.json"))
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # + the wrapper's whole config and the tile rule: a run of another decoder in the same folder is never resumed
    # (20/10: the token-cache runs resumed the cancelled tile runs' states, same iters / lr / seed)
    if os.environ.get("SFC_TILE_RULE") == "1" and tcache is not None and not LM:   # (31/10: landmarks are points,
        # their extent says nothing of the context they need: with the rule's 56^3 tiles fetal test MRE 1.86 vs 1.43)
        tile_rule(job, [load(it) for it in split["train"][:40]], LM, k)
    if os.environ.get("SFC_VOX_BUDGET"):
        vox_budget(job, [load(it) for it in split["train"][:None if os.environ.get("SFC_PLANNER_THR") else 40]]
                   if tcache is not None or not patch else [], patch, k, LM)
    key = {n: job[n] for n in ("init", "iters", "val_every", "batch", "lr", "seed")} | {"train": len(split["train"])} | \
        {"cfg": json.dumps(job["cfg"], sort_keys=True, default=str), "tile_cells": TILE_CELLS} | \
        ({"finetune": json.dumps(FT, sort_keys=True)} if FT else {})
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
        if hasattr(net, "plan"):   # a live token cache is written by load(): the fingerprint again, now complete
            net.plan([it["id"] for it in split["train"]])
        val_v = [load(it) for it in split["val"]]
        data = Patches(train_v, patch, k, rng, fit=getattr(net, "any_tile", False), snap=getattr(net, "snap", None),
                       fixed=getattr(net, "fixed_patch", None))
        # the training scans scored at each validation (user 30-31/10): the first TRAIN_EVAL holding foreground
        tr_v = [v for v in train_v if bool(((v["y"] > 0) & (v["y"] < 255)).any())][:TRAIN_EVAL]
        del train_v
        params = [p for p in net.parameters() if p.requires_grad]
        if FT:   # the pretrained part at enc_lr x the decoder's lr (discriminative lr), LoRA's factors at lora_lr x
            # (LoRA's best lr ~10x full fine-tuning's, Biderman et al. 2024), the same poly decay
            # (08/11) side 3: the per-scale offsets (gates, decay) at the absolute lr SFC_FT_OFF_LR (default 5e-3: AdamW
            # moves a bias ~lr per step, LoRA's erase bias reached ~12 logits), no weight decay, init 0
            side = lambda n: (0 if n not in enc_names else 3 if ".offset." in n or n.endswith(".0.d")
                              else 2 if ".parametrizations." in n else 1)
            mult = (1.0, float(FT.get("enc_lr", 0.1)), float(FT.get("lora_lr", 1.0)),
                    float(os.environ.get("SFC_FT_OFF_LR", "5e-3")) / job["lr"])
            groups = [{"params": [p for n, p in net.named_parameters() if n not in frozen_names and side(n) == g],
                       "lr": job["lr"] * mult[g], "side": g, **({"weight_decay": 0.0} if g == 3 else {})} for g in range(4)]
            groups = [g for g in groups if g["params"]]
            for g in groups[1:]:
                print(f"[seg] finetune: {('', 'encoder', 'LoRA', 'offsets')[g['side']]} {sum(p.numel() for p in g['params']) / 1e6:.2f}M "
                      f"at lr x{mult[g['side']]:g} ({g['lr']:.1e}, weight decay {g.get('weight_decay', 1e-4):g})", flush=True)
        opt = torch.optim.AdamW(groups if FT else params, lr=job["lr"], weight_decay=1e-4)
        # (06/11) FT["ramp"]: each group's lr ramps up over that fraction of the iterations from the iteration it starts
        # training (a warm-started decoder 1, the encoder after the warm-up): Adam's first steps are ~lr whatever the
        # gradient, and P2's warm-started decoders and its encoder took them at full lr; a decoder from scratch keeps
        # the table's schedule (start -R: no ramp)
        R = max(1, int(float(FT.get("ramp", 0)) * iters)) if FT else 1
        w0 = int(float(FT.get("warm", 0)) * iters) if FT else 0
        starts = [w0 if g.get("side") else 0 if (FT or {}).get("init") else -R for g in opt.param_groups]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [
            lambda i, s=s: (1 - i / iters) ** 0.9 * min(1.0, max(0.0, (i + 1 - s) / R)) for s in starts])
        best, curve, start, t0 = {"val": -1.0}, [], 1, time.time()
        vp = False
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
            vp = bool(state.get("val_pending"))   # (05/11) killed inside the validation of `it`: redo only it
            start -= vp
            print(f"[seg] resumed from {last} at iter {state['it']} ({state['min']:.1f} min done)", flush=True)
        t_save = time.time()

        def save_last(it: int, val_pending: bool = False) -> None:
            torch.save({"key": key, "it": it, "net": trained(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state() if dev.type == "cuda" else None,
                        "best": best, "curve": curve, "min": (time.time() - t0) / 60, "val_pending": val_pending}, last.with_suffix(".tmp"))
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
        lsum = [0.0, 0]   # the training loss summed since the last val line (user 30/10: log the mean, not the last batch)
        run_l, run_d = [0.0, 0], None   # the same every TRAIN_LOG iterations, + the batches' soft Dice (any_tile nets)
        # SFC_SEG_PROFILE="w,n" (27/10, where the step's time goes): after w iterations, n under cProfile + torch.profiler,
        # the tables to <out>.prof.txt, then the run stops (no val, no test)
        pw, pn = (int(v) for v in os.environ.get("SFC_SEG_PROFILE", "0,0").split(","))
        for it in range(start, iters + 1):
            if FT and it in (start, int(FT.get("warm", 0) * iters) + 1):   # decoder warm-up: the encoder frozen for the
                on = it > int(FT.get("warm", 0) * iters)                      # first `warm` of the iterations (asparagus'
                for n, p in net.named_parameters():                            # recipe; same for every model)
                    if n in enc_names:
                        p.requires_grad_(on)
                print(f"[seg] finetune: encoder {'trained' if on else 'frozen (decoder warm-up)'} from iter {it}", flush=True)
            if vp:   # resumed at a pending validation: its step was done and saved before the kill
                vp, tc = False, time.time()
            else:
                if FT and hasattr(net.cache if hasattr(net, "cache") else None, "clear"):   # GradCache: each step reads its
                    net.cache.clear()                                                         # scans anew (every branch below)
                if pn and it in (start + pw, start + pw + pn):
                    torch.cuda.synchronize()
                    if it == start + pw:
                        import cProfile
                        prof = (cProfile.Profile(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                                      torch.profiler.ProfilerActivity.CUDA], record_shapes=True), time.time())
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
                                                                  + "\n\nby input shape (06/11):\n" + prof[1].key_averages(group_by_input_shape=True).table(
                                                                      sort_by="self_device_time_total", row_limit=40, max_shapes_column_width=120)
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
                elif int(os.environ.get("SFC_SEG_CHUNK", "0")):   # (04/11, opt-in, frozen or fine-tuned: e.g. a 12 GB GPU)
                    # the batch's tiles SFC_SEG_CHUNK at a time, one backward each, the same gradient (micro_step)
                    loss = micro_step(net, x, y, c, k, amp, dev, int(os.environ["SFC_SEG_CHUNK"]))
                    logits = torch.zeros(1, device=dev)
                elif isinstance(x, list) and getattr(net, "any_tile", False):   # tiles of several shapes, grouped by shape
                    loss = group_step(net, x, y, c, k, amp, dev)
                    logits = torch.zeros(1, device=dev)
                elif isinstance(x, list) or (getattr(w, "seg_microbatch", False) and k > 10):   # lists: tiles of several shapes
                    loss = micro_step(net, x, y, c, k, amp, dev)
                    logits = torch.zeros(1, device=dev)
                else:
                    sel_target(net, y)
                    with amp():
                        logits = call(net, x.to(dev, non_blocking=True).float(), c)
                    loss = loss_fn(logits, y.to(dev), k) + pop_aux(net)
                    loss.backward()
                    with torch.no_grad():   # the batch's soft Dice sums for the running log (as group_step's)
                        group_step.sums = list(batch_sums(logits.detach(), y.to(dev), k)[2:])
                if FT and hasattr(getattr(net, "cache", None), "flush"):   # GradCache: the maps' gradient through the encoder
                    net.cache.flush()
                if FT and it % TRAIN_LOG == 0:   # the encoder's share of the gradient (fine-tuning: must be > 0 once trained)
                    ge = [p.grad.norm() for n, p in net.named_parameters() if n in enc_names and p.grad is not None]
                    print(f"[seg] it {it} encoder grad norm {float(torch.stack(ge).norm()) if ge else 0.0:.3e} "
                          f"({len(ge)} tensors)", flush=True)
                gn = torch.nn.utils.clip_grad_norm_(params, 12)
                lsum = [lsum[0] + float(loss), lsum[1] + 1]
                run_l = [run_l[0] + float(loss), run_l[1] + 1]
                if getattr(group_step, "sums", None) is not None:   # user 31/10: the training batches' Dice as it goes
                    run_d = [a + b for a, b in zip(run_d, group_step.sums)] if run_d else list(group_step.sums)
                    group_step.sums = None
                if it % TRAIN_LOG == 0:
                    td = "" if not run_d else \
                        f" soft Dice {float(((2 * run_d[0][1:] + 1e-5) / (run_d[1][1:] + run_d[2][1:] + 1e-5)).mean()):.4f}"
                    print(f"[seg] it {it} train (last {run_l[1]} batches): loss {run_l[0] / max(run_l[1], 1):.3f}{td} "
                          f"({(time.time() - t0) / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
                    run_l, run_d = [0.0, 0], None
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
                # (05/11, Francesco) the state saved before every validation: a job killed inside it resumes here, at
                # this validation (TotalSeg in 30-min blocks redid 9889 -> 10000 and a 13-min val four times)
                save_last(it, val_pending=True)
                t_save = time.time()
                net.eval()
                # probabilities kept for the landmark tasks (a point per landmark from each class's map); labels: argmax
                # scan by scan, only the labels kept (31/10: TotalSeg's ~50-class whole-body probabilities of every val
                # scan held at once, on the GPU when they fit, ran out of memory)
                lab = (lambda p: p.to(dev)) if job.get("select") == "mre" else (lambda p: p.argmax(0).to(dev))
                md = "det" if job.get("select") == "det" else True   # det: detection F1 (centroids within DET_MM)
                f1s = job.get("select") in ("f1", "det")

                def scores(v, val_scan: bool, lab=lab, md=md, f1s=f1s) -> dict:   # one scan's Dice (+ MRE, + F1 sums)
                    p = lab(sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"])) if val_scan else \
                        sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"]).argmax(0).to(dev)
                    r = {"dice": dice_on(p.argmax(0) if p.dim() == 4 else p, v["y"].to(dev), k)}
                    if val_scan and LM:
                        r["mre"] = mre_on(p, v["y"].to(dev), v["sp"] if "sp" in v else v["ctx"][:3], k)
                    if f1s:
                        r["f1"] = [_score((p == c).cpu().numpy(), (v["y"] == c).numpy(), md, sp=v.get("sp"), pts=v.get("pts"))
                                   for c in range(1, k)]
                    return r
                # (07/11, Francesco: validations left hanging) scan by scan into <out>.valpart.pt: a job killed inside a
                # validation resumes at the scan it reached (TotalSeg's 100 val scans outlasted a 45-min block, ResEnc-L
                # redid the validation of 1250 nine times)
                vpart_p = out_p.with_suffix(".valpart.pt")
                vpart = torch.load(vpart_p, map_location="cpu", weights_only=False) if vpart_p.exists() else {}
                if vpart.get("it") != it or vpart.get("key") != key:
                    vpart = {"it": it, "key": key, "val": {}, "train": {}}
                elif vpart["val"]:
                    print(f"[seg] validation of {it} resumed: {len(vpart['val'])} / {len(val_v)} val scans done", flush=True)
                for side, vols in (("val", val_v), ("train", [] if LM else tr_v)):
                    for n, v in enumerate(vols):
                        if n in vpart[side]:
                            continue
                        vpart[side][n] = scores(v, side == "val")
                        torch.save(vpart, vpart_p.with_suffix(".tmp"))
                        os.replace(vpart_p.with_suffix(".tmp"), vpart_p)
                        if side == "val" and (n + 1) % 20 == 0:
                            print(f"[seg] validation of {it}: {n + 1} / {len(vols)} val scans", flush=True)
                vr = [vpart["val"][n] for n in range(len(val_v))]
                val = float(np.nanmean([r["dice"] for r in vr]))
                rec = {"iter": it, "loss": lsum[0] / max(lsum[1], 1), "val_dice": val, "min": (time.time() - t0) / 60}
                score, msg = val, f"val dice {val:.4f}"
                if not LM:   # user 30/10: the Dice on a few training scans too (overfit)
                    tr = [vpart["train"][n] for n in range(len(tr_v))]
                    rec["train_dice"] = float(np.nanmean([r["dice"] for r in tr]))
                    msg = f"train dice {rec['train_dice']:.4f} " + msg
                    if f1s:   # the lesion-wise F1 on them too (user 31/10: microbleeds)
                        rec["train_f1"] = float(np.nanmean([_metric([r["f1"][c] for r in tr], True) for c in range(k - 1)]))
                        msg = f"train F1 {rec['train_f1']:.3f} " + msg
                if LM:   # landmarks (user, 22/10): chosen on the val radial error, not the Dice
                    mre = float(np.mean([r["mre"] for r in vr]))
                    rec["val_mre"], score, msg = mre, -mre, msg + f" val MRE {mre:.2f} mm"
                if f1s:   # small lesions (CMB, user 23/10): chosen on the val lesion-wise F1
                    f1v = float(np.nanmean([_metric([r["f1"][c] for r in vr], True) for c in range(k - 1)]))
                    rec["val_f1"], score, msg = f1v, f1v, msg + f" val F1 {f1v:.3f}"
                curve.append(rec)
                tm["val"] = time.time() - tc
                print(f"[seg] it {it} loss {rec['loss']:.3f} {msg} ({(time.time() - t0) / 60:.1f} min; s: data {tm['data']:.0f}"
                      f" step {tm['step']:.0f} val {tm['val']:.0f})", flush=True)
                tm, lsum = dict.fromkeys(tm, 0.0), [0.0, 0]
                if "state" not in best or score > best.get("score", best["val"]):
                    best = {"val": val, "iter": it, "score": score, "state": trained()} | \
                        ({"val_mre": rec["val_mre"]} if "val_mre" in rec else {}) | \
                        ({"val_f1": rec["val_f1"]} if "val_f1" in rec else {})
                save_last(it)   # (07/11) the validation done: a kill now resumes after it, not inside it
                t_save = time.time()
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
        # the last iteration's state kept too (user 30/10: never delete checkpoints; diagnosis of the val's choice)
        torch.save({"key": key, "it": it, "state": trained()}, out_p.with_suffix(".final.pt"))
        last.unlink(missing_ok=True)
    if hasattr(net, "read_taus") and not LM and "read_tau" not in best:
        # fm/hier_dec (user 30/10): the read threshold for the test chosen on val with the best state, one pass (every
        # cell read, each threshold a mask: the decoder returns their log-probabilities side by side), by the task's own
        # selection score (Dice, or the lesion / detection F1). ponytail: landmarks keep the training threshold
        load_trained(best["state"])
        net.eval()
        # (08/11) AlignUNet (two_heads): one pass with both heads' probabilities, every alpha of ALPHA_GRID mixed after
        # (sliding sums probabilities linearly: the mix of the sums is the sum of the mixes) and kept per val scan in
        # <out>.heads, so the chosen alpha's val predictions are composed there, not read again for the post-processing
        two = getattr(net, "two_heads", False)
        if two:
            from .hier_dec import ALPHA_GRID
        net.full, net.read_taus = True, ["heads"] if two else list(net.TAUS)
        taus = list(ALPHA_GRID) if two else net.read_taus
        nt, hd = len(taus), out_p.with_suffix(".heads")
        # (05/11) scored volume by volume and kept in <out>.taus.json: a job killed inside this pass resumes it (TotalSeg:
        # longer than a 45-min block, redone by every block); the same scores as keeping every prediction
        fdet = job.get("select") in ("f1", "det")
        md = ("det" if job["select"] == "det" else True) if fdet else None
        tf, tk = out_p.with_suffix(".taus.json"), {"best_iter": best["iter"], "taus": [float(t) for t in taus]}
        st = json.loads(tf.read_text()) if tf.exists() else {}
        st = st if st.get("key") == tk else {"key": tk, "ids": [], "per": []}
        t_th = time.time()
        for n_, it in enumerate(split["val"]):
            if it["id"] in st["ids"]:
                continue
            v = load(it)
            p = sliding(net, v["x"].to(dev).float(), patch, k * (2 if two else nt), amp, v["ctx"])
            if two:   # the heads x 2 (each was / 2) as uint8, the GT mapping with them; mixed and scored on the GPU
                p = (p.to(dev) * 2).clamp_(0, 1)   # (fp16; 11 alphas on a CPU accumulator: ~2x the pass)
                hd.mkdir(parents=True, exist_ok=True)
                np.savez(hd / f"{it['id']}.npz", p=(p * 255).round().to(torch.uint8).cpu().numpy(),
                         grid_from_gt=np.asarray(v["grid_from_gt"]))
            rows = []
            for a in range(nt):
                q = (taus[a] * p[:k] + (1 - taus[a]) * p[k:] if two else p[a * k:(a + 1) * k]).argmax(0).to(torch.uint8)
                if fdet:
                    q = q.cpu()
                rows.append([[float(x) for x in _score((q == c).numpy(), (v["y"] == c).numpy(), md, sp=v.get("sp"),
                                                       pts=v.get("pts"))] for c in range(1, k)] if fdet
                            else dice_on(q, v["y"].to(q.device), k))
            st["ids"].append(it["id"])
            st["per"].append(rows)
            tf.write_text(json.dumps(st))
            print(f"[seg] read threshold: val {len(st['ids'])}/{len(split['val'])} ({(time.time() - t_th) / 60:.1f} min)",
                  flush=True)
            del v, p
        per = st["per"]
        if fdet:
            by = [float(np.nanmean([_metric([tuple(r[a][c - 1]) for r in per], True) for c in range(1, k)])) for a in range(nt)]
        else:
            by = [float(np.nanmean([r[a] for r in per])) for a in range(nt)]
        # 30/10: reading every cell is not the training read (the encoder reads the chosen cells as one sequence, so
        # their tokens change): the training read (None) is a candidate too, its val score the selection's own
        by_t = {str(t): d for t, d in zip(taus, by)} | {"train": float(best.get("score", best["val"]))}
        b_ = max(by_t, key=lambda t: by_t[t])
        best |= {"read_tau": None if b_ == "train" else float(b_), "val_by_tau": by_t}
        print(f"[seg] read threshold {best['read_tau']} (val {job.get('select') or 'dice'} per threshold "
              f"{best['val_by_tau']})", flush=True)
        torch.save({"key": key, "best": best, "curve": curve}, ckpt)
        if two:   # the chosen alpha's val predictions from the kept heads (a scan missing here: read again below)
            a, pv_ = best["read_tau"] if best["read_tau"] is not None else 0.5, out_p.with_suffix(".pred_val")
            for f in sorted(hd.glob("*.npz")) if hd.exists() else []:
                z = np.load(f)
                h = torch.from_numpy(z["p"]).float() / 255
                save_pred(pv_, f.stem, a * h[:k] + (1 - a) * h[k:], z["grid_from_gt"])
            shutil.rmtree(hd, ignore_errors=True)   # both heads: an intermediate (the val predictions are kept)
            print(f"[seg] alpha {a}: val predictions composed from the heads (no second val pass)", flush=True)
    if best.get("read_tau") is not None:   # the test reads every cell, its threshold a mask (as chosen)
        net.full, net.read_taus = True, [best["read_tau"]]
    elif hasattr(net, "read_taus"):   # the training read
        net.full, net.read_taus = False, [net.tau]
    if job.get("val_only"):  # pretraining tuning: the best val Dice, no test volume read
        if job.get("save_pred"):   # the best state's val predictions kept anyway (user, 24/10: lesion counts on val)
            load_trained(best["state"])
            net.eval()
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
    rule, f1 = {}, "det" if DET else job.get("select") == "f1"
    pv = out_p.with_suffix(".pred_val")
    # (05/11) the chosen rule kept on disk: a job killed after choosing it does not choose it again (SOOP stroke,
    # 45-min blocks: ~50 silent minutes of val loading + search, redone by every block)
    rf, rkey = out_p.with_suffix(".postproc.json"), {"best_iter": best["iter"], "read_tau": best.get("read_tau")}
    if not LM and rf.exists() and (rr := json.loads(rf.read_text()))["key"] == rkey:
        rule = {int(c): r for c, r in rr["rule"]}
        print(f"[seg] post-processing rule from {rf}", flush=True)
    elif not LM:   # val predictions of the best state -> the post-processing rule (chosen on val only)
        vals, t_pp = [], time.time()
        print(f"[seg] post-processing: {len(split['val'])} val predictions", flush=True)
        for n_, it in enumerate(split["val"]):
            if n_ % 10 == 0:
                print(f"[seg] post-processing: val {n_}/{len(split['val'])} ({(time.time() - t_pp) / 60:.1f} min)", flush=True)
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
        print(f"[seg] post-processing: choosing the rule ({(time.time() - t_pp) / 60:.1f} min)", flush=True)
        rule = choose_postproc(vals, k, sizes, f1, out_p.with_suffix(".postproc.part.json"), rkey)
        rf.write_text(json.dumps({"key": rkey, "rule": [[c, r] for c, r in rule.items()]}))
        del vals
    tkey = key | {"best_iter": best["iter"], "postproc": {str(c): r for c, r in rule.items()}} | \
        ({"read_tau": best["read_tau"]} if "read_tau" in best else {})   # (only when chosen: other runs' partial tests resume)
    # (05/11, SOOP: 556 test volumes, ~6 h on one GPU) SFC_TEST_SHARD=i/n: this job tests every n-th volume from the
    # i-th into its own <out>.test.s<i>of<n>.json and stops; a job without it merges the shards' volumes first, then
    # tests the rest and writes the result as before
    shard = os.environ.get("SFC_TEST_SHARD")
    si, sn = (int(a) for a in shard.split("/")) if shard else (0, 1)
    skip = set()
    if shard:   # the volumes the unsharded test already did are not done again
        mp = json.loads(part.read_text()) if part.exists() else {}
        skip = set(mp["res"]["test_ids"]) if mp.get("key") == tkey else set()
        part = out_p.with_suffix(f".test.s{si}of{sn}.json")
    done = json.loads(part.read_text()) if part.exists() else {}
    res = done["res"] if done.get("key") == tkey else {"per_volume": [], "per_class": [], "test_ids": []}
    if res["test_ids"]:
        print(f"[seg] test resumed from {part}: {len(res['test_ids'])} volumes done", flush=True)
    if not shard:
        for sp_ in sorted(out_p.parent.glob(out_p.stem + ".test.s*of*.json")):
            sd = json.loads(sp_.read_text())
            if sd.get("key") != tkey:
                continue
            have, n0 = set(res["test_ids"]), len(res["test_ids"])
            for j_, i_ in enumerate(sd["res"]["test_ids"]):
                if i_ not in have:
                    for f_ in ("per_volume", "per_class", "test_ids") + (("det",) if "det" in sd["res"] else ()):
                        res.setdefault(f_, []).append(sd["res"][f_][j_])
            print(f"[seg] test: {len(res['test_ids']) - n0} volumes from {sp_.name}", flush=True)
        part.write_text(json.dumps({"key": tkey, "res": res}))
    for it in split["test"][si::sn]:
        if it["id"] in res["test_ids"] or it["id"] in skip:
            continue
        t1 = time.time()
        v = load(it)
        t2 = time.time()
        probs = sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"])
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t3 = time.time()   # (05/11) where a test volume's time goes: load, sliding window, the rest below
        lab, _, _ = gt_labels(it["seg"], it.get("classes"), it.get("ignore"))
        pl = None
        if rule:   # the val-chosen post-processing, then scored as the argmax of its one-hot
            q = (probs.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy() if k <= 8 else None
            pl = postproc(probs.argmax(0).to(torch.uint8).cpu().numpy(), q, rule)
            probs_s = F.one_hot(torch.from_numpy(pl).long(), k).permute(3, 0, 1, 2).half()
        t4 = time.time()
        d, per = dice_full_res(probs_s if rule else probs, v["grid_from_gt"], lab, k, dev)
        t5 = time.time()
        if DET:   # detection sums (tp, points, false detections) per class on the model grid, as the val selection
            pm = pl if pl is not None else probs.argmax(0).cpu().numpy()
            yy = (v["y"][0] if v["y"].dim() == 4 else v["y"]).numpy()
            res.setdefault("det", []).append([_score(pm == c, yy == c, "det", sp=v["sp"], pts=v["pts"]) for c in range(1, k)])
        save_pred(out_p.with_suffix(".pred"), it["id"], probs, v["grid_from_gt"], lab=pl)
        t6 = time.time()
        res["per_volume"].append(d)
        res["per_class"].append(per)
        res["test_ids"].append(it["id"])
        part.write_text(json.dumps({"key": tkey, "res": res}))
        print(f"[seg] test {len(res['test_ids'])}/{len(split['test'])} {it['id']}: dice {d:.4f} grid "
              f"{tuple(v['x'].shape[1:])} {time.time() - t1:.0f}s (load {t2 - t1:.0f}, sliding {t3 - t2:.0f}, labels + post {t4 - t3:.0f}, "
              f"dice {t5 - t4:.0f}, save {t6 - t5:.0f}), peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.0f} GB", flush=True)
        del v, probs, lab
    if shard:
        print(f"[seg] test shard {shard}: {len(res['test_ids'])} volumes -> {part}", flush=True)
        return
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


def read_points(path: str) -> dict[str, np.ndarray]:
    """The task's points csv (id, i, j, k GT voxels) -> {id: [n, 3]}; csv, not pandas: segrun runs in the models' own
    envs, some without pandas (fm-asparagus, fm-medicalnet: the externals' prep failed on a top-level import, 28/10)."""
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["id"], []).append([float(r["i"]), float(r["j"]), float(r["k"])])
    return {k: np.asarray(v) for k, v in out.items()}


def true_points(pts, i: str, grid_from_gt) -> np.ndarray | None:
    """The task's true points of volume i (GT voxels, read_points) on the model grid [n, 3]; None without a points csv
    (the label's components then)."""
    if pts is None:
        return None
    if i not in pts:
        return np.zeros((0, 3))
    M = np.asarray(grid_from_gt, dtype=float)
    return pts[i] @ M[:3, :3].T + M[:3, 3]


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


_PP = None   # choose_postproc's (vals, boxes, sizes, f1) for its forked workers


def _pp_one(c: int) -> tuple:
    """One class of choose_postproc (a forked worker): (c, rule or None, base, best); (c, None, None, None) if absent."""
    vals, boxes, sizes, f1 = _PP
    if boxes is not None:   # this class's view of the val volumes (an empty 1-voxel box where it is absent)
        vals_c = []
        for (lab, q, y, *rest), bx in zip(vals, boxes):
            s = bx[c - 1] if c - 1 < len(bx) and bx[c - 1] is not None else tuple(slice(0, 1) for _ in lab.shape)
            qs = None if q is None else q[(slice(None), *s)]   # (probabilities: cropped with the class's box)
            if c - 1 >= len(bx) or bx[c - 1] is None:
                vals_c.append((np.zeros((1,) * lab.ndim, lab.dtype), None if q is None else
                               np.zeros((q.shape[0],) + (1,) * lab.ndim, q.dtype), np.zeros((1,) * lab.ndim, y.dtype), *rest))
            else:
                vals_c.append((lab[s], qs, y[s], *rest))
    else:
        vals_c = vals
    t = [(y == c) for _, _, y, *_ in vals_c]
    if not any(x.any() for x in t):
        return c, None, None, None
    gts = [_components(tt) if f1 else None for tt in t]
    base = _metric([_score(lab == c, tt, f1, g, sp=sp, pts=pp) for (lab, _, _, sp, pp), tt, g in zip(vals_c, t, gts)], f1)
    best, br = base, None
    taus = (None,) + (TAUS if vals_c[0][1] is not None else ())
    for tau in taus:   # the thresholded masks' components once per tau, the filters on top of them
        comps = [_components((q[c] >= round(tau * 255)) if tau is not None else lab == c) for lab, q, *_ in vals_c]
        for smin in sorted({0, *sizes.get(c, ())}) if f1 != "det" else (0,):   # det: the threshold only
            for largest in (False, True) if f1 != "det" else (False,):
                if tau is None and not smin and not largest:
                    continue
                sc = _metric([_score(_pp_class(lab, q, c, tau, smin, largest, cp), tt, f1, g,
                                     cp if f1 == "det" else None, sp, pp)
                              for (lab, q, _, sp, pp), tt, g, cp in zip(vals_c, t, gts, comps)], f1)
                if sc > best + 1e-4:
                    best, br = sc, (tau, int(smin), largest)
    return c, br, base, best


def choose_postproc(vals: list, k: int, sizes: dict, f1: bool, part: Path | None = None, pkey=None) -> dict:
    """Per foreground class, on the val volumes [(argmax, uint8 probs or None, labels)], the rule (tau, min size,
    largest only) that improves the task's metric (Dice, or lesion F1); no rule if nothing does (user, 23/10:
    nnU-Net-style post-processing chosen on val, the same for every task). (08/11) `part`: each class's choice
    written as it is made, a killed job resumes at the next class (ResEnc-L TotalSeg: 51 classes outlasted a
    30-min block after ~9 min of val loading)."""
    rule, done = {}, {}
    if part is not None and part.exists() and (pd := json.loads(part.read_text())).get("key") == pkey:
        done = {int(c): r for c, r in pd["done"]}
        print(f"[seg] post-processing search resumed: {len(done)} / {k - 1} classes done", flush=True)

    def mark(c, r) -> None:
        done[c] = None if r is None else list(r)
        if part is not None:
            part.with_suffix(".tmp").write_text(json.dumps({"key": pkey, "done": [[a, b] for a, b in done.items()]}))
            os.replace(part.with_suffix(".tmp"), part)
    boxes = None
    if f1 != "det" and vals and vals[0][1] is not None:   # (05/11) with probabilities (k <= 8: SOOP stroke, prostate):
        # each class on the box of its argmax, true and q >= min(TAUS) voxels, which holds every candidate mask and its
        # components: the same scores (outside it no mask or label of the class) on a box instead of 512 x 512 x 370
        # volumes (stroke: ~50 min for 100 val volumes, every candidate labelled on the whole grid)
        from scipy import ndimage
        tmin, boxes = round(min(TAUS) * 255), []
        for lab, q, y, *_ in vals:
            bx = []
            for c in range(1, k):
                fo = ndimage.find_objects(((lab == c) | (y == c) | (q[c] >= tmin)).astype(np.int8))
                bx.append(fo[0] if fo else None)
            boxes.append(bx)
    elif f1 != "det" and vals and vals[0][1] is None:   # argmax only (many classes): each class on the bounding box of its
        # predicted and true voxels, found for all classes in one pass per volume; same masks and scores, the components
        # and filters on the box only (01/11: TotalSeg's ~50 classes on whole-body volumes took hours)
        from scipy import ndimage
        boxes = []
        for lab, _, y, *_ in vals:
            a, b = ndimage.find_objects(lab.astype(np.int32), k - 1), ndimage.find_objects(y.astype(np.int32), k - 1)
            boxes.append([None if sa is None and sb is None else tuple(
                slice(min(x.start for x in (u, v) if x is not None), max(x.stop for x in (u, v) if x is not None))
                for u, v in zip(sa or (None,) * lab.ndim, sb or (None,) * lab.ndim)) for sa, sb in zip(a, b)])
    for c, r in done.items():
        if r is not None:
            rule[c] = tuple(r)
    # (08/11) the classes in parallel, one process each (fork: the val arrays shared, not copied; numpy / scipy only):
    # ~100 s per class on one core (ResEnc-L TotalSeg, 51 classes), the job holds 8
    global _PP
    _PP = (vals, boxes, sizes, f1)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [c for c in range(1, k) if c not in done]
    try:
        with ProcessPoolExecutor(max(1, min(len(todo), len(os.sched_getaffinity(0)))), mp_context=mp.get_context("fork")) as ex:
            for f in as_completed([ex.submit(_pp_one, c) for c in todo]):
                c, br, base, best = f.result()
                if br is not None:
                    rule[c] = br
                    print(f"[seg] post-processing class {c}: tau {br[0]}, min {br[1]} voxels, largest {br[2]}: val "
                          f"{base:.4f} -> {best:.4f}", flush=True)
                mark(c, br)
    finally:
        _PP = None
    return rule


def save_pred(d: Path, i: str, probs: torch.Tensor, grid_from_gt, lab=None) -> None:
    """Prediction of one volume on the model grid (user, 23/10: always kept, val and test, to choose thresholds and
    filters after the run): argmax, class probabilities as uint8 (p * 255; all classes up to 8, else the argmax's
    probability only) and the GT mapping (LFC landmark error, CMB lesion F1). Uncompressed (05/11: compressing
    ~500 MB SOOP grids took seconds per volume; scratch has room)."""
    d.mkdir(parents=True, exist_ok=True)
    # (07/11) the max over classes before the fp32 copy (the same uint8: rounding is monotonic): 51 classes on a
    # whole-body grid in fp32 asked 26 GB of the GPU (ResEnc-L TotalSeg, accumulator on the device since 44c68e5)
    p = probs if len(probs) <= 8 else probs.max(0).values[None]
    q = (p.float().clamp(0, 1) * 255).round().to(torch.uint8)
    np.savez(d / f"{i}.npz", pred=(probs.argmax(0).to(torch.uint8).cpu().numpy() if lab is None else lab),
                        prob=q.cpu().numpy(), grid_from_gt=np.asarray(grid_from_gt))


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
