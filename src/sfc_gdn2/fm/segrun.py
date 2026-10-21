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
AHEAD = 6   # batches whose cached tokens are read in advance
TILE_MAX = 1e9   # voxels x classes of one tile
JOINT_MAX = 2e8   # voxels x classes of a batch that joint_step keeps in graphs (~5 GB of fp32 logits and softmax)
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
            lab, gt_aff, names = gt_labels(it["seg"])
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
def sliding(net, x: torch.Tensor, patch, n_out: int, amp, ctx=None, align=None) -> torch.Tensor:
    """[C, *S] -> class probabilities [n_out, *S] (Gaussian-weighted tiles, step 0.5 of the patch). align: tile corners
    on that grid (token cache), the volume padded so that the last tile ends on it."""
    if getattr(net, "matryoshka", False) and ctx is not None:   # the whole scan in one pass (no tiles)
        with amp():
            return net.matryoshka_predict(x, ctx.to(x.device), n_out)
    if getattr(net, "adaptive", False) and ctx is not None:
        with amp():
            return net.adaptive_predict(x, ctx.to(x.device), n_out)
    S = tuple(x.shape[1:])
    patch = tile_of(patch, align, n_out)
    patch = S if patch is None else tuple(patch)
    if align is not None:
        tgt = [p + -(-max(s - p, 0) // a) * a for s, p, a in zip(S, patch, align)]
        xp, _ = pad_to(x, None, tgt)
    else:
        xp, _ = pad_to(x, None, patch)
    P = tuple(xp.shape[1:])
    starts = [np.unique(np.round(np.linspace(0, s - p, math.ceil((s - p) / (p / 2)) + 1)).astype(int)) if s > p else [0]
              for s, p in zip(P, patch)]
    if align is not None:
        starts = [np.unique(np.append(np.asarray(st) // a * a, s - p)) for st, a, s, p in zip(starts, align, P, patch)]
    g = gaussian(patch, x.device)
    # fp16 accumulator on the device if it takes <= 1/4 of the free memory, else on the CPU (nnU-Net's predictor does
    # both; "fits at all" left the tiles' forward no room: nnF CNN OOM on whole-body TotalSeg grids)
    fits = x.is_cuda and n_out * math.prod(P) * 2 <= torch.cuda.mem_get_info(x.device)[0] / 4
    out = torch.zeros(n_out, *P, dtype=torch.half, device=x.device if fits else "cpu")
    wsum = torch.zeros(P, device=x.device)
    corners = list(product(*starts))
    # token-cache decoders take a batch of tiles at once (their refine levels pack the tiles' sequences): up to 8, at
    # most ~JOINT_MAX voxels x classes; the others one tile at a time, as before
    nb = 1 if align is None else max(1, min(8, int(JOINT_MAX // (n_out * math.prod(patch)))))
    for j in range(0, len(corners), nb):
        cs = corners[j: j + nb]
        sls = [(slice(a, a + patch[0]), slice(b, b + patch[1]), slice(c, c + patch[2])) for a, b, c in cs]
        with amp():
            cx = None if ctx is None else torch.stack([ctx if align is None else
                                                       torch.cat([ctx, torch.tensor(co, dtype=ctx.dtype)]) for co in cs])
            logits = call(net, torch.stack([xp[(slice(None), *sl)] for sl in sls]), cx)
        for lg, sl in zip(logits, sls):
            out[(slice(None), *sl)] += (lg.float().softmax(0) * g).to(out)
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
        tp += torch.bincount(t[pred == t], minlength=k)[:k].float()
        np_ += torch.bincount(pred, minlength=k)[:k].float()
        nt += torch.bincount(t, minlength=k)[:k].float()
    per = [(2 * tp[c] / (np_[c] + nt[c])).item() if (np_[c] + nt[c]) > 0 else float("nan") for c in range(1, k)]
    return (float(np.nanmean(per)) if not all(math.isnan(d) for d in per) else float("nan")), per


class Patches:
    """Random training patches from volumes held in RAM; 1/3 centred on a voxel of a random present class."""

    def __init__(self, vols: list[dict], patch, k: int, rng: np.random.Generator):
        self.vols, self.ctx, self.patch, self.rng, self.align = [], [], [], rng, []
        for v in vols:
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
            ys.append(y[sl])
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
    """The gradient of loss_fn on the whole batch, one scan at a time (wrappers with `seg_microbatch`, many classes:
    decoder F on TotalSeg did not fit with 2 tiles). The soft Dice sums over the batch before dividing, so a first
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


def joint_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """micro_step's loss and gradient with every scan's graph kept: one forward per scan and one backward (tiles of
    several shapes that fit together: the token-cache runs' 16 cm tiles; micro_step's no-grad pass doubled the work)."""
    parts = list(zip(x, y, c if c is not None else [None] * len(x))) if isinstance(x, list) else \
        [(x[i:i + 1], y[i:i + 1], None if c is None else c[i:i + 1]) for i in range(len(x))]
    sums = []
    for xi, yi, ci in parts:
        with amp():
            logits = call(net, xi.to(dev).float(), ci)
        sums.append(batch_sums(logits, yi.to(dev), k))
    n = sum(s[1] for s in sums)
    inter, psum, tsum = (sum(s[j] for s in sums) for j in (2, 3, 4))
    dice = (2 * inter[1:] + 1e-5) / (psum[1:] + tsum[1:] + 1e-5)
    loss = sum(s[0] for s in sums) / n + 1 - dice.mean()
    loss.backward()
    return loss.detach()


def sums_loss(t: list) -> torch.Tensor:
    """SegNet.adaptive_sums' sums -> CE per voxel + 1 - mean soft Dice of the classes > 0 (loss_fn's)."""
    ce, n, inter, psum, tsum = t
    return ce / n.clamp_min(1) + 1 - ((2 * inter[1:] + 1e-5) / (psum[1:] + tsum[1:] + 1e-5)).mean()


def matryoshka_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """Decoder F matryoshka (21/10): the final prediction's loss plus each cell head's (16, 8, 4 mm), one backward."""
    parts = [(t, u, v) for t, u, v in zip(x, y, c)] if isinstance(x, list) else [(x, y, c)]
    fin, aux = None, None
    for xi, yi, ci in parts:
        with amp():
            f, a = net.matryoshka_sums(xi.to(dev).float(), yi.to(dev), ci.to(dev), k)
        fin = f if fin is None else [u + v for u, v in zip(fin, f)]
        aux = a if aux is None else [[u + v for u, v in zip(p, q)] for p, q in zip(aux, a)] + (aux[len(a):] or a[len(aux):])
    loss = sums_loss(fin) + sum(sums_loss(a) for a in aux)
    loss.backward()
    return loss.detach()


def adaptive_step(net, x, y, c, k: int, amp, dev) -> torch.Tensor:
    """Decoder F adaptive (21/10): the batch's sums (final prediction, cell head) in one pass (a list of tiles of
    several shapes: one pass each), one backward of both losses."""
    parts = [(t, u, v) for t, u, v in zip(x, y, c)] if isinstance(x, list) else [(x, y, c)]
    fin = aux = None
    for xi, yi, ci in parts:
        with amp():
            f, a = net.adaptive_sums(xi.to(dev).float(), yi.to(dev), ci.to(dev), k)
        fin = f if fin is None else [u + v for u, v in zip(fin, f)]
        aux = a if aux is None else [u + v for u, v in zip(aux, a)]
    loss = sums_loss(fin) + sums_loss(aux)
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
        return {"x": r["x"], "y": r["y"], "grid_from_gt": r["grid_from_gt"], "id": it["id"], "ctx": ctx,
                "align": None if tcache is None else [4 * a for a in tcache.meta(it["id"])["k4"]]}

    split = {s: [it for it in job["items"] if it["split"] == s and (cache / f"{it['id']}.pt").exists()]
             for s in ("train", "val", "test")}
    # on disk as the run goes, so a killed job resumes: the training state every 10 min (.last.pt), the best state
    # until the test is written (.best.pt), the test volumes done (.test.json)
    out_p = Path(job["out"])
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
        missing, unexpected = net.load_state_dict(state, strict=False)
        assert not unexpected and set(missing) <= frozen_names, (unexpected, missing)

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
        queue = [draw() for _ in range(min(AHEAD, iters - start + 1))]
        for it in range(start, iters + 1):
            x, y, c = queue.pop(0)
            if it + len(queue) < iters:
                queue.append(draw())
            opt.zero_grad(set_to_none=True)
            # cm tiles of a few voxels: both graphs fit, one backward (20/10); TotalSeg's 0.3 mm scans make 16 cm tiles
            # of ~1e7 voxels x 50 classes: then micro_step (the two graphs of joint_step went OOM)
            vox = sum(t.numel() for t in x) if isinstance(x, list) else x.numel()
            if getattr(net, "matryoshka", False):
                loss = matryoshka_step(net, x, y, c, k, amp, dev)
                logits = torch.zeros(1, device=dev)
            elif getattr(net, "adaptive", False):
                loss = adaptive_step(net, x, y, c, k, amp, dev)
                logits = torch.zeros(1, device=dev)
            elif tcache is not None and (isinstance(x, list) or k > 10) and vox * k <= JOINT_MAX:
                loss = joint_step(net, x, y, c, k, amp, dev)
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
            if it % every == 0 or it == iters:
                net.eval()
                ds = [dice_on(sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"], v["align"]).argmax(0).to(dev),
                              v["y"].to(dev), k) for v in val_v]
                val = float(np.nanmean(ds))
                curve.append({"iter": it, "loss": float(loss), "val_dice": val, "min": (time.time() - t0) / 60})
                print(f"[seg] it {it} loss {float(loss):.3f} val dice {val:.4f} ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
                if "state" not in best or val > best["val"]:
                    best = {"val": val, "iter": it, "state": trained()}
                set_mode()
            if time.time() - t_save > SAVE_S and it < iters:
                save_last(it)
                t_save = time.time()
        del val_v, data
        torch.save({"key": key, "best": best, "curve": curve}, ckpt)
        last.unlink(missing_ok=True)
    if job.get("val_only"):  # pretraining tuning: the best val Dice, no test volume read
        best.pop("state")
        out_p.write_text(json.dumps({"model": job["cfg"]["model"], "init": job["init"], "iters": iters,
                                     "best": best, "curve": curve, "val_dice": best["val"],
                                     "n": {s: len(v) for s, v in split.items() if s != "test"}}, indent=1))
        ckpt.unlink()
        print(f"[seg] best val dice {best['val']:.4f} (iter {best['iter']}) -> {job['out']}", flush=True)
        return
    load_trained(best.pop("state"))
    net.eval()
    tkey = key | {"best_iter": best["iter"]}
    done = json.loads(part.read_text()) if part.exists() else {}
    res = done["res"] if done.get("key") == tkey else {"per_volume": [], "per_class": [], "test_ids": []}
    if res["test_ids"]:
        print(f"[seg] test resumed from {part}: {len(res['test_ids'])} volumes done", flush=True)
    for it in split["test"]:
        if it["id"] in res["test_ids"]:
            continue
        t1 = time.time()
        v = load(it)
        probs = sliding(net, v["x"].to(dev).float(), patch, k, amp, v["ctx"], v["align"])
        lab, _, _ = gt_labels(it["seg"])
        d, per = dice_full_res(probs, v["grid_from_gt"], lab, k, dev)
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
           "test_dice": float(np.nanmean(res["per_volume"])), **res}
    out_p.write_text(json.dumps(out, indent=1))
    ckpt.unlink()
    part.unlink(missing_ok=True)
    print(f"[seg] test dice {out['test_dice']:.4f} on {len(res['per_volume'])} volumes -> {job['out']}", flush=True)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    {"prep": prep, "train": train}[job["stage"]](job)


if __name__ == "__main__":
    main(sys.argv[1])
