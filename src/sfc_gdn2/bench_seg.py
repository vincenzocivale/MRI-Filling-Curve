"""Segmentation probes on the stored dense maps, the same for every model (protocol as in bench_probe.py).

Input per volume: the model's maps whose stride is at most twice its finest (bench_geom.STRIDES), resized to the
finest grid and concatenated (DINOv2 "+ms"). Target per cell: the fraction of each class (0 = background) among the
GT voxels falling in it (GT -> world -> source voxel -> model input grid, bench_geom). Two heads, trained the same
way: `linear` (1x1x1 conv) and `conv` (3x3x3 conv to 128, GELU, 3x3x3 conv). AdamW, soft cross-entropy with
class-balanced weights, one volume per step, EPOCHS epochs, grid lr x weight decay; the config and epoch are
selected on val (soft Dice at cell level). Test: class probabilities interpolated to every GT voxel, argmax, Dice per
foreground class at full resolution averaged per volume, mean over volumes with a 95% CI over test-subject resamples;
`oracle` = the same for the GT cell fractions (the ceiling of the model's grid).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import bench_geom as geom

EPOCHS = 20
GRID = ((1e-3, 1e-4), (1e-3, 1e-2), (3e-4, 1e-4), (3e-4, 1e-2))
MAX_TRAIN, MAX_VAL = 400, 100  # ponytail: volumes held in RAM; a streaming loader lifts the cap


def gt_labels(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(labels 0..K, affine, class names): a mask file (any > 0 = 1) or a directory of one mask per class."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.nii*"))
        ref = nib.load(files[0])
        lab = np.zeros(ref.shape, dtype=np.uint8)
        for k, f in enumerate(files, 1):
            lab[np.asarray(nib.load(f).dataobj) > 0] = k
        return lab, ref.affine, [f.name.split(".")[0] for f in files]
    img = nib.load(p)
    return (np.asarray(img.dataobj) > 0).astype(np.uint8), img.affine, ["lesion"]


def model_input(model: str, rec: dict) -> tuple[torch.Tensor, object]:
    """([C, *grid] fp16, stride of the grid): the dense maps of a stored volume, finest grid."""
    strides = geom.STRIDES[geom.FAMILY[model]]
    maps = {n: (rec["features"][n], s) for n, s in strides.items() if n in rec["features"]}
    finest = min(max(np.atleast_1d(s)) for _, s in maps.values())
    maps = {n: (m, s) for n, (m, s) in maps.items() if max(np.atleast_1d(s)) <= 2 * finest}
    base, stride = min(maps.values(), key=lambda ms: max(np.atleast_1d(ms[1])))
    grid = tuple(base.shape[1:])
    # ponytail: coarser maps resized to the finest grid (align_corners=False), exact only when the extents match
    xs = [m.float() if tuple(m.shape[1:]) == grid else
          F.interpolate(m[None].float(), size=grid, mode="trilinear", align_corners=False)[0] for m, _ in maps.values()]
    return torch.cat(xs).half(), stride


def gt_coords(model: str, meta: dict, input_path: str, gt_affine: np.ndarray, gt_shape, stride) -> torch.Tensor:
    m = geom.source_to_input(model, meta) @ np.linalg.inv(nib.load(input_path).affine) @ gt_affine
    return geom.cell_coords(m, stride, gt_shape)


def head(kind: str, c: int, k: int) -> nn.Module:
    if kind == "linear":
        return nn.Conv3d(c, k, 1)
    return nn.Sequential(nn.Conv3d(c, 128, 3, padding=1), nn.GELU(), nn.Conv3d(128, k, 3, padding=1))


def soft_dice(prob: torch.Tensor, t: torch.Tensor) -> float:
    ok = torch.isfinite(t[0])
    p, t = prob[:, ok], t[:, ok]
    present = t[1:].sum(1) > 0
    if not present.any():
        return float("nan")
    d = 2 * (p[1:] * t[1:]).sum(1) / (p[1:].sum(1) + t[1:].sum(1)).clamp_min(1e-6)
    return d[present].mean().item()


def full_res_dice(prob: torch.Tensor, coords: torch.Tensor, lab: np.ndarray) -> float:
    """Dice per foreground class present in GT or prediction, at GT resolution, averaged."""
    pred = geom.from_cells(prob, coords).argmax(0).cpu().numpy()
    ds = [2 * ((pred == c) & (lab == c)).sum() / ((pred == c).sum() + (lab == c).sum())
          for c in range(1, prob.shape[0]) if (lab == c).any() or (pred == c).any()]
    return float(np.mean(ds)) if ds else float("nan")


def run_seg(name: str, cfg: dict, rows: pd.DataFrame, model: str, model_dir: Path, seed: int,
            device: str = "cuda") -> dict:
    rng = np.random.default_rng(seed)
    sel = {s: rows.index[rows["split"] == s].to_numpy() for s in ("train", "val", "test")}
    sel["train"] = rng.permutation(sel["train"])[:MAX_TRAIN]
    sel["val"] = rng.permutation(sel["val"])[:MAX_VAL]
    if cfg.get("test_query"):
        sel["test"] = rows.loc[sel["test"]].query(cfg["test_query"]).index.to_numpy()

    def load(i, keep_coords=False):
        r = rows.loc[i]
        rec = torch.load(model_dir / f"{r['id']}.pt", map_location="cpu", weights_only=False)
        x, stride = model_input(model, rec)
        lab, aff, names = gt_labels(r["seg"])
        c = gt_coords(model, rec["meta"], r["input"], aff, lab.shape, stride)
        t = geom.cell_fractions(torch.from_numpy(lab), c, tuple(x.shape[1:]), len(names))
        return (x, t, names) + ((c, lab) if keep_coords else ())

    data = {s: [load(i)[:2] for i in sel[s]] for s in ("train", "val")}
    k1 = data["train"][0][1].shape[0]
    s1 = sum(x.flatten(1).float().sum(1) for x, _ in data["train"])
    s2 = sum(x.flatten(1).float().square().sum(1) for x, _ in data["train"])
    n = sum(x[0].numel() for x, _ in data["train"])
    mu = (s1 / n)[:, None]
    sd = (s2 / n - mu[:, 0].square()).clamp_min(1e-12).sqrt()[:, None]
    tsum = torch.stack([torch.nan_to_num(t).flatten(1).sum(1) for _, t in data["train"]]).sum(0)
    weight = (tsum.sum() / (k1 * tsum.clamp_min(1))).to(device)
    norm = lambda x: ((x.float().flatten(1) - mu) / sd).view(x.shape).to(device)

    out = {"task": name, "model": model, "seed": seed, "n": {s: len(v) for s, v in sel.items()}}
    for kind in ("linear", "conv"):
        best = None
        for lr, wd in GRID:
            torch.manual_seed(seed)
            h = head(kind, mu.shape[0], k1).to(device)
            opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
            for ep in range(EPOCHS):
                h.train()
                for j in rng.permutation(len(data["train"])):
                    x, t = data["train"][j]
                    t = t.to(device)
                    ok = torch.isfinite(t[0])
                    logp = h(norm(x)[None])[0].log_softmax(0)
                    loss = -(weight[:, None] * torch.nan_to_num(t[:, ok]) * logp[:, ok]).sum(0).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                h.eval()
                with torch.no_grad():
                    val = np.nanmean([soft_dice(h(norm(x)[None])[0].softmax(0), t.to(device)) for x, t in data["val"]])
                if best is None or val > best["val"]:
                    best = {"val": float(val), "lr": lr, "wd": wd, "epoch": ep + 1, "state": copy.deepcopy(h.state_dict())}
        h.load_state_dict(best.pop("state"))
        out[kind] = best | {"head": h}
    dice = {"linear": [], "conv": [], "oracle": []}
    with torch.no_grad():
        for i in sel["test"]:
            x, t, _, c, lab = load(i, keep_coords=True)
            for kind in ("linear", "conv"):
                dice[kind].append(full_res_dice(out[kind]["head"](norm(x)[None])[0].softmax(0), c.to(device), lab))
            dice["oracle"].append(full_res_dice(torch.nan_to_num(t).to(device), c.to(device), lab))
    subj = rows.loc[sel["test"], "subject"].to_numpy()
    for kind, d in dice.items():
        d = np.asarray(d)
        res = {"test_dice": float(np.nanmean(d)), "test_ci95": _ci(d, subj), "per_volume": d.tolist()}
        out[kind] = (out.get(kind, {}) | res) if kind != "oracle" else res
        out[kind].pop("head", None)
    out["test_ids"] = rows.loc[sel["test"], "id"].tolist()
    return out


def _ci(d: np.ndarray, subj: np.ndarray, n: int = 1000) -> list[float]:
    rng = np.random.default_rng(0)
    groups = pd.Series(range(len(subj))).groupby(subj).indices
    keys = sorted(groups)
    vals = [np.nanmean(d[np.concatenate([groups[s] for s in rng.choice(keys, len(keys))])]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def write(res: dict, out_dir: Path) -> None:
    dst = out_dir / res["task"] / f"s{res['seed']}" / f"{res['model']}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=1))
