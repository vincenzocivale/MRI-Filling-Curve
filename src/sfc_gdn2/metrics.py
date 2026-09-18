from __future__ import annotations

import numpy as np
import torch

from .curves import ordered_coords


def geometry_metrics(curve: str, grid: int, windows: list[int], radius: float, seed: int) -> dict:
    c = ordered_coords(curve, grid, seed).astype(np.float32)
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    # inverse: canonical raster point -> sequence position
    raster = np.array([(x,y,z) for x in range(grid) for y in range(grid) for z in range(grid)])
    pos = {tuple(v.astype(int)): i for i, v in enumerate(c)}
    recalls = {}
    offsets = np.array([(dx,dy,dz) for dx in range(-2,3) for dy in range(-2,3) for dz in range(-2,3)
                        if 0 < np.sqrt(dx*dx+dy*dy+dz*dz) <= radius])
    for w in windows:
        hit = total = 0
        for p in raster:
            i = pos[tuple(p)]
            for o in offsets:
                q = p + o
                if np.all((q >= 0) & (q < grid)):
                    total += 1; hit += abs(i - pos[tuple(q)]) <= w
        recalls[f"neighbor_recall_w{w}"] = hit / max(total, 1)
    return {
        "curve": curve,
        "step_mean": float(steps.mean()),
        "step_p95": float(np.quantile(steps, .95)),
        "step_max": float(steps.max()),
        **recalls,
    }


def masked_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float,float]:
    e = (pred - target)[mask]
    return float((e*e).mean().item()), float(e.abs().mean().item())


def copy_baseline_errors(x: torch.Tensor, perm: torch.Tensor, k: int = 1) -> tuple[float, float]:
    """Zero-order-hold baseline for next_patch: predict patch i+k (curve order) as patch i.
    Isolates how much of a curve's k-step MSE is explained by raw local
    redundancy in that ordering, vs. anything the model actually learned."""
    ordered = x[:, perm]
    pred, target = ordered[:, :-k], ordered[:, k:]
    e = pred - target
    return float((e * e).mean().item()), float(e.abs().mean().item())


def auc(history: list[dict], key: str = "val_mse") -> float:
    ys = np.array([r[key] for r in history if key in r], dtype=float)
    if len(ys) < 2: return float(ys[0]) if len(ys) else float("nan")
    return float(np.trapz(ys, dx=1) / (len(ys)-1))
