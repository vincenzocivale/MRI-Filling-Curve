"""Runs inside a model's conda env: `python -m sfc_gdn2.fm.worker <job.json>`.

job = {"cfg": {...resolved model config...}, "items": [{"id": str, "image": path | {modality: path}}],
       "out": dir, "device": "cuda" | "cpu", "overwrite": bool}
Writes `<out>/<id>.pt` (the wrapper's result dict, tensors on CPU). Only stdlib + torch are imported
here, so it runs in any model env without our own dependencies.

Optional `cfg["save"]` = {"features": [names] (default: all; the canonical one is always kept),
"dtype": "float16" (default: as computed)} only trims what is stored, never what is computed.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import torch

from . import MODELS


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def trim(res: dict, save: dict | None) -> dict:
    if not save:
        return res
    keep = save.get("features")
    feats = {k: v for k, v in res["features"].items() if keep is None or k in keep or k == res["canonical"]}
    if save.get("dtype"):
        dtype = getattr(torch, save["dtype"])
        feats = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in feats.items()}
    return res | {"features": feats, "saved": save}


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    cfg, out = job["cfg"], Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    wrapper = importlib.import_module(f"sfc_gdn2.fm.{MODELS[cfg['model']]}").build(cfg, job["device"])
    for item in job["items"]:
        dst = out / f"{item['id']}.pt"
        if dst.exists() and not job.get("overwrite"):
            print(f"[fm] {cfg['model']} {item['id']}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        res = trim(to_cpu(wrapper(item["image"])), cfg.get("save")) | {"id": item["id"], "image": item["image"]}
        torch.save(res, dst)
        shapes = {k: tuple(v.shape) for k, v in res["features"].items() if isinstance(v, torch.Tensor)}
        print(f"[fm] {cfg['model']} {item['id']}: {shapes} canonical={res['canonical']} "
              f"({time.time() - t0:.1f}s) -> {dst}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
