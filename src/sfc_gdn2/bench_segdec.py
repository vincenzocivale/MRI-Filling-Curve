"""Segmentation benchmark through each model's official downstream decoder on its frozen pretrained encoder
(protocol and wrapper hooks in fm/segrun.py, which runs inside the model's env).

Volumes per task: bench_seg.select (seed-fixed: at most 400 train / 100 val, every test volume), the same for every
model. Two stages, as SLURM jobs:
- prep (per preprocessing group, fm.groups in the bench config): the group's preprocessing + GT on the model grid,
  cached under <segdec.cache>/<group lead>/ for the union of the tasks' volumes;
- train (per model x task x init): `init` = pretrained (the protocol) or random (same network, encoder randomly
  initialised and frozen: what the decoder gets from the architecture alone). Writes
  <segdec.out>/<task>/s<seed>/<model>__<init>.json with a 95% CI over test-subject resamples (bench_seg._ci).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import bench_seg
from .bench_probe import task_rows
from .fm.api import SRC, load_config


def lead_of(fm_cfg: dict, model: str) -> str:
    return next(g["configs"][0] for g in fm_cfg["groups"].values() if model in g["configs"])


def task_items(tasks: dict, vols: pd.DataFrame, splits: pd.DataFrame, seed: int,
               sample: int | None = None) -> dict[str, list[dict]]:
    """{task: items}; `sample` keeps the first k volumes of each split (pilots)."""
    out = {}
    for name, cfg in tasks.items():
        rows = task_rows(vols, splits, cfg, seed)
        sel = {s: idx[:sample] for s, idx in bench_seg.select(rows, cfg, np.random.default_rng(seed)).items()}
        out[name] = [{"id": r["id"], "image": r["input"], "seg": r["seg"], "split": s, "subject": r["subject"]}
                     for s, idx in sel.items() for _, r in rows.loc[idx].iterrows()]
    return out


def run_job(cfg: dict, job: dict) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"cfg": cfg, **job}, f)
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]),
                        "PYTHONNOUSERSITE": "1"}
    try:
        subprocess.run([str(Path(cfg["env"]) / "bin" / "python"), "-m", "sfc_gdn2.fm.segrun", f.name], env=env,
                       check=True)
    finally:
        os.unlink(f.name)


def prep(bench: dict, group: str, items: dict[str, list[dict]], device: str = "cuda", shard: str | None = None) -> None:
    """`shard` = "i/n": every n-th volume from the i-th."""
    g, s = bench["fm"]["groups"][group], bench["segdec"]
    lead = g["configs"][0]
    every = {it["id"]: it for its in items.values() for it in its}
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        every = dict(list(every.items())[i::n])
    run_job(load_config(f"configs/fm/{lead}.yaml"),
            {"stage": "prep", "cache": str(Path(s["cache"]) / lead), "items": list(every.values()),
             "device": device, "workers": int(g["workers"])})


def train(bench: dict, model: str, task: str, items: list[dict], seed: int, init: str, device: str = "cuda",
          iters: int | None = None, resume: bool = False) -> dict:
    """`resume`: a run whose json is written (same iters) is skipped, for chained jobs (a killed run always resumes
    from its checkpoints, fm/segrun.py)."""
    s = bench["segdec"] | ({"iters": iters, "val_every": max(iters // 2, 1)} if iters else {})
    out = Path(s["out"]) / task / f"s{seed}" / f"{model}__{init}.json"
    if resume and out.exists() and "test_ci95" in (res := json.loads(out.read_text())) and res["iters"] == int(s["iters"]):
        print(f"[segdec] {task} s{seed} {model} {init}: done ({out}), skipped", flush=True)
        return res
    seg = Path(items[0]["seg"])
    k = 1 + (len(list(seg.glob("*.nii*"))) if seg.is_dir() else 1)  # as bench_geom.gt_labels counts classes
    run_job(load_config(f"configs/fm/{model}.yaml"),
            {"stage": "train", "cache": str(Path(s["cache"]) / lead_of(bench["fm"], model)), "items": items,
             "k": k, "init": init, "iters": int(s["iters"]), "val_every": int(s["val_every"]), "batch": int(s["batch"]),
             "lr": float(s["lr"]), "seed": seed, "device": device, "out": str(out)})
    res = json.loads(out.read_text())
    subj = {it["id"]: it["subject"] for it in items}
    res |= {"task": task, "seed": seed, "test_ci95": bench_seg._ci(np.asarray(res["per_volume"], dtype=float),
                                                                  np.array([subj[i] for i in res["test_ids"]]))}
    out.write_text(json.dumps(res, indent=1))
    print(f"[segdec] {task} s{seed} {model} {init}: dice {res['test_dice']:.3f} {res['test_ci95']}", flush=True)
    return res
