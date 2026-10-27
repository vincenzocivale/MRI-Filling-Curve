"""Segmentation benchmark through each model's official downstream decoder on its frozen pretrained encoder
(protocol and wrapper hooks in fm/segrun.py, which runs inside the model's env).

Volumes per task: `select` (seed-fixed: at most 400 train / 100 val, every test volume), the same for every
model. Two stages, as SLURM jobs:
- prep (per preprocessing group, fm.groups in the bench config): the group's preprocessing + GT on the model grid,
  cached under <segdec.cache>/<group lead>/ for the union of the tasks' volumes;
- train (per model x task x init): `init` = pretrained (the protocol) or random (same network, encoder randomly
  initialised and frozen: what the decoder gets from the architecture alone). Writes
  <segdec.out>/<task>/s<seed>/<model>__<init>.json with a 95% CI over test-subject resamples (probe.subject_ci).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..fm.api import load_config, run_in_env
from ..io import override
from .probe import subject_ci, task_rows

MAX_TRAIN, MAX_VAL = 400, 100  # volumes per task held in RAM


def lead_of(fm_cfg: dict, model: str) -> str:
    return next(g["configs"][0] for g in fm_cfg["groups"].values() if model in g["configs"])


def task_items(tasks: dict, vols: pd.DataFrame, splits: pd.DataFrame, seed: int,
               sample: int | None = None) -> dict[str, list[dict]]:
    """{task: items}; `sample` keeps the first k volumes of each split (pilots)."""
    out = {}
    for name, cfg in tasks.items():
        rows = task_rows(vols, splits, cfg, seed)
        sel = {s: idx[:sample] for s, idx in select(rows, cfg, np.random.default_rng(seed)).items()}
        lm = {k: cfg[k] for k in ("classes", "ignore", "save_pred", "select") if cfg.get(k)}   # multi-label maps (21/10)
        out[name] = [{"id": r["id"], "image": r["input"], "seg": r["seg"], "split": s, "subject": r["subject"]} | lm
                     for s, idx in sel.items() for _, r in rows.loc[idx].iterrows()]
    return out


def run_job(cfg: dict, job: dict) -> None:
    run_in_env(cfg["env"], "sfc_gdn2.fm.segrun", {"cfg": cfg, **job})


def prep(bench: dict, group: str, items: dict[str, list[dict]], device: str = "cuda", shard: str | None = None,
         fm_set: list[str] | None = None) -> None:
    """`shard` = "i/n": every n-th volume from the i-th; `fm_set`: KEY=VALUE fields of the fm config."""
    g, s = bench["fm"]["groups"][group], bench["segdec"]
    lead = g["configs"][0]
    every = {it["id"]: it for its in items.values() for it in its}
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        every = dict(list(every.items())[i::n])
    run_job(override(load_config(f"configs/fm/{lead}.yaml"), fm_set),
            {"stage": "prep", "cache": str(Path(s["cache"]) / lead), "items": list(every.values()),
             "device": device, "workers": int(g["workers"])})


def train(bench: dict, model: str, task: str, items: list[dict], seed: int, init: str, device: str = "cuda",
          iters: int | None = None, resume: bool = False, val_only: bool = False,
          fm_set: list[str] | None = None) -> dict:
    """`resume`: a run whose json is written (same iters) is skipped, for chained jobs (a killed run always resumes
    from its checkpoints, fm/segrun.py). `val_only` (pretraining tuning): the best val Dice, no test volume read."""
    s = bench["segdec"] | ({"iters": iters, "val_every": min(int(bench["segdec"]["val_every"]), iters)} if iters else {})
    if s.get("screen"):   # exploratory: short, dead runs stopped, kept apart from the tables (bench.yaml)
        s |= {"val_every": 1250, "out": str(s["out"]).rstrip("/") + "_screen"}
    out = Path(s["out"]) / task / f"s{seed}" / f"{model}__{init}.json"
    done = "val_dice" if val_only else "test_ci95"
    if resume and out.exists() and done in (res := json.loads(out.read_text())) and res["iters"] == int(s["iters"]):
        print(f"[segdec] {task} s{seed} {model} {init}: done ({out}), skipped", flush=True)
        return res
    seg = Path(items[0]["seg"])
    k = 1 + (len(items[0]["classes"]) if items[0].get("classes") else
             len(list(seg.glob("*.nii*"))) if seg.is_dir() else 1)  # as geom.gt_labels counts classes
    run_job(override(load_config(f"configs/fm/{model}.yaml"), fm_set),
            {"stage": "train", "val_only": val_only, "cache": str(Path(s["cache"]) / lead_of(bench["fm"], model)), "items": items,
             "k": k, "init": init, "iters": int(s["iters"]), "val_every": int(s["val_every"]), "batch": int(s["batch"]),
             "lr": float(s["lr"]), "seed": seed, "device": device, "out": str(out),
             "save_pred": bool(items[0].get("save_pred")), "select": items[0].get("select"),
             "patience": 2 if s.get("screen") else 0})
    res = json.loads(out.read_text())
    if val_only:
        print(f"[segdec] {task} s{seed} {model} {init}: best val dice {res['val_dice']:.3f}", flush=True)
        return res
    subj = {it["id"]: it["subject"] for it in items}
    d = np.asarray(res["per_volume"], dtype=float)
    res |= {"task": task, "seed": seed,
            "test_ci95": subject_ci(lambda idx: np.nanmean(d[idx]), [subj[i] for i in res["test_ids"]])}
    out.write_text(json.dumps(res, indent=1))
    print(f"[segdec] {task} s{seed} {model} {init}: dice {res['test_dice']:.3f} {res['test_ci95']}", flush=True)
    return res


def select(rows: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Row indices per split: at most MAX_TRAIN / MAX_VAL random train / val volumes, every test volume (of
    `test_query`, if set)."""
    sel = {s: rows.index[rows["split"] == s].to_numpy() for s in ("train", "val", "test")}
    sel["train"] = rng.permutation(sel["train"])[:MAX_TRAIN]
    sel["val"] = rng.permutation(sel["val"])[:MAX_VAL]
    if cfg.get("test_query"):
        sel["test"] = rows.loc[sel["test"]].query(cfg["test_query"]).index.to_numpy()
    return sel
