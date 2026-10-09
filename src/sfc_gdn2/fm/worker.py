"""Runs inside a model's conda env: `python -m sfc_gdn2.fm.worker <job.json>`.

job = {"cfgs": [{...resolved model config...}, ...], "outs": [dir, ...],
       "items": [{"id": str, "image": path | {modality: path}, "dense": bool}],
       "device": "cuda" | "cpu", "overwrite": bool, "workers": int, "verify_shared": bool}
(`cfg` / `out` instead of `cfgs` / `outs`: one model.) The models of a job share ONE preprocessing, the first
model's `preprocess` (only group models whose repos preprocess identically; `verify_shared` re-runs every
model's own preprocessing and logs any difference). With `workers` > 0 the preprocessing runs ahead in that
many CPU-only processes (CUDA hidden, 1 thread each), each with its own copy of the first wrapper built on cpu,
while this process runs the networks (and `preprocess_gpu`, for wrappers that split their preprocessing into
`preprocess_cpu` + `preprocess_gpu`). Writes `<out>/<id>.pt` (the wrapper's result dict, tensors on CPU);
a failing item is logged to `<out>/_failed/<id>.txt` and skipped. Only stdlib + torch are imported here, so it
runs in any model env without our own dependencies.

Optional `cfg["save"]` trims what is stored, never what is computed: `features` (names kept for every item;
default all), `dense` (more names kept for items with `dense`), `pool` ([{from, factor, to}]: average pooling,
ceil mode, added to `derived`), `drop_canonical` (do not force-keep the canonical tensor), `dtype`.
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import importlib
import itertools
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from . import MODELS

ONE_THREAD = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")


def build(cfg: dict, device: str):
    return importlib.import_module(f"sfc_gdn2.fm.{MODELS[cfg['model']]}").build(cfg, device)


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def trim(res: dict, save: dict | None, dense: bool = False) -> dict:
    if not save:
        return res
    keep = save.get("features")
    if keep is not None:
        keep = set(keep) | (set(save.get("dense") or ()) if dense else set())
    feats, derived = dict(res["features"]), list(res.get("derived", []))
    for op in save.get("pool") or ():
        if keep is None or op["to"] in keep:
            k = [op["factor"]] * 3 if isinstance(op["factor"], int) else list(op["factor"])
            feats[op["to"]] = F.avg_pool3d(feats[op["from"]][None].float(), k, k, ceil_mode=True)[0]
            derived.append(op["to"])
    force = None if save.get("drop_canonical") else res["canonical"]
    feats = {k: v for k, v in feats.items() if keep is None or k in keep or k == force}
    if save.get("dtype"):
        dtype = getattr(torch, save["dtype"])
        feats = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in feats.items()}
    return res | {"features": feats, "derived": derived, "saved": dict(save, dense=dense)}


def differences(a, b, path="") -> list[str]:
    """Paths where two preprocessing outputs differ (exact equality for arrays / tensors)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in a.keys() | b.keys() for d in differences(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    if hasattr(a, "shape") and hasattr(b, "shape"):
        same = tuple(a.shape) == tuple(b.shape) and bool((torch.as_tensor(a) == torch.as_tensor(b)).all())
        return [] if same else [path]
    try:
        return [] if bool(a == b) else [path]
    except Exception:  # noqa: BLE001 -- objects without a usable ==
        return [] if type(a) is type(b) else [path]


_PREP = None


def _init(cfg: dict) -> None:
    global _PREP
    torch.set_num_threads(1)
    try:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except ImportError:
        pass
    _PREP = build(cfg, "cpu")


def _prep(image) -> bytes:
    t0 = time.time()
    x = getattr(_PREP, "preprocess_cpu", _PREP.preprocess)(image)  # CPU part only, if the wrapper splits it
    return pickle.dumps((x, time.time() - t0), protocol=pickle.HIGHEST_PROTOCOL)


def prepared(lead, cfg: dict, items: list, workers: int):
    """Yield (item, preprocessed | None, error | None, seconds), in item order."""
    if workers <= 0:
        for it in items:
            t0 = time.time()
            try:
                yield it, lead.preprocess(it["image"]), None, time.time() - t0
            except Exception:  # noqa: BLE001 -- logged per item, the shard goes on
                yield it, None, traceback.format_exc(), time.time() - t0
        return
    env = dict(os.environ)
    os.environ.update({**dict.fromkeys(ONE_THREAD, "1"), "CUDA_VISIBLE_DEVICES": ""})  # inherited at spawn
    try:
        ex = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(cfg,))
        todo = iter(items)
        window = [(it, ex.submit(_prep, it["image"])) for it in itertools.islice(todo, 2 * workers)]  # spawns
    finally:
        os.environ.clear()
        os.environ.update(env)
    # ponytail: a crashed CPU worker (OOM kill) breaks the pool and ends the job; resubmitting resumes (skip-existing)
    with ex:
        while window:
            it, fut = window.pop(0)
            try:
                x, dt = pickle.loads(fut.result())
                yield it, x, None, dt
            except cf.process.BrokenProcessPool:
                raise
            except Exception:  # noqa: BLE001 -- logged per item, the shard goes on
                yield it, None, traceback.format_exc(), float("nan")
            window += [(n, ex.submit(_prep, n["image"])) for n in itertools.islice(todo, 1)]


def fail(out: Path, item_id: str, err: str) -> None:
    (out / "_failed").mkdir(parents=True, exist_ok=True)
    (out / "_failed" / f"{item_id}.txt").write_text(err)
    print(f"[fm] FAILED {out.name} {item_id}: {err.strip().splitlines()[-1]}", flush=True)


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    cfgs = job.get("cfgs") or [job["cfg"]]
    outs = [Path(o) for o in (job.get("outs") or [job["out"]])]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    wrappers = [build(c, job["device"]) for c in cfgs]
    lead = wrappers[0]
    items = [it for it in job["items"]
             if job.get("overwrite") or not all((o / f"{it['id']}.pt").exists() for o in outs)]
    print(f"[fm] {[c['model'] for c in cfgs]}: {len(items)} of {len(job['items'])} items to do, "
          f"{job.get('workers', 0)} preprocessing workers", flush=True)
    done = 0
    workers = int(job.get("workers", 0))
    for it, x, err, t_pre in prepared(lead, cfgs[0], items, workers):
        if workers > 0 and not err and hasattr(lead, "preprocess_gpu"):
            try:
                t0 = time.time()
                x = lead.preprocess_gpu(x)
                t_pre += time.time() - t0
            except Exception:  # noqa: BLE001 -- logged per item, the shard goes on
                err = traceback.format_exc()
        if err:
            for o in outs:
                fail(o, it["id"], err)
            continue
        for i, (w, cfg, out) in enumerate(zip(wrappers, cfgs, outs)):
            dst = out / f"{it['id']}.pt"
            if dst.exists() and not job.get("overwrite"):
                continue
            try:
                if job.get("verify_shared") and w is not lead:
                    diff = differences(x, w.preprocess(it["image"]))
                    print(f"[fm] verify_shared {cfg['model']} {it['id']}: "
                          f"{'identical' if not diff else 'DIFFERENT at ' + ', '.join(diff)}", flush=True)
                t0 = time.time()
                res = w.finish(x if i == len(wrappers) - 1 else copy.deepcopy(x))
                t_feat = time.time() - t0
                res = to_cpu(trim(res, cfg.get("save"), bool(it.get("dense"))))
                res |= {"id": it["id"], "image": it["image"], "dense": bool(it.get("dense"))}
                tmp = dst.with_suffix(".tmp")
                torch.save(res, tmp)
                os.replace(tmp, dst)
                (out / "_failed" / f"{it['id']}.txt").unlink(missing_ok=True)
                done += 1
                shapes = {k: tuple(v.shape) for k, v in res["features"].items() if isinstance(v, torch.Tensor)}
                print(f"[fm] {cfg['model']} {it['id']}: {shapes} canonical={res['canonical']} pre={t_pre:.1f}s "
                      f"net={t_feat:.1f}s {dst.stat().st_size / 2**20:.1f}MB -> {dst}", flush=True)
            except Exception:  # noqa: BLE001 -- logged per item, the shard goes on
                fail(out, it["id"], traceback.format_exc())
    print(f"[fm] done: {done} files written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
