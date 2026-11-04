"""Fine-tuned classification (04/11, P2): the task's volumes and labels as the probes' (task_rows, targets, split seed),
fm/clsrun.py in the model's env (prep, then train), then the probes' metrics and paired-bootstrap CI on test.

    python -m sfc_gdn2.bench.clsft --model sfc_gdn2 --task idh --out <dir> [--fm-set k=v ...] [--iters N ...]
-> <out>/<task>/<model>.json (clsrun's predictions) + <model>.metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..fm.api import load_config, run_in_env
from ..io import load_yaml, override
from .probe import KINDS, bootstrap, targets, task_rows

MV = "/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2"
CACHE = "/leonardo_scratch/fast/IscrC_SFMRI/fcorrent/clsft_inputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fm-set", nargs="*", default=[])
    ap.add_argument("--tasks-cfg", default="configs/bench/probe_tasks_v2.yaml")
    ap.add_argument("--manifests", default=MV)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc-lr", type=float, default=0.1)
    ap.add_argument("--warm", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=0, help="validations without a new best before stopping (0: off)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["all", "prep", "train"], default="all")
    a = ap.parse_args()

    cfg = load_yaml(a.tasks_cfg)[a.task]
    vols = pd.read_csv(Path(a.manifests) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(a.manifests) / "splits.csv")
    rows = task_rows(vols, splits, cfg, a.seed)
    rows = rows[rows["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    y, classes = targets(rows, cfg)
    items = [{"id": r["id"], "image": r["input"], "split": r["split"], "y": int(v)}
             for (_, r), v in zip(rows.iterrows(), y.tolist())]
    fm = override(load_config(f"configs/fm/{a.model}.yaml"), a.fm_set)
    out = Path(a.out) / a.task / f"{a.model}.json"
    cache = str(Path(CACHE) / a.model)
    print(f"[clsft] {a.model} {a.task}: {len(items)} volumes, classes {classes}", flush=True)
    if a.stage in ("all", "prep"):
        run_in_env(fm["env"], "sfc_gdn2.fm.clsrun", {"stage": "prep", "cfg": fm, "cache": cache, "items": items,
                                                     "device": "cuda", "workers": a.workers})
    if a.stage == "prep":
        return
    if not out.exists():
        run_in_env(fm["env"], "sfc_gdn2.fm.clsrun",
                   {"stage": "train", "cfg": fm, "cache": cache, "items": items, "device": "cuda", "out": str(out),
                    "iters": a.iters, "val_every": a.val_every, "batch": a.batch, "lr": a.lr, "enc_lr": a.enc_lr,
                    "warm": a.warm, "seed": a.seed} | ({"patience": a.patience} if a.patience else {}))
    res = json.loads(out.read_text())
    tp = res["test_pred"]
    test_rows = rows.set_index("id").loc[tp["id"]].reset_index()
    ymap = dict(zip(rows["id"], y.tolist()))
    yt = torch.tensor([ymap[i] for i in tp["id"]], dtype=torch.float32)
    kind = KINDS[cfg["type"]](cfg, y[(rows["split"] == "train").to_numpy()])
    pred = torch.tensor(np.asarray(tp["prob"]), dtype=torch.float32)
    m = {"task": a.task, "model": a.model, "val": res["val"], "best_iter": res["best_iter"],
         "test": kind.metrics(pred, yt, test_rows), "test_ci95": bootstrap(kind, pred, yt, test_rows),
         "n_test": len(tp["id"]), "minutes": res["minutes"]}
    out.with_suffix(".metrics.json").write_text(json.dumps(m, indent=1))
    print(f"[clsft] {a.model} {a.task}: val {m['val']:.4f}, test {m['test']} CI {m['test_ci95']}", flush=True)


if __name__ == "__main__":
    main()
