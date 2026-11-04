"""Fine-tuned classification (04/11, P2): the task's volumes and labels as the probes' (task_rows, targets, split seed),
fm/clsrun.py in the model's env (prep, then train), then the probes' metrics and paired-bootstrap CI on test.

    python -m sfc_gdn2.bench.clsft --model sfc_gdn2 --task idh --out <dir> [--fm-set k=v ...] [--iters N ...]
-> <out>/<task>/<model>.json (clsrun's predictions) + <model>.metrics.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..fm.api import load_config, run_in_env
from ..io import load_yaml, override
from .probe import KINDS, bootstrap, targets, task_rows

MV = "/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2"
CACHE = "/leonardo_scratch/fast/IscrC_SFMRI/fcorrent/clsft_inputs"


def kind_metrics(cfg, y, rows, pred, yt, test_rows) -> dict:
    return KINDS[cfg["type"]](cfg, y[(rows["split"] == "train").to_numpy()]).metrics(pred, yt, test_rows)


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
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lr-warm", type=float, default=0.1, help="fraction of the iterations of linear lr warm-up")
    ap.add_argument("--l2", type=float, default=1.0, help="the starting head's L2, the same for every model (strong: "
                    "the encoder must adapt, 04/11); -1 = the table's chosen L2")
    ap.add_argument("--enc-lr", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing of the fine-tuning CE")
    ap.add_argument("--aug", type=int, default=1, help="training augmentation (flips, intensity scale): 1 on, 0 off")
    ap.add_argument("--light", action="store_true", help="ours: one curve (hilbert, ablation C) and 16 + 8 mm "
                    "(ablation R) instead of the table's reading; the frozen probe then refitted on the L2 grid")
    ap.add_argument("--warm", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=6, help="validations without a new best before stopping (0: off)")
    ap.add_argument("--subset", type=int, default=0, help="pilot: the first n train volumes, n // 2 val and test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=["all", "prep", "train"], default="all")
    ap.add_argument("--probe-json", required=True, help="the table's probe result of this model, task, split: its "
                    "head chosen on val (linear / mlp, the better val) gives feature, head and L2")
    a = ap.parse_args()

    cfg = load_yaml(a.tasks_cfg)[a.task]
    vols = pd.read_csv(Path(a.manifests) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(a.manifests) / "splits.csv")
    rows = task_rows(vols, splits, cfg, a.seed)
    rows = rows[rows["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    if a.subset:   # pilot (04/11): a few volumes per split, by sorted id
        lab = rows.eval(cfg["label"])   # both classes in every split
        rows = pd.concat([g.head(max(1, (a.subset if s == "train" else a.subset // 2) // 2))
                          for (s, _), g in rows.groupby(["split", lab])]).sort_values("id").reset_index(drop=True)
    y, classes = targets(rows, cfg)
    items = [{"id": r["id"], "image": r["input"], "split": r["split"], "y": int(v)}
             for (_, r), v in zip(rows.iterrows(), y.tolist())]
    if a.light:   # the reading's knobs reach the model's env through the environment (fm/sfc FINE, probe._grains)
        assert a.model.startswith("sfc_gdn2"), "--light: ours"
        os.environ |= {"SFC_FINE": "8", "SFC_READ_GRAINS": "16,8"}
        a.fm_set = [*a.fm_set, "args.curves=[hilbert]"]
    fm = override(load_config(f"configs/fm/{a.model}.yaml"), a.fm_set)
    out = Path(a.out) / a.task / f"{a.model}.json"
    cache = str(Path(CACHE) / f"{a.model}_t")   # (the same inputs for both readings: the extraction's dict)   # (04/11) the table reading's inputs (ours: the extraction's dict)
    print(f"[clsft] {a.model} {a.task}: {len(items)} volumes, classes {classes}", flush=True)
    if a.stage in ("all", "prep"):
        run_in_env(fm["env"], "sfc_gdn2.fm.clsrun", {"stage": "prep", "cfg": fm, "cache": cache, "items": items,
                                                     "device": "cuda", "workers": a.workers})
    if a.stage == "prep":
        return
    pj = json.loads(Path(a.probe_json).read_text())
    pj = pj.get(a.task, pj)
    var = max(("linear", "mlp"), key=lambda v: pj[v]["val"][pj["metric"]])
    table = {"head": var, "feature": pj[var]["feature"], "l2": pj[var]["l2"], "val": pj[var]["val"][pj["metric"]],
             "test": pj[var]["test"][pj["metric"]]}
    print(f"[clsft] the table's probe (split {a.seed}): {table}", flush=True)
    if not out.exists():
        run_in_env(fm["env"], "sfc_gdn2.fm.clsrun",
                   {"stage": "train", "cfg": fm, "cache": cache, "items": items, "device": "cuda", "out": str(out),
                    "iters": a.iters, "val_every": a.val_every, "batch": a.batch, "lr": a.lr, "enc_lr": a.enc_lr,
                    "warm": a.warm, "seed": a.seed, "feature": "mean" if a.light else table["feature"],
                    "head": "linear" if a.light or a.l2 >= 0 else table["head"],
                    "l2": a.l2 if a.l2 >= 0 else (None if a.light else table["l2"]), "smooth": a.smooth,
                    "lr_warm": a.lr_warm, "aug": a.aug} | ({"patience": a.patience} if a.patience else {}))
    res = json.loads(out.read_text())
    tp = res["test_pred"]
    test_rows = rows.set_index("id").loc[tp["id"]].reset_index()
    ymap = dict(zip(rows["id"], y.tolist()))
    yt = torch.tensor([ymap[i] for i in tp["id"]], dtype=torch.float32)
    kind = KINDS[cfg["type"]](cfg, y[(rows["split"] == "train").to_numpy()])
    pred = torch.tensor(np.asarray(tp["prob"]), dtype=torch.float32)
    fp = res["frozen_probe"]   # the table's head refitted here, encoder frozen: its test must match the table's
    pf = torch.tensor(np.asarray(fp["test_pred"]), dtype=torch.float32)
    m = {"task": a.task, "model": a.model, "val": res["val"], "best_iter": res["best_iter"], "table": table,
         "frozen_here": {"val": fp["val"], "test": kind_metrics(cfg, y, rows, pf, yt, test_rows)},
         "test": kind.metrics(pred, yt, test_rows), "test_ci95": bootstrap(kind, pred, yt, test_rows),
         "n_test": len(tp["id"]), "minutes": res["minutes"]}
    out.with_suffix(".metrics.json").write_text(json.dumps(m, indent=1))
    print(f"[clsft] frozen here: val {fp['val']:.4f} test {m['frozen_here']['test']} | table val {table['val']:.4f} "
          f"test {table['test']:.4f}", flush=True)
    print(f"[clsft] {a.model} {a.task}: val {m['val']:.4f}, test {m['test']} CI {m['test_ci95']}", flush=True)


if __name__ == "__main__":
    main()
