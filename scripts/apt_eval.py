"""Tuning probe of an APT extraction (apt.py, `args.apt`) against the uniform readings of the same checkpoint.

    python scripts/apt_eval.py <apt feature dir> <out.csv> [--uniform 16=<dir> 12=<dir>] [--tasks age idh ...]

Same protocol as the multiscale probes of notes/LOG.md (18/10): volumes of the subjects in no test split
(`bp.tune_subjects`: the test split is never read), per task `bp.tune_task` chooses head and readout on val among a
method's two readings, `mean` and `blocks+std`, and reports the val skill. A method is an APT arm (volume-weighted
`mean`, or `mean_tok`, one weight per leaf) or a uniform reading. Prints skill per task and, per arm, the leaves per
volume and the encoder seconds."""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import torch

from sfc_gdn2.bench import probe as bp
from sfc_gdn2.io import load_yaml

OUT = "/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench"
TASKS = ("age", "sex", "idh", "grade", "survival", "mrs90")


def unique(f: dict) -> dict:
    """bi = h ++ h for one-stream encoders: keep one half."""
    out = {}
    for k, v in f.items():
        w = v.shape[1] // 2
        out[k] = v if k.startswith(("ntok", "time")) or not torch.equal(v[:, :w], v[:, w:]) else v[:, :w]
    return out


def readings(f: dict, ids: list, common: list, mean: str, std: str, tail: str = "") -> dict:
    """{mean, blocks+std} of the volumes `common` from a feature dict whose keys are `<mean>[_b<k>]<tail>`."""
    ix = [ids.index(i) for i in common]

    def take(k):
        return f[k][ix].float()

    blocks = sorted(int(m.group(1)) for k in f if (m := re.fullmatch(rf"{mean}_b(\d+){re.escape(tail)}", k)))
    cat = [*(take(f"{mean}_b{b}{tail}") for b in blocks), take(mean + tail), *(take(f"{std}_b{b}{tail}") for b in blocks), take(std + tail)]
    return {"mean": take(mean + tail), "blocks+std": torch.cat(cat, 1)}


def methods(apt_dir: str, uniform: dict) -> tuple[list, dict, pd.DataFrame]:
    ids, f = bp.load_globals(Path(apt_dir))
    f = unique(f)
    glob = {}
    for u, d in uniform.items():
        i, g = bp.load_globals(Path(d))
        glob[u] = (i, unique(g))
    common = sorted(set(ids).intersection(*(set(g[0]) for g in glob.values())))
    arms = sorted({k.split("__")[1] for k in f if "__" in k})
    out, stats = {}, []
    for u, (i, g) in glob.items():
        out[f"uniform {u} mm"] = readings(g, i, common, "mean", "std")
    for a in arms:
        for suffix, label in (("", "vol"), ("_tok", "tok")):
            out[f"{a} ({label})"] = readings(f, ids, common, f"mean{suffix}", f"std{suffix}", f"__{a}")
        ix = [ids.index(i) for i in common]
        stats.append({"arm": a, "leaves/volume": f[f"ntok__{a}"][ix].float().sum(1).mean().item(),
                      "encoder s/volume (median)": f[f"time__{a}"][ix].float().median().item()})
    return common, out, pd.DataFrame(stats)


def one(task: str, tcfg: dict, rows: pd.DataFrame, apt_dir: str, uniform: dict, method: str, device: str) -> dict:
    torch.set_num_threads(2)
    common, m, _ = methods(apt_dir, uniform)
    feats = {f"{r}@{method}": v for r, v in m[method].items()}
    res = bp.tune_task(task, tcfg, rows, feats, {i: k for k, i in enumerate(common)}, device)
    return {"method": method, "task": task, "skill": res["skill"], "chosen": res["chosen"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("apt_dir")
    ap.add_argument("out")
    ap.add_argument("--uniform", nargs="*", default=[], metavar="MM=DIR")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args()
    uniform = dict(u.split("=", 1) for u in a.uniform)
    cfg = load_yaml("configs/bench/bench.yaml")
    tasks = load_yaml(cfg["probe"]["tasks"])
    vols, splits = pd.read_csv(OUT + "/volumes.csv", low_memory=False), pd.read_csv(OUT + "/splits.csv")
    common, m, stats = methods(a.apt_dir, uniform)
    print(f"{len(common)} volumes in common\n{stats.to_string(index=False)}", flush=True)
    v = vols[vols["id"].isin(set(common)) & vols["subject"].isin(bp.tune_subjects(splits))]
    jobs = [(t, tasks[t], bp.task_rows(v, splits, tasks[t], 0), n) for t in a.tasks for n in m]
    res = []
    with ProcessPoolExecutor(max(1, a.gpus), mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(one, t, c, r, a.apt_dir, uniform, n, f"cuda:{k % max(1, a.gpus)}")
                for k, (t, c, r, n) in enumerate(jobs)]
        for fut in as_completed(futs):
            try:
                res.append(fut.result())
                print({k: res[-1][k] for k in ("method", "task", "skill")}, flush=True)
            except Exception as e:   # noqa: BLE001 (one failed fit must not lose the others)
                print("failed", repr(e), flush=True)
            pd.DataFrame(res).to_csv(a.out, index=False)
    t = pd.DataFrame(res).pivot(index="method", columns="task", values="skill")
    t["mean"] = t.mean(1)
    print(t.round(3).to_string())
