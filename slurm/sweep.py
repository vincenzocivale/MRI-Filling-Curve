#!/usr/bin/env python3
"""Pretraining screens (notes/PIANO_TARATURA.md): `python slurm/sweep.py <spec.yaml> [--dry-run]`.

Configs: `runs` (a list of explicit configs, e.g. the arms of an ablation) or `n` random draws from `space` (with
`seed`: a spec always gives the same configs); each is `fixed` + its own fields. Space entries: {log: [lo, hi]} |
{uniform: [lo, hi]} | {choice: [...]}; with `scales: {key: base}` one draw multiplies every base (several parameters on
one axis). One SLURM job per config (preset 1gpu + the `slurm` --set fields, e.g. gres: "gpu:2"), so the evaluations
wait in no second queue: <run>/job.sh pretrains (torchrun on the job's GPUs, output_root <root>/<name>/<i>), then
- `tasks` (non-empty): skill (bench-extract --tune + bench-probe --tune -> <run>/tune/tune.csv),
- `dense: {task, iters}`: segdec pilot, best val Dice (bench-segdec --val-only -> <run>/segdec/),
the two side by side on GPUs 0 and 1 when the job has two. <root>/<name>/sweep.csv lists configs and jobs.
(boost_qos_dbg takes at most 2 submitted jobs per user: no place for evaluations of a whole sweep.)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def draw(space: dict, rng: random.Random) -> dict:
    out = {}
    for key, s in space.items():
        if "choice" in s:
            v = rng.choice(s["choice"])
        else:
            lo, hi = s.get("log") or s["uniform"]
            v = float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))) if 'log' in s else rng.uniform(lo, hi):.3g}")
        out |= {k: float(f"{b * v:.3g}") for k, b in s["scales"].items()} if "scales" in s else {key: v}
    return out


def job_script(spec: dict, cfg: dict, run: Path, gpus: int) -> str:
    over = " ".join(shlex.quote(f"{k}={json.dumps(v)}") for k, v in cfg.items())
    evals, fm = [], f"--fm-set checkpoint={run}"
    if spec.get("tasks"):
        evals.append(f"sfc bench-extract configs/bench/bench.yaml --group sfc_tune --tune {fm} --out {run}/features"
                     f" && sfc bench-probe configs/bench/bench.yaml --models sfc_gdn2 --tune"
                     f" --tasks {' '.join(spec['tasks'])} --out {run}/tune --override fm.out_root={run}/features")
    if d := spec.get("dense"):
        evals.append(f"sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2 --tasks {d['task']}"
                     f" --iters {d['iters']} --val-only {fm} --override segdec.out={run}/segdec")
    lines = ["#!/bin/bash", f"# {run.name} of sweep {spec['name']}: pretraining, then its evaluations", "set -eo pipefail",
             (f"torchrun --standalone --nproc_per_node={gpus} -m sfc_gdn2.cli pretrain {spec['config']} --override {over}"
              f" output_root={run}")]
    if gpus > 1:
        lines += [f"CUDA_VISIBLE_DEVICES={g} bash -c {shlex.quote(e)} & p{g}=$!" for g, e in enumerate(evals)]
        lines += [f"wait $p{g}" for g in range(len(evals))]
    else:
        lines += evals
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the configs and the first job script, submit nothing")
    a = ap.parse_args()
    spec = yaml.safe_load(a.spec.read_text())
    rng, root = random.Random(spec.get("seed", 0)), Path(spec["root"]) / spec["name"]
    slurm = spec.get("slurm") or {}
    gpus = int(str(slurm.get("gres", "gpu:1")).split(":")[-1])
    sets = [x for k, v in slurm.items() for x in ("--set", f"{k}={v}")]
    configs = spec.get("runs") or [draw(spec.get("space") or {}, rng) for _ in range(spec["n"])]
    rows = []
    for i, own in enumerate(configs):
        cfg, run = (spec.get("fixed") or {}) | own, root / f"{i:02d}"
        script, row = job_script(spec, cfg, run, gpus), {"i": i, **cfg}
        if a.dry_run:
            print(script if i == 0 else "", end="")
        else:
            run.mkdir(parents=True, exist_ok=True)
            (run / "job.sh").write_text(script)
            cmd = [sys.executable, str(REPO / "slurm" / "submit.py"), "slurm/presets/1gpu.yaml", "--job-name",
                   f"{spec['name']}_{i:02d}", *sets, "--", "bash", str(run / "job.sh")]
            out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO).stdout
            row["job"] = out.split("Submitted batch job ")[1].split()[0]
        rows.append(row)
        print(row, flush=True)
    if rows and not a.dry_run:
        with open(root / "sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
