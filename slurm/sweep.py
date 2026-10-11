#!/usr/bin/env python3
"""Pretraining screens (notes/PIANO_TARATURA.md): `python slurm/sweep.py <spec.yaml> [--dry-run]`.

Configs: `runs` (a list of explicit configs, e.g. the arms of an ablation) or `n` random draws from `space` (with
`seed`: a spec always gives the same configs); each is `fixed` + its own fields. Space entries: {log: [lo, hi]} |
{uniform: [lo, hi]} | {choice: [...]}; with `scales: {key: base}` one draw multiplies every base (several parameters on
one axis). Each config is a pretraining job (`preset`, default 4gpu, with the `slurm` --set fields; output_root
<root>/<name>/<i>) and, after it, its evaluations, which fail if the run wrote no model.pt (did not finish):
- `tasks` (non-empty): skill on boost_qos_dbg (bench-extract --tune + bench-probe --tune -> <run>/tune/tune.csv);
- `dense: {task, iters}`: segdec pilot, best val Dice (bench-segdec --val-only -> <run>/segdec/).
<root>/<name>/sweep.csv lists configs and jobs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


def submit(preset: str, name: str, command: list[str], *extra: str) -> str:
    cmd = [sys.executable, str(REPO / "slurm" / "submit.py"), preset, "--job-name", name, *extra, "--", *command]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO).stdout
    return out.split("Submitted batch job ")[1].split()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the configs, submit nothing")
    a = ap.parse_args()
    spec = yaml.safe_load(a.spec.read_text())
    rng, root = random.Random(spec.get("seed", 0)), Path(spec["root"]) / spec["name"]
    sets = [x for k, v in (spec.get("slurm") or {}).items() for x in ("--set", f"{k}={v}")]
    configs = spec.get("runs") or [draw(spec.get("space") or {}, rng) for _ in range(spec["n"])]
    rows = []
    for i, own in enumerate(configs):
        cfg, run, name = (spec.get("fixed") or {}) | own, root / f"{i:02d}", f"{spec['name']}_{i:02d}"
        row = {"i": i, **cfg}
        if not a.dry_run:
            over = [f"{k}={json.dumps(v)}" for k, v in cfg.items()] + [f"output_root={run}"]
            pre = row["pretrain_job"] = submit(spec.get("preset", "slurm/presets/4gpu.yaml"), name,
                                               ["-m", "sfc_gdn2.cli", "pretrain", spec["config"], "--override", *over],
                                               *sets)
            done, fm = f"ls {run}/*/model.pt", f"--fm-set checkpoint={run}"
            if spec.get("tasks"):
                tune = (f"{done} && sfc bench-extract configs/bench/bench.yaml --group sfc_tune --tune {fm}"
                        f" --out {run}/features && sfc bench-probe configs/bench/bench.yaml --models sfc_gdn2 --tune"
                        f" --tasks {' '.join(spec['tasks'])} --out {run}/tune --override fm.out_root={run}/features")
                row["tune_job"] = submit("slurm/presets/dbg.yaml", f"{name}_tune", ["bash", "-c", tune],
                                         "--after-job", pre)
            if d := spec.get("dense"):
                seg = (f"{done} && sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2"
                       f" --tasks {d['task']} --iters {d['iters']} --val-only {fm} --override segdec.out={run}/segdec")
                row["dense_job"] = submit("slurm/presets/1gpu.yaml", f"{name}_dense", ["bash", "-c", seg],
                                          "--after-job", pre, "--set", f"time={d.get('time', '02:00:00')}")
        rows.append(row)
        print(row, flush=True)
    if rows and not a.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "sweep.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
