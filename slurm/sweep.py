#!/usr/bin/env python3
"""Random search over pretraining configs (notes/IPERPARAMETRI.md): `python slurm/sweep.py <spec.yaml> [--dry-run]`.

Each of the spec's `n` configs (drawn with `seed`: a spec always gives the same configs) is a pretraining job
(preset 4gpu, `--override` of `fixed` + the draw, output_root <root>/<name>/<i>) and, after it, a tuning job (preset
1gpu: `bench-extract --tune` of the run's last checkpoint, `bench-probe --tune` on `tasks` -> <run>/tune/tune.csv;
it fails if the run has no model.pt, i.e. did not finish). Space entries: {log: [lo, hi]} | {uniform: [lo, hi]} |
{choice: [...]}; with `scales: {key: base}` one draw multiplies every base (several parameters on one axis).
`slurm`: --set fields of the pretraining jobs (e.g. time). <root>/<name>/sweep.csv lists configs and jobs.
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
    rng, root = random.Random(spec["seed"]), Path(spec["root"]) / spec["name"]
    sets = [x for k, v in (spec.get("slurm") or {}).items() for x in ("--set", f"{k}={v}")]
    rows = []
    for i in range(spec["n"]):
        cfg, run, name = (spec.get("fixed") or {}) | draw(spec.get("space") or {}, rng), root / f"{i:02d}", f"{spec['name']}_{i:02d}"
        row = {"i": i, **cfg}
        if not a.dry_run:
            over = [f"{k}={json.dumps(v)}" for k, v in cfg.items()] + [f"output_root={run}"]
            row["pretrain_job"] = submit("slurm/presets/4gpu.yaml", name,
                                         ["-m", "sfc_gdn2.cli", "pretrain", spec["config"], "--override", *over], *sets)
            tune = (f"ls {run}/*/model.pt"
                    f" && sfc bench-extract configs/bench/bench.yaml --group sfc_tune --tune --checkpoint {run}"
                    f" --out {run}/features"
                    f" && sfc bench-probe configs/bench/bench.yaml --models sfc_gdn2 --tune --tasks {' '.join(spec['tasks'])}"
                    f" --out {run}/tune --override fm.out_root={run}/features")
            row["tune_job"] = submit("slurm/presets/1gpu.yaml", f"{name}_tune", ["bash", "-c", tune],
                                     "--after-job", row["pretrain_job"], "--set", "time=03:00:00")
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
