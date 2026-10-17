#!/usr/bin/env python3
"""Pretraining screens (notes/PIANO_TARATURA.md): `python slurm/sweep.py <spec.yaml> [--dry-run]`.

Configs: `runs` (a list of explicit configs, e.g. the arms of an ablation) or `n` random draws from `space` (with
`seed`: a spec always gives the same configs); each is `fixed` + its own fields. Space entries: {log: [lo, hi]} |
{uniform: [lo, hi]} | {choice: [...]}; with `scales: {key: base}` one draw multiplies every base (several parameters on
one axis). One SLURM job per config (preset 1gpu + the `slurm` --set fields, e.g. gres: "gpu:2"), so the evaluations
wait in no second queue: <run>/job.sh pretrains (torchrun on the job's GPUs, output_root <root>/<name>/<i>), then
- `tasks` (non-empty): skill (bench-extract --tune, split over the GPUs, + bench-probe --tune -> <run>/tune/tune.csv),
- `dense: {task, iters}`: segdec pilot, best val Dice (bench-segdec --val-only -> <run>/segdec/),
the probe and the pilot side by side on GPUs 0 and 1 when the job has two. <root>/<name>/sweep.csv lists configs and jobs;
`--collect` adds what the finished runs wrote (last SSL loss, skill per task and mean, val Dice): results.csv.
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


SEGDEC = (("segdec", ""), ("segdec_taps3_refine_image", "args.taps=3 args.refine=true args.image=true"),
          ("segdec_taps3_refine2_image", "args.taps=3 args.refine=2 args.image=true"))
# the segdec pilots of every run (user, 17/10: D2 and little else): Primus' decoder on the last block (the plain
# reference), and ours: + 3 intermediate blocks, 16 -> 8 mm along the Hilbert curve with the voxels' KernelEmbed at
# 8 mm (p7: 0.657 vs 0.649 for Primus' decoder on the same blocks), and the same down to 4 mm (p7: 0.648 at 2500
# iterations, still rising, lowest loss: the user expects it to win with more). Dropped on p7 (17/10): + 2 blocks,
# neighbour mixing, the patch embedding, the tile's mean token, reversed readings.


def job_script(spec: dict, cfg: dict, run: Path, gpus: int) -> str:
    """Pretraining on all the job's GPUs, then its `evaluation`."""
    over = " ".join(shlex.quote(f"{k}={json.dumps(v)}") for k, v in cfg.items())
    lines = ["#!/bin/bash", f"# {run.name} of sweep {spec['name']}: pretraining, then its evaluation", "set -eo pipefail",
             (f"torchrun --standalone --nproc_per_node={gpus} -m sfc_gdn2.cli pretrain {spec['config']} --override {over}"
              f" output_root={run}")]
    return "\n".join(lines + evaluation(spec, run, run, gpus)) + "\n"


def evaluation(spec: dict, ckpt: Path, run: Path, gpus: int) -> list[str]:
    """The one evaluation of a checkpoint (user, 16/10: always the full reading, chosen on val), outputs in `run`: the
    tuning extraction split over the GPUs (--shard); on the CPU the probe's readings (`probe.readings`: means of the
    last and intermediate blocks, stds, covariance square roots, the 2x-reading means) while the GPUs share the SEGDEC
    pilots; then the probe over every reading on all GPUs. `collect` keeps the best val Dice."""
    fm, lines = f"--fm-set checkpoint={ckpt}", []
    if spec.get("tasks"):
        lines += [f"CUDA_VISIBLE_DEVICES={g} sfc bench-extract configs/bench/bench.yaml --group sfc_tune --tune {fm}"
                  f" --shard {g}/{gpus} --out {run}/features & x{g}=$!" for g in range(gpus)]
        lines += [f"wait $x{g}" for g in range(gpus)]
        code = f"from pathlib import Path; from sfc_gdn2.bench.probe import readings; readings('sfc_gdn2', Path('{run}/features/sfc_gdn2'))"
        lines.append(f"python -c {shlex.quote(code)} & c=$!")
    if d := spec.get("dense"):
        seg = [f"sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2 --tasks {d['task']}"
               f" --iters {d['iters']} --val-only {fm} {a} --override segdec.out={run}/{out}" for out, a in SEGDEC]
        lines += [f"CUDA_VISIBLE_DEVICES={g} bash -c {shlex.quote(' ; '.join(seg[g::gpus]))} & s{g}=$!" for g in range(gpus)]
        lines += [f"wait $s{g} || true" for g in range(gpus)]  # a failed pilot must not skip the probe
    if spec.get("tasks"):
        lines += ["wait $c || true", (f"sfc bench-probe configs/bench/bench.yaml --models sfc_gdn2 --tune --tasks"
                                      f" {' '.join(spec['tasks'])} --out {run}/tune --override fm.out_root={run}/features")]
    return lines


def collect(root: Path, sub: str = "") -> None:
    """<root>/results.csv: sweep.csv + per run the last logged SSL loss, the skill per task and its mean, the val Dice
    (`sub`: of the evaluation in <run>/<sub>, written to <root>/results_<sub>.csv)."""
    import pandas as pd
    rows = pd.read_csv(root / "sweep.csv").to_dict("records")
    for r in rows:
        run = root / f"{r['i']:02d}" / sub
        if h := sorted(run.glob("*/history.csv")):
            r |= pd.read_csv(h[0]).iloc[-1][["step", "loss"]].to_dict()
        if (t := run / "tune" / "tune.csv").exists():
            skill = pd.read_csv(t).set_index("task")["skill"]
            r |= {"skill": skill.mean(), "tasks": len(skill), **skill.add_prefix("skill_").to_dict()}
        dice = {p.parts[-4]: json.loads(p.read_text())["val_dice"] for p in run.glob("segdec*/*/*/*.json")}
        if dice:  # the best decoder input on val
            r |= {"val_dice": max(dice.values()), "decoder": max(dice, key=dice.get)}
    df = pd.DataFrame(rows)
    df.to_csv(root / f"results{'_' + sub if sub else ''}.csv", index=False)
    show = [c for c in df if df[c].astype(str).nunique(dropna=False) > 1 or c in ("i", "loss", "skill", "tasks", "val_dice")]
    print(df[[c for c in show if c != "job" and not c.startswith("skill_")]].to_string(index=False, float_format="%.4g"))


def reevaluate(spec: dict, root: Path, name: str) -> None:
    """One job per run of the sweep: `evaluation` of its checkpoint (the run dir: its last), outputs in <run>/<name>."""
    slurm = (spec.get("slurm") or {}) | {"time": "08:00:00"}
    gpus = int(str(slurm.get("gres", "gpu:1")).split(":")[-1])
    for i in range(len(spec.get("runs") or range(spec.get("n", 0)))):
        run = root / f"{i:02d}"
        out = run / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "job.sh").write_text("\n".join(["#!/bin/bash", f"# {name}: evaluation of {run}", "set -eo pipefail",
                                               *evaluation(spec, run, out, gpus)]) + "\n")
        cmd = [sys.executable, str(REPO / "slurm" / "submit.py"), "slurm/presets/1gpu.yaml", "--job-name",
               f"{spec['name']}_{i:02d}_{name}", *[x for k, v in slurm.items() for x in ("--set", f"{k}={v}")], "--",
               "bash", str(out / "job.sh")]
        print(run, subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO).stdout.strip(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the configs and the first job script, submit nothing")
    ap.add_argument("--collect", action="store_true", help="gather the finished runs' results into results.csv")
    ap.add_argument("--after-job", metavar="JOB_ID", help="every run starts after JOB_ID ends (submit.py --after-job)")
    ap.add_argument("--eval", metavar="NAME", help="evaluate the sweep's finished runs again (evaluation of now), outputs "
                                                    "in <run>/NAME; with --collect: gather those")
    a = ap.parse_args()
    spec = yaml.safe_load(a.spec.read_text())
    rng, root = random.Random(spec.get("seed", 0)), Path(spec["root"]) / spec["name"]
    if a.collect:
        return collect(root, a.eval or "")
    if a.eval:
        return reevaluate(spec, root, a.eval)
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
                   f"{spec['name']}_{i:02d}", *sets, *(["--after-job", a.after_job] if a.after_job else []), "--", "bash",
                   str(run / "job.sh")]
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
