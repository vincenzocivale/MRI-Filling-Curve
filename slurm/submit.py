#!/usr/bin/env python3
"""Submit any command of this repo to SLURM from a YAML preset (CINECA Leonardo).

    python slurm/submit.py slurm/presets/<preset>.yaml [options] -- <command ...>

The batch script runs from the repo root with the conda env active, so `sfc` and repo-relative
paths work as interactively. `single` presets run `srun <command>`; `torchrun` presets run
`srun torchrun ... <command>`, and torchrun starts python itself, so pass `-m <module>` or a
script path there, never `python`:

    python slurm/submit.py slurm/presets/dbg.yaml -- sfc fm configs/fm/brainiac.yaml --images a.nii.gz --out o
    python slurm/submit.py slurm/presets/1gpu.yaml --job-name probe -- \\
        sfc bench-probe configs/bench/bench.yaml --models <model>
    python slurm/submit.py slurm/presets/4gpu.yaml -- -m sfc_gdn2.cli pretrain configs/pretrain_lejepa.yaml
    python slurm/submit.py slurm/presets/cpu.yaml -- sfc pretrain-pool configs/pretrain_lejepa.yaml --workers 8

Multi-node: `4gpu.yaml --set nodes=2` (2 nodes x 4 GPUs; torchrun --nnodes follows `nodes`).

Runs longer than the 24 h limit: `--chain N` submits N copies, each after the previous ends (afterany);
`sfc pretrain` resumes its run directory's ckpt.pt, so the copies continue one run:

    python slurm/submit.py slurm/presets/4gpu.yaml --chain 5 -- -m sfc_gdn2.cli pretrain configs/pretrain_lejepa.yaml

Config = slurm/defaults.yaml <- slurm/local.yaml (per user, gitignored) <- preset <- --set key=value.
Logs in outputs/slurm/.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SLURM_DIR = Path(__file__).resolve().parent
REPO = SLURM_DIR.parent


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_config(preset: Path, overrides: list[str]) -> dict[str, Any]:
    cfg = load_yaml(SLURM_DIR / "defaults.yaml")
    if (SLURM_DIR / "local.yaml").exists():
        cfg = merge(cfg, load_yaml(SLURM_DIR / "local.yaml"))
    cfg = merge(cfg, load_yaml(preset))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item!r}: use key=value (dots for nesting, e.g. launch.nnodes=2)")
        *parents, leaf = key.split(".")
        cursor = cfg
        for k in parents:
            cursor = cursor.setdefault(k, {})
        cursor[leaf] = value
    return cfg


def directive(key: str, value: Any) -> str | None:
    if value is None or value is False:
        return None
    flag = key.replace("_", "-")
    return f"#SBATCH --{flag}" if value is True else f"#SBATCH --{flag}={value}"


def build_script(cfg: dict[str, Any], command: list[str]) -> tuple[str, str]:
    logs_dir = cfg["logs"]["dir"].format(repo=REPO)
    launch = cfg.get("launch") or {}
    mail = cfg.get("mail") or {}
    head = [directive(k, cfg.get(k)) for k in
            ("partition", "account", "qos", "job_name", "nodes", "ntasks_per_node", "cpus_per_task", "gres",
             "mem", "time")]
    head += [directive("output", f"{logs_dir}/%x_%j.out"), directive("error", f"{logs_dir}/%x_%j.err"),
             directive("mail_type", mail.get("type") if mail.get("user") else None),
             directive("mail_user", mail.get("user"))]

    cmd = " ".join(shlex.quote(c) for c in command)
    if launch.get("type", "single") == "torchrun":
        run = ['HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)',
               (f"srun torchrun --nnodes={launch.get('nnodes', cfg.get('nodes', 1))} "
                f"--nproc_per_node={launch.get('nproc_per_node', 1)} --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d "
                f"--rdzv_endpoint=$HEAD_NODE:{launch.get('rdzv_port', 29500)} {cmd}")]
    elif launch.get("type", "single") == "single":
        run = [f"srun {cmd}"]
    else:
        raise SystemExit(f"launch.type must be single | torchrun, got {launch['type']!r}")

    body = [f"cd {shlex.quote(str(REPO))}",
            *(f"module load {m}" for m in cfg.get("modules") or []),
            f"source {shlex.quote(str(cfg['conda']['base']) + '/etc/profile.d/conda.sh')}",
            f"conda activate {shlex.quote(str(cfg['conda']['env']))}",
            *(f"export {k}={shlex.quote(str(v))}" for k, v in (cfg.get("env") or {}).items()),
            'echo "[sfc] $(date) job $SLURM_JOB_ID on $SLURM_NODELIST, commit $(git rev-parse --short HEAD)"',
            *run]
    return "\n".join(["#!/bin/bash", *filter(None, head), "set -eo pipefail", "", *body]) + "\n", logs_dir


def parse_args(argv: list[str]) -> argparse.Namespace:
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", type=Path, help="slurm/presets/<name>.yaml")
    ap.add_argument("--job-name", help="override the job name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config field")
    ap.add_argument("--after-job", metavar="JOB_ID", help="start after JOB_ID ends (--dependency=afterany)")
    ap.add_argument("--chain", type=int, default=1, metavar="N", help="submit N copies, each after the previous")
    ap.add_argument("--dry-run", action="store_true", help="print the batch script, do not submit")
    args = ap.parse_args(argv[:sep])
    args.command = argv[sep + 1:]
    if not args.command:
        raise SystemExit("Missing command after '--'; see `python slurm/submit.py -h`.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    preset = args.preset if args.preset.exists() else REPO / args.preset
    if not preset.exists():
        raise SystemExit(f"Preset not found: {args.preset}")
    cfg = load_config(preset, args.set)
    if args.job_name:
        cfg["job_name"] = args.job_name
    script, logs_dir = build_script(cfg, args.command)
    if args.dry_run:
        print(script)
        return 0
    if not shutil.which("sbatch"):
        raise SystemExit("sbatch not found (not on a SLURM login node?)")
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    after, jobs = args.after_job, []
    for _ in range(args.chain):
        cmd = ["sbatch", "--parsable", *([f"--dependency=afterany:{after}"] if after else [])]
        out = subprocess.run(cmd, input=script, check=True, capture_output=True, text=True).stdout
        after = out.strip().split(";")[0]
        jobs.append(after)
    print(f"Submitted batch job {' -> '.join(jobs)} (logs: {logs_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
