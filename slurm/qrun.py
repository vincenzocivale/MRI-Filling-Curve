#!/usr/bin/env python3
"""Run a short analysis on a compute node without queueing a new job (CINECA Leonardo).

    python slurm/qrun.py [--mem 16] [--cpus 4] [--gpu] [--min-left 15] -- <command ...>

The command runs as an extra step (`srun --overlap`) inside one of your RUNNING booster jobs: no queue, no extra
hours (the job's cores are paid anyway), output and exit code straight to the terminal, from the current directory
with the env of slurm/defaults.yaml. lrd_all_serial jobs never host (slow login-like nodes) and the host's GPU
stays to the host (--gres=none). OOM safety: the step gets its own
cgroup limit (--mem GB), so going over it kills only the step, and a job hosts it only if its free memory
(limit - rss - other limited steps' reservations - max(8G, 15%) margin) covers --mem.

If no running job fits (or with --gpu) it uses the box: a boost_qos_dbg job (1 GPU, 30 min) named sfc_qbox,
started on demand, reused by later calls, that ends by itself after 10 minutes without steps.
A step dies with its host job: anything longer than --min-left minutes goes through submit.py.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from submit import SLURM_DIR, build_script, load_config, setup_lines

BOX = "sfc_qbox"
IDLE_S = 600
WAITING = ("PENDING", "CONFIGURING")
# The box's own step: ends the job once no other step has run for IDLE_S seconds (a qrun step may start
# before it, so it skips its own step id, not step 0).
BOX_LOOP = (f"idle=0; while [ $idle -lt {IDLE_S} ]; do sleep 30; "
            "if squeue -h -s -j $SLURM_JOB_ID -o %i | grep -qvE \"[.]($SLURM_STEP_ID|batch|extern)$\"; "
            "then idle=0; else idle=$((idle + 30)); fi; done")
# `<cgroup> <memory limit> <rss + shmem>` for the job and each of its steps (cgroup v1).
MEM_PROBE = ("cd /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$SLURM_JOB_ID && for d in . step_*; do "
             "echo $d $(cat $d/memory.limit_in_bytes) "
             "$(awk '$1==\"total_rss\"||$1==\"total_shmem\"{s+=$2} END{print s+0}' $d/memory.stat); done")
GB = 2**30


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def log(msg: str) -> None:
    print(f"[qrun] {msg}", file=sys.stderr, flush=True)


def minutes(left: str) -> float:
    """squeue %L ([D-]HH:MM:SS, MM:SS, UNLIMITED) in minutes."""
    if not left[:1].isdigit():
        return float("inf") if left == "UNLIMITED" else 0.0
    days, _, hms = left.rpartition("-")
    s = 0
    for part in hms.split(":"):
        s = 60 * s + int(part)
    return int(days or 0) * 1440 + s / 60


def free_gb(job: str, node: str) -> float:
    """Memory the job can lend on `node`: limit - rss - unused limits of other steps - margin."""
    out = sh(["srun", f"--jobid={job}", "--overlap", "-N1", "-n1", f"-w{node}", "--gres=none", "--mem=100M",
              "bash", "-c", MEM_PROBE])
    rows = [line.split() for line in out.splitlines()]
    limit, used = next((int(lim), int(rss)) for d, lim, rss in rows if d == ".")
    reserved = sum(max(int(lim) - int(rss), 0) for d, lim, rss in rows if d != "." and int(lim) < limit)
    return (limit - used - reserved - max(8 * GB, 0.15 * limit)) / GB


def host(mem: int, min_left: float) -> tuple[str, str] | None:
    """A running booster job (not the box) with enough free memory and time left, longest time left first."""
    out = sh(["squeue", "--me", "-h", "-t", "R", "-p", "boost_usr_prod", "-o", "%i|%j|%B|%L"])
    jobs = sorted((line.split("|") for line in out.splitlines()), key=lambda j: -minutes(j[3]))
    for job, name, node, left in jobs:
        if name == BOX or minutes(left) < min_left:
            continue
        try:
            free = free_gb(job, node)
        except subprocess.CalledProcessError:   # ended meanwhile
            continue
        if free >= mem:
            log(f"job {job} ({name}) on {node}: {free:.0f}G free, {minutes(left):.0f} min left")
            return job, node
    return None


def box(min_left: float) -> tuple[str, str]:
    """The box job and its node, submitted and waited for if none has min_left minutes left."""
    out = sh(["squeue", "--me", "-h", "-n", BOX, "-o", "%i|%T|%L"])
    job = next((i for i, state, left in (line.split("|") for line in out.splitlines())
                if state in WAITING or (state == "RUNNING" and minutes(left) >= min_left)), None)
    if job is None:
        cfg = load_config(SLURM_DIR / "presets" / "dbg.yaml", [])
        cfg["job_name"] = BOX
        script, logs_dir = build_script(cfg, ["bash", "-c", BOX_LOOP])
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        job = subprocess.run(["sbatch", "--parsable"], input=script, check=True, capture_output=True,
                             text=True).stdout.strip().split(";")[0]
        log(f"box {job} submitted (boost_qos_dbg, 1 GPU, ends after {IDLE_S // 60} idle min)")
    reason = None
    while True:
        state, node, why = (sh(["squeue", "-h", "-j", job, "-o", "%T|%B|%r"]).strip() or "GONE||").split("|")
        if state == "RUNNING":
            log(f"box {job} on {node}")
            return job, node
        if state not in WAITING:
            raise SystemExit(f"[qrun] box {job} is {state}")
        if why != reason:
            log(f"box {job} pending ({why})")
            reason = why
        time.sleep(3)


def main() -> int:
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mem", type=int, default=16, help="step memory limit in GB (default 16)")
    ap.add_argument("--cpus", type=int, default=4, help="cpus and OMP threads of the step (default 4)")
    ap.add_argument("--gpu", action="store_true", help="needs a GPU: run in the box")
    ap.add_argument("--min-left", type=float, default=15, help="minutes the host must still run (default 15)")
    args = ap.parse_args(argv[:sep])
    command = argv[sep + 1:]
    if not command:
        raise SystemExit("Missing command after '--'; see `python slurm/qrun.py -h`.")

    target = None if args.gpu else host(args.mem, args.min_left)
    job, node = target or box(args.min_left)
    gres = "--gres=none" if target else "--gres=gpu:1"
    cfg = load_config(SLURM_DIR / "presets" / "dbg.yaml", [])
    prelude = "\n".join(["set -e", f"cd {shlex.quote(str(Path.cwd()))}", *setup_lines(cfg),
                         f"export OMP_NUM_THREADS={args.cpus}", 'exec "$@"'])
    return subprocess.run(["srun", f"--jobid={job}", "--overlap", "-N1", "-n1", f"-w{node}", gres,
                           f"--mem={args.mem}G", f"--cpus-per-task={args.cpus}",
                           "bash", "-c", prelude, "qrun", *command], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
