"""`sfc <command> <config>`: prepare | split | pretrain | probe | fm | summarize | bench-split | bench-extract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .io import load_yaml


def prepare(cfg: dict, _) -> None:
    from .data.manifest import build
    df = build(cfg)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} samples -> {out}")


def split(cfg: dict, _) -> None:
    from .data.labels import attach
    from .data.manifest import build
    from .data.splits import assign_splits, summarize
    df = pd.read_csv(cfg["manifest"]) if cfg.get("manifest") else build(cfg)
    df = attach(df, cfg["kind"], cfg["root"], cfg["labels"])
    df["split"] = assign_splits(df, int(cfg["seed"]), cfg["counts"], cfg["labels"][0])
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(summarize(df, cfg["labels"][0]).to_string(index=False), f"\n{out}")


def pretrain(cfg: dict, args) -> None:
    from .pretrain import pretrain as run
    if path := run(cfg):
        print(path)


def probe(cfg: dict, args) -> None:
    """Under torchrun, candidates (checkpoint x curve x features) are split across GPUs."""
    import os
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from .probe import probe_raw, probe_run
    if "WORLD_SIZE" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        # ranks fit different candidate shards (hours on segmentation): the gather waits for the slowest
        dist.init_process_group("nccl", timeout=timedelta(hours=12))
    if args.raw:
        probe_raw(cfg, load_yaml(args.raw))
    for run_dir in args.run_dir or []:
        probe_run(cfg, run_dir)
    if dist.is_initialized():
        dist.destroy_process_group()


def fm(_, args) -> None:
    """Features of one external model for a list of images (`--images`, or `--csv` with an `id` column
    plus one column per modality for multi-modal models)."""
    from .fm.api import extract
    if args.csv:
        df = pd.read_csv(args.csv)
        mods = [c for c in df.columns if c != "id"]
        images = df[mods[0]].tolist() if len(mods) == 1 else df[mods].to_dict("records")
        ids = df["id"].astype(str).tolist() if "id" in df else None
    else:
        images, ids = args.images, None
    extract(args.config, images, args.out, device=args.device, overwrite=args.overwrite, ids=ids)


def bench_split(cfg: dict, _) -> None:
    """Volume list + fixed splits of the downstream benchmark (src/sfc_gdn2/bench.py)."""
    from . import bench
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    vols = bench.stage(bench.volumes(pd.read_csv(cfg["catalog"], low_memory=False)), cfg["inputs_dir"])
    spl = bench.splits(vols, cfg["totalseg_meta"], int(cfg["seeds"]))
    vols.to_csv(out / "volumes.csv", index=False)
    spl.to_csv(out / "splits.csv", index=False)
    (out / "SHA256SUMS").write_text("".join(f"{bench.sha256(out / n)}  {n}\n" for n in ("volumes.csv", "splits.csv")))
    print(vols.groupby("dataset").agg(volumes=("id", "size"), dense=("dense", "sum")).to_string())
    print(spl.groupby(["dataset", "split_s0"]).size().unstack().to_string(), f"\n{out}")


def bench_extract(cfg: dict, args) -> None:
    """Features of one model group (`--group`) for the benchmark volumes, one shard (`--shard i/n`, `auto/n` =
    $SLURM_ARRAY_TASK_ID) or `--sample k` volumes per dataset (pilot)."""
    from .fm.api import extract, load_config
    vols = pd.read_csv(Path(cfg["out_dir"]) / "volumes.csv")
    if args.query:
        vols = vols.query(args.query)
    if args.sample:
        vols = pd.concat([g.sort_values("dense", ascending=False, kind="stable").head(args.sample)
                          for _, g in vols.groupby("dataset")])
    if args.shard:
        i, n = args.shard.split("/")
        i = int(os.environ["SLURM_ARRAY_TASK_ID"]) if i == "auto" else int(i)
        vols = vols.iloc[i::int(n)]
    group, fm_cfg = cfg["fm"]["groups"][args.group], cfg["fm"]
    root = Path(args.out or fm_cfg["out_root"])
    cfgs = [load_config(f"configs/fm/{s}.yaml") | {"save": fm_cfg["save"][s] | {"dtype": "float16"}}
            for s in group["configs"]]
    extract(cfgs, vols["input"].tolist(), [root / s for s in group["configs"]], device=args.device,
            ids=vols["id"].tolist(), dense=vols["dense"].tolist(), workers=int(group["workers"]),
            verify_shared=args.verify_shared)


def bench_probe(cfg: dict, args) -> None:
    """Probes of the benchmark tasks on the stored features of `--models` (default: every model under out_root),
    on the volumes all of them have; `--cache-only` just builds each model's `_globals.pt`."""
    from . import bench_probe as bp
    from .io import load_yaml
    root = Path(cfg["fm"]["out_root"])
    names = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())
    models = {m: root / m for m in names}
    if args.cache_only:
        for d in models.values():
            bp.load_globals(d)
            print(f"[probe] cached {d}", flush=True)
        return
    tasks = load_yaml(cfg["probe"]["tasks"])
    tasks = {t: tasks[t] for t in (args.tasks or tasks)}
    vols = pd.read_csv(Path(cfg["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(cfg["out_dir"]) / "splits.csv")
    seeds = [int(s) for s in args.seeds.split(",")]
    bp.run(models, tasks, vols, splits, Path(args.out or cfg["probe"]["out"]), seeds, args.device)


def summarize(_, args) -> None:
    """One row per (run, probe, candidate) under <root>: every checkpoint per run, plus the raw baselines."""
    root, rows = Path(args.config), []
    for p in sorted([*root.glob("*/probe-*/metrics.json"), *root.glob("raw-probe-*/metrics.json")]):
        m = json.loads(p.read_text())
        key = m["metric"]
        for group, r in m["groups"].items():
            lo, hi = r[f"test_{key}_ci95"]
            rows.append({"label": m["label"], "metric": key, "objective": m["objective"], "curve": m["curve"],
                         "candidate": group, "l2": r["l2"], "test": r["test"][key],
                         "ci95": f"[{lo:.3f}, {hi:.3f}]", "run": p.parent.parent.name})
    if not rows:
        raise SystemExit(f"No probe results under {root}.")
    df = pd.DataFrame(rows).sort_values(["label", "objective", "run", "candidate"])
    print(df.to_string(index=False))
    df.to_csv(root / "probe_summary.csv", index=False)


COMMANDS = {"prepare": prepare, "split": split, "pretrain": pretrain, "probe": probe, "fm": fm, "summarize": summarize,
            "bench-split": bench_split, "bench-extract": bench_extract, "bench-probe": bench_probe}


def main() -> None:
    ap = argparse.ArgumentParser(prog="sfc")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("config", help="YAML config (for `summarize`: a pretraining output_root)")
    ap.add_argument("--run-dir", nargs="+", help="probe: pretraining run directories")
    ap.add_argument("--raw", help="probe: also run the encoder-free baseline (pass the pretrain config)")
    ap.add_argument("--images", nargs="+", help="fm: image files (single-modality models)")
    ap.add_argument("--csv", help="fm: CSV with `id` + one column per modality")
    ap.add_argument("--out", help="fm: output directory (one <id>.pt per image); bench-extract: output root")
    ap.add_argument("--device", default="cuda", help="fm: cuda | cpu")
    ap.add_argument("--overwrite", action="store_true", help="fm: recompute existing outputs")
    ap.add_argument("--group", help="bench-extract: model group of the config")
    ap.add_argument("--shard", help="bench-extract: i/n or auto/n ($SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--sample", type=int, help="bench-extract: only k volumes per dataset (dense first)")
    ap.add_argument("--query", help="bench-extract: pandas query on volumes.csv, e.g. \"dataset != 'IXI'\"")
    ap.add_argument("--models", nargs="+", help="bench-probe: model dirs under fm.out_root (default: all)")
    ap.add_argument("--tasks", nargs="+", help="bench-probe: task names (default: all)")
    ap.add_argument("--seeds", default="0", help="bench-probe: split seeds, e.g. 0,1,2,3,4")
    ap.add_argument("--cache-only", action="store_true", help="bench-probe: only build the feature caches")
    ap.add_argument("--verify-shared", action="store_true", help="bench-extract: check the shared preprocessing")
    args = ap.parse_args()
    if args.command == "fm" and not ((args.images or args.csv) and args.out):
        ap.error("fm needs --images or --csv, and --out")
    raw = args.command in ("summarize", "fm")  # fm: the config is resolved (env vars) by fm.api
    COMMANDS[args.command](None if raw else load_yaml(args.config), args)


if __name__ == "__main__":
    main()
