"""`sfc <command> <config>`: pretrain-pool | pretrain | fm | bench-split | bench-extract | bench-probe | bench-segdec."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .io import load_yaml, override


def pretrain_pool(cfg: dict, args) -> None:
    """Pretraining manifest `data.manifests[0]` from the catalog (`data.pool`, data/pool.py)."""
    from .data import pool
    d, p = cfg["data"], cfg["data"]["pool"]
    cat = pd.read_csv(p["catalog"], usecols=pool.COLUMNS, low_memory=False, dtype={"session": str})
    df = pool.select(cat, p["datasets"], pd.read_csv(p["exclude"], usecols=["subject"])["subject"])
    if args.sample:  # uniform over the pool (its real mix): a manifest for the loader check
        df = df.sample(args.sample, random_state=int(cfg["seed"])).reset_index(drop=True)
    n = len(df)
    df = pool.with_patches(df, float(d["patch_mm"]), args.workers)
    out = Path(args.out or d["manifests"][0])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("dataset").agg(scans=("path", "size"), subjects=("subject", "nunique"),
                                    multi_frame=("frames", lambda f: int((f > 1).sum())), frames=("frames", "sum"),
                                    patches=("n_patch", "sum")).to_string())
    print(f"{len(df)} of {n} scans kept (readable, 3D or a frame per access) -> {out}")


def pretrain(cfg: dict, args) -> None:
    from .pretrain import pretrain as run
    if path := run(cfg):
        print(path)


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
    from .bench import split as bench
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
    if args.tune:  # pretraining tuning: no subject of any seed's test split is even read
        from .bench.probe import tune_subjects
        vols = vols[vols["subject"].isin(tune_subjects(pd.read_csv(Path(cfg["out_dir"]) / "splits.csv")))]
    if args.sample:
        vols = pd.concat([g.sort_values("dense", ascending=False, kind="stable").head(args.sample)
                          for _, g in vols.groupby("dataset")])
    if args.shard:
        i, n = args.shard.split("/")
        i = int(os.environ["SLURM_ARRAY_TASK_ID"]) if i == "auto" else int(i)
        vols = vols.iloc[i::int(n)]
    group, fm_cfg = cfg["fm"]["groups"][args.group], cfg["fm"]
    root = Path(args.out or fm_cfg["out_root"])
    cfgs = [override(load_config(f"configs/fm/{s}.yaml"), args.fm_set) | {"save": fm_cfg["save"][s] | {"dtype": "float16"}}
            for s in group["configs"]]
    extract(cfgs, vols["input"].tolist(), [root / s for s in group["configs"]], device=args.device,
            ids=vols["id"].tolist(), workers=int(group["workers"]))


def bench_probe(cfg: dict, args) -> None:
    """Probes of the benchmark tasks on the stored features of `--models` (default: every model under out_root),
    on the volumes all of them have; `--cache-only` just builds each model's `_globals.pt`."""
    from .bench import probe as bp
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
    if args.tune:
        bp.tune(models, tasks, vols, splits, Path(args.out or cfg["probe"]["out"]), args.device)
        return
    seeds = [int(s) for s in args.seeds.split(",")]
    common = {m: root / m for m in args.common} if args.common else None
    bp.run(models, tasks, vols, splits, Path(args.out or cfg["probe"]["out"]), seeds, args.device, common)


def bench_segdec(cfg: dict, args) -> None:
    """Segmentation through each model's official decoder on its frozen encoder (bench/segdec.py): `--stage prep
    --group g [--shard i/n|auto/n]` caches a group's preprocessed inputs; `--stage train --models m.. --init
    pretrained|random` trains and tests (`--tasks`: default every seg task; `--sample k`, `--iters n`: pilots)."""
    from .bench import segdec as bs
    tasks = load_yaml(cfg["probe"]["tasks"])
    tasks = {t: tasks[t] for t in (args.tasks or [t for t, c in tasks.items() if c["type"] == "seg"])}
    vols = pd.read_csv(Path(cfg["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(cfg["out_dir"]) / "splits.csv")
    seed = int(args.seeds.split(",")[0])
    items = bs.task_items(tasks, vols, splits, seed, args.sample)
    if args.stage == "prep":
        shard = args.shard and args.shard.replace("auto", os.environ.get("SLURM_ARRAY_TASK_ID", "auto"))
        bs.prep(cfg, args.group, items, args.device, shard, args.fm_set)
        return
    for m in args.models:
        for t in tasks:
            bs.train(cfg, m, t, items[t], seed, args.init, args.device, args.iters, args.resume, args.val_only,
                     args.fm_set)


COMMANDS = {"pretrain-pool": pretrain_pool, "pretrain": pretrain, "fm": fm, "bench-split": bench_split,
            "bench-extract": bench_extract, "bench-probe": bench_probe, "bench-segdec": bench_segdec}


def main() -> None:
    ap = argparse.ArgumentParser(prog="sfc")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("config", help="YAML config")
    ap.add_argument("--images", nargs="+", help="fm: image files (single-modality models)")
    ap.add_argument("--csv", help="fm: CSV with `id` + one column per modality")
    ap.add_argument("--out", help="fm: output dir; bench-extract: output root; pretrain-pool: manifest path")
    ap.add_argument("--device", default="cuda", help="fm: cuda | cpu")
    ap.add_argument("--overwrite", action="store_true", help="fm: recompute existing outputs")
    ap.add_argument("--group", help="bench-extract: model group of the config")
    ap.add_argument("--shard", help="bench-extract: i/n or auto/n ($SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--sample", type=int, help="bench-extract: k volumes per dataset; pretrain-pool: k pool scans")
    ap.add_argument("--query", help="bench-extract: pandas query on volumes.csv, e.g. \"dataset != 'IXI'\"")
    ap.add_argument("--models", nargs="+", help="bench-probe: model dirs under fm.out_root (default: all)")
    ap.add_argument("--tasks", nargs="+", help="bench-probe: task names (default: all)")
    ap.add_argument("--seeds", default="0", help="bench-probe: split seeds, e.g. 0,1,2,3,4")
    ap.add_argument("--common", nargs="+", help="bench-probe: models whose shared volumes are used (default: --models)")
    ap.add_argument("--cache-only", action="store_true", help="bench-probe: only build the feature caches")
    ap.add_argument("--tune", action="store_true",
                    help="bench-extract: only subjects in no seed's test split; bench-probe: their tuning skill")
    ap.add_argument("--fm-set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="bench-extract / bench-segdec: fm config fields, e.g. checkpoint=<run> args.crop=true")
    ap.add_argument("--val-only", action="store_true", help="bench-segdec: best val Dice only, no test volume read")
    ap.add_argument("--stage", choices=["prep", "train"], help="bench-segdec: prep (group inputs) | train")
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "random"], help="bench-segdec: encoder")
    ap.add_argument("--iters", type=int, help="bench-segdec: training iterations (pilot override)")
    ap.add_argument("--resume", action="store_true", help="bench-segdec: skip runs whose json is written (chained jobs)")
    ap.add_argument("--workers", type=int, default=8, help="pretrain-pool: header-reading processes")
    ap.add_argument("--override", nargs="+", default=[], metavar="KEY=VALUE",
                    help="config fields, dotted keys and YAML values, e.g. train.steps=200 data.manifests=[a.csv]")
    args = ap.parse_args()
    if args.command == "fm" and not ((args.images or args.csv) and args.out):
        ap.error("fm needs --images or --csv, and --out")
    # fm: the config is resolved (env vars) by fm.api
    cfg = None if args.command == "fm" else load_yaml(args.config)
    override(cfg, args.override)
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    main()
