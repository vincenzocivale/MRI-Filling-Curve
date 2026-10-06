"""`sfc <command> <config>`: prepare | split | pretrain | probe | fm | summarize."""
from __future__ import annotations

import argparse
import json
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
    from .probe import probe_raw, probe_run
    if args.raw:
        probe_raw(cfg, load_yaml(args.raw))
    for run_dir in args.run_dir or []:
        probe_run(cfg, run_dir)


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


def summarize(_, args) -> None:
    """One row per (run, probe, group) under <root>: pretrained/init per run, plus the raw baseline."""
    root, rows = Path(args.config), []
    for p in sorted([*root.glob("*/probe-*/metrics.json"), *root.glob("raw-probe-*/metrics.json")]):
        m = json.loads(p.read_text())
        key = m["select_on"].split("_", 1)[1]  # val_<metric> or cv<k>_<metric>
        for group, r in m["groups"].items():
            lo, hi = r[f"test_{key}_ci95"]
            rows.append({"label": m["label"], "metric": key, "objective": m["objective"], "curve": m["curve"],
                         "group": group, "selected": r["selected"], "val": r["val"][key], "test": r["test"][key],
                         "ci95": f"[{lo:.3f}, {hi:.3f}]", "run": p.parent.parent.name})
    if not rows:
        raise SystemExit(f"No probe results under {root}.")
    df = pd.DataFrame(rows).sort_values(["label", "group", "objective", "curve"])
    print(df.to_string(index=False))
    df.to_csv(root / "probe_summary.csv", index=False)


COMMANDS = {"prepare": prepare, "split": split, "pretrain": pretrain, "probe": probe, "fm": fm, "summarize": summarize}


def main() -> None:
    ap = argparse.ArgumentParser(prog="sfc")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("config", help="YAML config (for `summarize`: a pretraining output_root)")
    ap.add_argument("--run-dir", nargs="+", help="probe: pretraining run directories")
    ap.add_argument("--raw", help="probe: also run the encoder-free baseline (pass the pretrain config)")
    ap.add_argument("--images", nargs="+", help="fm: image files (single-modality models)")
    ap.add_argument("--csv", help="fm: CSV with `id` + one column per modality")
    ap.add_argument("--out", help="fm: output directory (one <id>.pt per image)")
    ap.add_argument("--device", default="cuda", help="fm: cuda | cpu")
    ap.add_argument("--overwrite", action="store_true", help="fm: recompute existing outputs")
    args = ap.parse_args()
    if args.command == "fm" and not ((args.images or args.csv) and args.out):
        ap.error("fm needs --images or --csv, and --out")
    raw = args.command in ("summarize", "fm")  # fm: the config is resolved (env vars) by fm.api
    COMMANDS[args.command](None if raw else load_yaml(args.config), args)


if __name__ == "__main__":
    main()
