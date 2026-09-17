from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd
import torch
from .data.manifest import build
from .io import dump_json, dump_yaml, environment, load_yaml, stable_id
from .metrics import geometry_metrics


def prepare():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); a = ap.parse_args()
    cfg = load_yaml(a.config); df = build(cfg)
    out = Path(cfg["output"]); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False)
    print(f"{len(df)} samples -> {out}")


def run():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--geometry-only", action="store_true"); a = ap.parse_args()
    cfg = load_yaml(a.config); shape = cfg["data"]["target_shape"]; patch = cfg["data"]["patch_size"]
    if len(set(shape)) != 1 or shape[0] % patch: raise ValueError("Use a cubic target_shape divisible by patch_size.")
    grid = shape[0] // patch
    geom = [geometry_metrics(c, grid, cfg["geometry"]["windows"], cfg["geometry"]["spatial_radius"], cfg["seed"]) for c in cfg["curves"]]
    if a.geometry_only:
        out = Path(cfg["output_root"])/f"geometry-{stable_id(cfg)}"; out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(geom).to_csv(out/"geometry.csv", index=False); dump_yaml(cfg, out/"config.yaml"); dump_json(environment(), out/"environment.json")
        print(out); return
    from .engine import microtrain
    for curve, gm in zip(cfg["curves"], geom):
        run_cfg = {**cfg, "curve": curve}; rid = f"{curve}-{stable_id(run_cfg)}"; out = Path(cfg["output_root"])/rid; out.mkdir(parents=True, exist_ok=True)
        dump_yaml(run_cfg, out/"config.yaml"); dump_json(environment(), out/"environment.json")
        pd.DataFrame([gm]).to_csv(out/"geometry.csv", index=False)
        model, hist, per_ds, extra = microtrain(cfg, curve)
        pd.DataFrame(hist).to_csv(out/"history.csv", index=False); pd.DataFrame(per_ds).to_csv(out/"per_dataset.csv", index=False)
        dump_json({"curve": curve, **extra, "mean_mse": sum(x["mse"] for x in per_ds)/len(per_ds), "mean_mae": sum(x["mae"] for x in per_ds)/len(per_ds)}, out/"metrics.json")
        torch.save(model.state_dict(), out/"checkpoint.pt"); print(out)


def summarize():
    ap = argparse.ArgumentParser(); ap.add_argument("output_root"); a = ap.parse_args(); rows = []
    for p in Path(a.output_root).glob("*/metrics.json"):
        with open(p) as f: r = json.load(f)
        r["run"] = p.parent.name; rows.append(r)
    if not rows: raise SystemExit("No metrics.json found.")
    df = pd.DataFrame(rows).sort_values("mean_mse"); out = Path(a.output_root)/"summary.csv"; df.to_csv(out, index=False)
    print(df.to_string(index=False)); print(f"\n{out}")
