"""Ablation basket (01/11, user), step 1: Small's tokens of the WHOLE volume (dense) at 16, 8 and 4 mm, for the simple
heads of abl/heads.py. The reading is the AGD token cache's (fm/token_cache.read: the segdec input on the native
canonical grid, nested cells k4 = round(4 / spacing) voxels per axis at 4 mm, 2 k4 at 8 mm, 4 k4 at 16 mm, windows
past nmax tokens), so the 3-curve mean at 16 / 8 mm is the cache's (fm/token_cache.py) file for file. Kept: last
block ++ the decoder's taps (args.taps 3 -> blocks 1, 3, 5) at 16 / 8 mm, the last block only at 4 mm (disk: a WMH
scan's dense 4 mm grid is ~2e5 tokens). `--curves`: mean (the run's 3 curves averaged, default) and / or single
curves (raster, morton, hilbert), every name read in one pass (each curve once).

Out: <out>/<id>.<mm>.<curve>.npy [gx, gy, gz, C] fp16 (as TokenCache names them; `{curve}` in --out puts each curve
in its own folder) + <out>/<id>.json, written last (the volume's done marker): k4, spacing, shape (the segdec input's
grid), per level its cell (voxels per axis), grid and width; token (a, b, c) of level mm covers the input voxels
[a k, (a + 1) k) x ... (cut at shape), the input <inputs>/<id>.pt holds the map to the source / GT voxels
(grid_from_gt). <out>/cache.json: the encoder's signature (TokenCache.check's).

    python -m sfc_gdn2.abl.tokens --task wmh_seg --out $SCRATCH/sfc_runs/abl/tok/{curve}/wmh_seg [--curves mean,hilbert]
        [--sample k: the first k volumes per split] [--shard i/n] [--ids a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..fm.token_cache import LEVELS, read

ROOT = Path(__file__).resolve().parents[3]   # the repo (worktree) holding configs/
CKPT = "/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/tune/p32_heal8/run/lejepa-48e158e8b4/encoder_step002600.pt"
MANIFESTS = "/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2"
INPUTS = "/leonardo_scratch/fast/IscrC_SFMRI/fcorrent/segdec_inputs/sfc_gdn2"
TASKS = "configs/bench/probe_tasks_v2.yaml"
TAPS = 3                 # as the AGD runs (seg_dense.sh args.taps=3): the quarter points below the last block
SEED = 0                 # the bench seed of the AGD runs (--seeds 0): the splits and select()'s draw
NMAX, BLOCK, MARGIN = 200_000, 64, 8   # token_cache's windows (write_volume's defaults)


def task_items(task: str, data_seed: int | None = None, sample: int | None = None) -> list[dict]:
    """The task's volumes as bench-segdec draws them (bench/segdec.task_items, seed SEED); data_seed k: select()'s
    SFC_DATA_SEED (90 % of the train subjects, rng 1000 + k), val and test unchanged."""
    import pandas as pd

    from ..bench import segdec as bs
    from ..io import load_yaml

    cfg = load_yaml(ROOT / TASKS)[task]
    vols = pd.read_csv(Path(MANIFESTS) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(MANIFESTS) / "splits.csv")
    keep = os.environ.pop("SFC_DATA_SEED", None)
    try:
        if data_seed is not None:
            os.environ["SFC_DATA_SEED"] = str(int(data_seed))
        return bs.task_items({task: cfg}, vols, splits, SEED, sample)[task]
    finally:
        os.environ.pop("SFC_DATA_SEED", None)
        if keep is not None:
            os.environ["SFC_DATA_SEED"] = keep


def encoder(ckpt: str = CKPT):
    """Small, frozen, eval, on the GPU, taps as SFC.seg_net sets them -> (encoder, the run's curves, the run's seed,
    signature as segrun stamps the AGD cache)."""
    from ..fm.api import load_config
    from ..fm.sfc import SFC

    cfg = load_config(ROOT / "configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"], cfg["args"] = ckpt, {"taps": TAPS}
    w = SFC(cfg, "cuda")
    enc = w._encoder(True)
    enc.taps = tuple(len(enc.blocks) * i // 4 - 1 for i in range(4 - TAPS, 4))   # SFC.seg_net's int taps
    sig = {"checkpoint": str(w.ckpt), "taps": list(enc.taps), "curves": list(w.curves), "blocks": None}
    return enc.cuda().eval(), list(w.curves), int(w.run_cfg["seed"]), sig


def dump(enc, curves: list, seed: int, x, sp, out: dict, i: str) -> dict:
    """One volume: every level, every name of `out` ({curve: folder}) -> {mm: seconds}; the json last."""
    from ..data.volume import patch_dims, patch_grid

    x, sp = x.cuda(), sp.float().cuda()
    k4 = patch_dims(sp.tolist(), 4.0)
    meta, dt = {"k4": list(k4), "spacing": sp.tolist(), "shape": list(x.shape[1:]), "levels": {}}, {}
    for mm, m in LEVELS:
        k = [a * m for a in k4]
        names = [c for c, d in out.items() if not (d / f"{i}.{int(mm)}.{c}.npy").exists()]
        taps = mm >= 8   # 4 mm: the last block only
        meta["levels"][str(int(mm))] = {"cell": k, "grid": patch_grid(x.shape[1:], k).tolist(),
                                        "width": enc.dim * ((1 + len(enc.taps)) if taps else 1),
                                        "blocks": ["last", *enc.taps] if taps else ["last"]}
        if not names:
            continue
        torch.cuda.synchronize()
        t0 = time.time()
        a = read(enc, curves, seed, x, sp, k, mm, NMAX, BLOCK, MARGIN, taps=taps, want=names)
        torch.cuda.synchronize()
        dt[int(mm)] = time.time() - t0
        for c, t in a.items():
            dst = out[c] / f"{i}.{int(mm)}.{c}.npy"
            tmp = dst.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                np.save(f, t)
            os.replace(tmp, dst)
    for d in out.values():
        tmp = d / f"{i}.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, d / f"{i}.json")
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True, help="output folder; {curve}: one folder per curve name")
    ap.add_argument("--curves", default="mean", help="comma list among mean, raster, morton, hilbert")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--inputs", default=INPUTS, help="the segdec inputs <id>.pt (sfc_gdn2 preprocessing group)")
    ap.add_argument("--sample", type=int, help="the first k volumes of each split (tests)")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th volume from the i-th")
    ap.add_argument("--ids", help="comma list of ids (instead of the task's)")
    a = ap.parse_args()
    names = a.curves.split(",")
    assert len(names) == 1 or "{curve}" in a.out, "several curves: --out needs {curve}"
    out = {c: Path(a.out.replace("{curve}", c)) for c in names}
    enc, curves, seed, sig = encoder(a.checkpoint)
    assert all(c == "mean" or c in curves for c in names), f"curves among mean, {curves}"
    for c, d in out.items():
        d.mkdir(parents=True, exist_ok=True)
        f, s = d / "cache.json", sig | {"curve": c, "abl": "16/8 mm last ++ taps, 4 mm last (abl/tokens.py)"}
        if f.exists() and json.loads(f.read_text()) != json.loads(json.dumps(s)):
            raise ValueError(f"{d} holds another encoder's tokens: {f.read_text()}")
        f.write_text(json.dumps(s, indent=1))
    ids = a.ids.split(",") if a.ids else sorted(it["id"] for it in task_items(a.task, sample=a.sample))
    i0, n = (int(v) for v in a.shard.split("/"))
    ids = ids[i0::n]
    print(f"[abl-tok] {a.task}: {len(ids)} volumes, curves {names} -> {a.out}", flush=True)
    for c, i in enumerate(ids):
        if all((d / f"{i}.json").exists() for d in out.values()):
            continue
        r = torch.load(Path(a.inputs) / f"{i}.pt", map_location="cpu", weights_only=False)
        dt = dump(enc, curves, seed, r["x"], r["ctx"], out, i)
        mb = {mm: sum((d / f"{i}.{mm}.{w}.npy").stat().st_size for w, d in out.items()) / 2**20 for mm in (16, 8, 4)}
        print(f"[abl-tok] {c + 1}/{len(ids)} {i} grid {tuple(r['x'].shape[1:])}: "
              + " ".join(f"{mm} mm {dt.get(mm, 0):.1f} s {mb[mm]:.0f} MB" for mm in (16, 8, 4)), flush=True)


if __name__ == "__main__":
    main()
