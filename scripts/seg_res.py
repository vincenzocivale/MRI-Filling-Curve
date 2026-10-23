"""Does reading finer help segmentation? TotalSeg-MRI with the frozen encoder read at patch_mm 16 / 8 / 4 and the
decoder emitting `decode` cells per token edge, best validation Dice only (no test volume read: val_only), same
recipe as `sfc bench-segdec` (fm/segrun.py `train`), written under --out (not fcorrent's scratch).

    python scripts/seg_res.py <run dir> --out $SCRATCH/seg_res [--only p8d8 p4d4] [--iters 3000] [--sample 20]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.fm import segrun
from sfc_gdn2.fm import sfc as sfc_mod
from sfc_gdn2.io import load_yaml

# name: (patch_mm, decode, init). decode <= the voxels per patch edge (11 at 16 mm, 5 at 8 mm, 3 at 4 mm on 1.5 mm voxels)
# adds nothing; p4d16 (4x the cost of p4d8, seg_cost.py) is the reference the cheaper decoders are judged against.
# p16d4: control, 4^3 cells where a 16 mm patch holds ~11^3 voxels (what a small decoder costs when it is too small).
# rand_*: the same net with a random encoder.
GRID = {"p16d16": (16, 16, "pretrained"), "p16d4": (16, 4, "pretrained"), "p8d16": (8, 16, "pretrained"), "p8d8": (8, 8, "pretrained"),
        "p8d4": (8, 4, "pretrained"), "p4d16": (4, 16, "pretrained"), "p4d8": (4, 8, "pretrained"),
        "p4d4": (4, 4, "pretrained"),
        "rand_p8d8": (8, 8, "random")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", default=list(GRID), choices=list(GRID))
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--sample", type=int, help="k volumes per split (smoke test)")
    ap.add_argument("--val-max", type=int, default=30, help="validation volumes (the first k of the seeded random "
                    "draw, the same in every configuration): the sliding-window validation is the expensive part")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0, help="patches per tile: a tile above it is read at a coarser "
                    "patch (SegNet.tile_mm); 0: no limit (default). 4 mm on TotalSeg: 1.6% of the scans above 5e5, up to 2e6")
    ap.add_argument("--val-every", type=int, help="default iters // 2 (the benchmark: 2500 of 12500)")
    ap.add_argument("--chunk", type=int, default=0, help="tokens per piece when the encoder reads a longer sequence "
                    "(model.py flat_chunked, the same function in constant memory); 0: whole sequence")
    ap.add_argument("--bench", default="configs/bench/bench.yaml")
    a = ap.parse_args()
    bench = load_yaml(a.bench)
    tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
    vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(bench["out_dir"]) / "splits.csv")
    items = bs.task_items(tasks, vols, splits, a.seed, a.sample)["totalseg"]
    val = [i for i in items if i["split"] == "val"][: a.val_max]
    items = [i for i in items if i["split"] != "val"] + val
    seg = Path(items[0]["seg"])
    k = 1 + (len(list(seg.glob("*.nii*"))) if seg.is_dir() else 1)
    s = bench["segdec"]
    for name in a.only:
        mm, decode, init = GRID[name]
        out = Path(a.out) / f"{name}.json"
        if out.exists() and json.loads(out.read_text()).get("iters") == a.iters:
            print(f"[seg_res] {name}: done, skipped", flush=True)
            continue
        job = {"stage": "train", "val_only": True, "device": "cuda", "cache": str(Path(s["cache"]) / "sfc_gdn2"),
               "cfg": {"model": "sfc_gdn2", "checkpoint": a.checkpoint, "args": {"patch_mm": mm, "decode": decode, "max_tokens": a.max_tokens, "chunk": a.chunk}},
               "items": items, "k": k, "init": init, "iters": a.iters, "val_every": a.val_every or max(a.iters // 2, 1),
               "batch": int(s["batch"]), "lr": float(s["lr"]), "seed": a.seed, "out": str(out)}
        print(f"[seg_res] {name}: patch_mm {mm} decode {decode} init {init}, {a.iters} iters, "
              f"{sum(i['split'] == 'train' for i in items)} train / {sum(i['split'] == 'val' for i in items)} val volumes",
              flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        sfc_mod.CAP_STATS.update(tiles=0, capped=0)
        segrun.train(job)
        res = json.loads(out.read_text())   # tiles read (training + validation) and those read at a coarser patch
        out.write_text(json.dumps(res | {"max_tokens": a.max_tokens, "chunk": a.chunk, "cap": dict(sfc_mod.CAP_STATS)}, indent=1))
        print(f"[seg_res] {name}: tiles {sfc_mod.CAP_STATS['tiles']}, read at a coarser patch {sfc_mod.CAP_STATS['capped']}", flush=True)


if __name__ == "__main__":
    main()
