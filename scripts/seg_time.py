"""Whole-volume inference (segrun.sliding: 160^3 tiles, step 0.5, the benchmark's own) on real TotalSeg-MRI volumes:
time and peak GPU memory per volume, for configurations given as patch_mm:decode:chunk. The decoder is untrained
(timing only). Volumes: by tokens per tile at the finest patch asked (header pass), from the median to the heaviest.

    python scripts/seg_time.py <run dir> --cfg 16:16:0 4:4:131072 [--n 8]
"""
import argparse
import json
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import sfc_gdn2.fm.sfc as m
from sfc_gdn2.fm.segrun import sliding

CACHE = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/segdec_bench/inputs/sfc_gdn2")


def tokens(path, mm):
    h = nib.load(path).header
    shape, zoom = h.get_data_shape()[:3], h.get_zooms()[:3]
    return int(np.prod([math.ceil(min(160, s) / max(1, round(mm / z))) for s, z in zip(shape, zoom)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cfg", nargs="+", default=["16:16:0", "4:4:131072"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="seg_time.json")
    a = ap.parse_args()
    vols = pd.read_csv("/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench/volumes.csv", low_memory=False)
    v = vols[(vols["dataset"] == "TotalSegmentator-MRI") & vols["id"].map(lambda i: (CACHE / f"{i}.pt").exists())].copy()
    finest = min(float(c.split(":")[0]) for c in a.cfg)
    v["tok"] = [tokens(p, finest) for p in v["input"]]
    v = v.sort_values("tok")
    pick = v.iloc[sorted({int(i) for i in np.linspace(len(v) // 2, len(v) - 1, a.n)})]
    print(f"{len(v)} cached volumes; picked tokens/tile at {finest} mm: {pick['tok'].tolist()}", flush=True)
    rows = []
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    for c in a.cfg:
        mm, dec, ch = (int(float(t)) for t in c.split(":"))
        w = m.SFC({"checkpoint": a.run, "args": {"patch_mm": mm, "decode": dec, "chunk": ch}}, "cuda")
        net = w.seg_net(51, pretrained=True)[0].cuda().eval()
        for _, r in pick.iterrows():
            d = torch.load(CACHE / f"{r['id']}.pt", map_location="cpu", weights_only=False)
            x = d["x"].float().cuda()
            ctx = torch.cat([d["ctx"].float(), torch.tensor(x.shape[1:], dtype=torch.float)]).cuda()
            row = {"cfg": c, "id": r["id"], "grid": list(x.shape[1:]), "tok_tile": int(r["tok"])}
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
                p = sliding(net, x, (160, 160, 160), 51, amp, ctx)
                torch.cuda.synchronize()
                row |= {"s": round(time.time() - t, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                        "finite": bool(torch.isfinite(p.float()).all())}
                del p
            except torch.cuda.OutOfMemoryError as e:
                row |= {"error": "OOM", "msg": str(e).splitlines()[0][:120]}
            print(json.dumps(row), flush=True)
            rows.append(row)
            Path(a.out).write_text(json.dumps(rows, indent=1))
            del x
            torch.cuda.empty_cache()
        del net


if __name__ == "__main__":
    main()
