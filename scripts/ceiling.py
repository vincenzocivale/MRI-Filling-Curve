"""Dice ceiling of a leaf partition: every leaf (8 mm cell, or its 8 children at 4 mm when split) predicts its majority label
(background included), the best a decoder that outputs one class per leaf could do. Policies: uniform 8 mm, uniform 4 mm,
random / oracle (true gain) share f of the 8 mm cells read at 4 mm. Same validation volumes and Dice (segrun.dice_on, on the
cache's grid) as scripts/adaptive_cached_train.py. Says whether the decoder or the leaf resolution limits the Dice.

    python scripts/ceiling.py --tokens <tok_pre> [--n 100] [--frac 0.3]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfc_gdn2.adaptive import block_sum, hist
from sfc_gdn2.adaptive_cached import cell_gain
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.fm import segrun
from sfc_gdn2.fm.token_cache import TokenCache, fine_ok
from sfc_gdn2.io import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--frac", type=float, default=0.3)
a = ap.parse_args()
dev = torch.device("cuda")
bench = load_yaml("configs/bench/bench.yaml")
tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
items = bs.task_items(tasks, vols, pd.read_csv(Path(bench["out_dir"]) / "splits.csv"), 0)["totalseg"]
cache = Path(bench["segdec"]["cache"]) / "sfc_gdn2"
tc = TokenCache(a.tokens)
val = [i for i in items if i["split"] == "val" and (cache / f"{i['id']}.pt").exists() and (Path(a.tokens) / f"{i['id']}.json").exists()][: a.n]
B = 1 + len(list(Path(val[0]["seg"]).glob("*.nii*")))
res = {p: [] for p in ("uni8", "uni4", f"random{a.frac}", f"oracle{a.frac}")}
for it in val:
    r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    y = r["y"].to(dev)
    k4 = tuple(tc.meta(it["id"])["k4"])
    k8 = tuple(2 * q for q in k4)
    shape = tuple(y.shape)
    pad = [p for n, q in zip(reversed(shape), reversed(k8)) for p in (0, -n % q)]
    yp = F.pad(y, pad, value=255)
    g8 = [s // q for s, q in zip(yp.shape, k8)]
    H2 = hist(yp, k4, B).float()                      # [N4, B] in 4 mm cells
    H1 = block_sum(H2, g8)
    maj1, maj2 = H1.argmax(1), H2.argmax(1)

    def paint(sel):   # sel [N8] bool: those cells at 4 mm, the others at 8 mm -> label volume on the grid
        m4 = sel.view(*g8)
        for ax in range(3):
            m4 = m4.repeat_interleave(2, ax)
        lab4 = maj2.view(*[2 * g for g in g8])
        lab8 = maj1.view(*g8)
        for ax in range(3):
            lab8 = lab8.repeat_interleave(2, ax)
        lab = torch.where(m4, lab4, lab8)                # on the 4 mm cell grid
        for ax, q in enumerate(k4):
            lab = lab.repeat_interleave(q, ax)
        return lab[: shape[0], : shape[1], : shape[2]].to(torch.uint8)

    n8, nsel = len(maj1), round(a.frac * len(maj1))
    four = fine_ok(k4)
    none, allc = torch.zeros(n8, dtype=torch.bool, device=dev), torch.ones(n8, dtype=torch.bool, device=dev)
    rnd = torch.zeros(n8, dtype=torch.bool, device=dev)
    rnd[torch.randperm(n8, device=dev)[:nsel]] = True
    orc = torch.zeros(n8, dtype=torch.bool, device=dev)
    orc[cell_gain(yp, k4, g8, B).argsort(descending=True)[:nsel]] = True
    for name, sel in (("uni8", none), ("uni4", allc if four else none), (f"random{a.frac}", rnd if four else none),
                      (f"oracle{a.frac}", orc if four else none)):
        res[name].append(segrun.dice_on(paint(sel), y, B))
print(f"{len(val)} validation volumes, ceiling Dice of one-class-per-leaf prediction (mean over volumes):")
for name, v in res.items():
    print(f"  {name:<12} {np.nanmean(v):.4f}")
