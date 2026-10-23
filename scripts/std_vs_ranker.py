"""Share of the class-weighted gain held by the top 25% of the 8 mm cells, per selection rule, on the validation (and test)
volumes: fcorrent's intensity std (what his matryoshka uses), our cross-fitted ranker, random, the true gain.
Maps: ranker_scores/q25 (scores) and ranker_scores/oracle (gain) from scripts/ranker_scores.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.data.volume import patchify
from sfc_gdn2.io import load_yaml

V = Path("/leonardo_scratch/large/userexternal/vcivale0/ranker_scores")
TOK = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/bench_sb/tok_pre")
bench = load_yaml("configs/bench/bench.yaml")
tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
items = bs.task_items(tasks, vols, pd.read_csv(Path(bench["out_dir"]) / "splits.csv"), 0)["totalseg"]
cache = Path(bench["segdec"]["cache"]) / "sfc_gdn2"
dev, q = torch.device("cuda"), 0.25
for split in ("val", "test"):
    acc = {k: [0.0, 0.0] for k in ("std", "ranker", "random", "oracle")}
    rng = np.random.default_rng(0)
    for it in [i for i in items if i["split"] == split and (V / "q25" / f"{i['id']}.npy").exists()]:
        g = torch.from_numpy(np.load(V / "oracle" / f"{it['id']}.npy")).float().flatten()
        if g.sum() <= 0:
            continue
        r = torch.from_numpy(np.load(V / "q25" / f"{it['id']}.npy")).float().flatten()
        k4 = json.loads((TOK / f"{it['id']}.json").read_text())["k4"]
        k8 = tuple(2 * a for a in k4)
        x = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)["x"][0].float().to(dev)
        xp = F.pad(x, [p for n, a in zip(reversed(x.shape), reversed(k8)) for p in (0, -n % a)])
        s = patchify(xp, k8).std(1).cpu()
        n = round(q * len(g))
        for name, sc in (("std", s), ("ranker", r), ("random", torch.from_numpy(rng.random(len(g)))), ("oracle", g)):
            acc[name][0] += float(g[sc.argsort(descending=True)[:n]].sum())
            acc[name][1] += float(g.sum())
    print(split, {k: round(v[0] / max(v[1], 1e-9), 3) for k, v in acc.items()})
