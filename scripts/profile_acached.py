"""Where does an iteration of adaptive_cached_train.py go? Per tile: cold token reads (8 / 4 mm), forward, backward."""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from sfc_gdn2.adaptive_cached import CachedAdaptiveNet
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.fm import segrun
from sfc_gdn2.fm.token_cache import fine_ok
from sfc_gdn2.io import load_yaml

TOK = "/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/bench_sb/tok_pre"
dev, T = torch.device("cuda"), 160
bench = load_yaml("configs/bench/bench.yaml")
tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
items = bs.task_items(tasks, vols, pd.read_csv(Path(bench["out_dir"]) / "splits.csv"), 0)["totalseg"]
cache = Path(bench["segdec"]["cache"]) / "sfc_gdn2"
items = [i for i in items if i["split"] == "train" and (Path(TOK) / f"{i['id']}.json").exists()][100:116]
net = CachedAdaptiveNet(TOK, 51).to(dev)
rng, rows = np.random.default_rng(1), []
sync = torch.cuda.synchronize
for it in items:
    r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    x, y, k4 = r["x"], r["y"], net.cache.meta(it["id"])["k4"]
    shape = x.shape[1:]
    ext = [min(T, s) for s in shape]
    st = np.array([int(rng.integers(s - e + 1)) for s, e in zip(shape, ext)])
    unit = [4 * q for q in k4]
    corner = [int(s) // u * u for s, u in zip(st, unit)]
    V = [min(T, s - c) for s, c in zip(shape, corner)]
    k8 = net.cell_dims(k4)
    g8 = [-(-v // a) for v, a in zip(V, k8)]
    t = time.time(); net.cache.tokens(it["id"], 8.0, corner, g8, k8); t8 = time.time() - t
    t = time.time()
    if fine_ok(k4):
        net.cache.tokens(it["id"], 4.0, corner, [2 * g for g in g8], k4)
    t4 = time.time() - t
    xv = x[0, corner[0]:corner[0] + V[0], corner[1]:corner[1] + V[1], corner[2]:corner[2] + V[2]].to(dev).float()
    yv = y[corner[0]:corner[0] + V[0], corner[1]:corner[1] + V[1], corner[2]:corner[2] + V[2]].to(dev)
    net.train(); net.q = 0.0
    sync(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = net(it["id"], corner, xv, yv)       # tokens now cached in _recent: no read
    sync(); tf = time.time() - t
    padw = [q for n_ in reversed(V) for q in (0, T - n_)]
    t = time.time()
    loss = segrun.loss_fn(F.pad(o, padw)[None], F.pad(yv, padw, value=255)[None], 51) + net.aux
    loss.backward(); sync(); tb = time.time() - t
    rows.append((it["id"][-18:], V, k4, g8, t8, t4, tf, tb))
    print(f"{it['id'][-22:]:<22} V{V} k4{k4} cells8 {np.prod(g8):>7} read8 {t8:5.2f}s read4 {t4:5.2f}s forward {tf:5.2f}s backward {tb:5.2f}s", flush=True)
a = np.array([[r[4], r[5], r[6], r[7]] for r in rows])
print("mean per tile: read8 %.2f read4 %.2f forward %.2f backward %.2f s" % tuple(a.mean(0)))
