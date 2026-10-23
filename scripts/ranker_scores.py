"""Our split ranker on fcorrent's token cache (p13, 8 mm tokens on the 512-channel PCA basis): cross-fitted score maps that his
decoder reads (fm/sfc.py `_select8`, args.score_dir) in place of the intensity std.

Per volume of the benchmark's totalseg task (400 train / 100 val / 110 test, as scripts/adaptive_cached_train.py): the 8 mm token
grid [gx,gy,gz,512], 4 voxel statistics of every cell, and the gain of splitting it into its 8 children (class-weighted voxels
that turn right: adaptive_cached.cell_gain; the labels are the privileged information). Ranker = adaptive.Ranker (3x3x3
depthwise blocks over the grid), ListNet loss against the gains. The 400 train volumes are split in `folds` groups by subject;
fold f's model is fitted on the others and scores the held-out ones (so training volumes get scores a test volume would get);
the val and test volumes get the mean of the folds' scores. The epoch of each fold is chosen on the validation volumes (share of
the weighted gain held by the top `quota` of cells). The test volumes are scored, never used to choose anything.

    python scripts/ranker_scores.py --tokens <tok_pre> --out $SCRATCH/ranker_scores/q25 [--quota 0.25] [--oracle-out <dir>]
    (--dummy: random maps for the first volumes of every split, to test the plumbing)
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfc_gdn2.adaptive import Ranker
from sfc_gdn2.adaptive_cached import cell_gain
from sfc_gdn2.apt import entropy
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.data.volume import patchify
from sfc_gdn2.io import load_yaml


def prepare(it, tok, cache, dev, B):
    """-> dict with the 8 mm tokens (fp16, cpu), hand features, gain, grid for one volume."""
    meta = __import__("json").loads((tok / f"{it['id']}.json").read_text())
    k4 = tuple(meta["k4"])
    k8 = tuple(2 * q for q in k4)
    t = torch.from_numpy(np.load(tok / f"{it['id']}.8.p512.npy"))
    g8 = list(t.shape[:3])
    r = torch.load(cache / f"{it['id']}.pt", map_location="cpu", weights_only=False)
    x, y = r["x"][0].float().to(dev), r["y"].to(dev)
    pad = [p for n, q in zip(reversed(x.shape), reversed(k8)) for p in (0, -n % q)]
    xp, yp = F.pad(x, pad), F.pad(y, pad, value=255)
    assert [s // q for s, q in zip(xp.shape, k8)] == g8, (it["id"], xp.shape, k8, g8)
    px = patchify(xp, k8)
    mean = px.mean(1)
    hand = torch.stack([mean, (px - mean[:, None]).pow(2).mean(1).sqrt(), px.amax(1) - px.amin(1), entropy(px, 16)], -1)
    return {"id": it["id"], "g8": g8, "tok": t.flatten(0, 2), "hand": hand.half().cpu(),
            "gain": cell_gain(yp, k4, g8, B).float().cpu()}


def scores_of(model, d, dev):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(d["tok"].to(dev).float(), d["hand"].to(dev).float(), d["g8"]).float().cpu()


def captured(model, vols, dev, quota):
    """Pooled class-weighted gain held by the top `quota` of the cells."""
    sel = tot = 0.0
    for d in vols:
        s, g = scores_of(model, d, dev), d["gain"]
        if g.sum() <= 0:
            continue
        top = s.argsort(descending=True)[: round(quota * len(s))]
        sel, tot = sel + float(g[top].sum()), tot + float(g.sum())
    return sel / max(tot, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oracle-out", help="also write the true gain maps here (the selection upper bound)")
    ap.add_argument("--quota", type=float, default=0.25)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Path(a.tokens)

    bench = load_yaml("configs/bench/bench.yaml")
    tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
    vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(bench["out_dir"]) / "splits.csv")
    items = bs.task_items(tasks, vols, splits, a.seed)["totalseg"]
    cache = Path(bench["segdec"]["cache"]) / "sfc_gdn2"
    split = {sp: [i for i in items if i["split"] == sp and (cache / f"{i['id']}.pt").exists() and (tok / f"{i['id']}.8.p512.npy").exists()]
             for sp in ("train", "val", "test")}
    B = 1 + len(list(Path(split["train"][0]["seg"]).glob("*.nii*")))
    print({s: len(v) for s, v in split.items()}, "classes", B, flush=True)

    if a.dummy:   # random maps with the right shapes for the first volumes of each split
        rng = np.random.default_rng(0)
        for sp, its in split.items():
            for it in its[:10]:
                shape = np.load(tok / f"{it['id']}.8.p512.npy", mmap_mode="r").shape[:3]
                np.save(out / f"{it['id']}.npy", rng.standard_normal(shape).astype(np.float16))
        print("dummy maps written", flush=True)
        return

    data = {sp: [] for sp in split}
    for sp, its in split.items():
        for n, it in enumerate(its):
            data[sp].append(prepare(it, tok, cache, dev, B))
            if (n + 1) % 100 == 0:
                print(f"{sp}: {n + 1} / {len(its)} prepared", flush=True)
    subjects = sorted({i["subject"] for i in split["train"]})
    random.Random(a.seed).shuffle(subjects)
    fold_of = {s: j % a.folds for j, s in enumerate(subjects)}
    fold = [fold_of[i["subject"]] for i in split["train"]]

    heldout, ens = {}, {}
    for f in range(a.folds):
        train = [d for d, fo in zip(data["train"], fold) if fo != f]
        torch.manual_seed(a.seed + f)
        model = Ranker(512).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * (len(train) // a.accum), pct_start=0.1)
        best, rng = (-1.0, None, 0), random.Random(a.seed + f)
        for ep in range(1, a.epochs + 1):
            model.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            for n, i in enumerate(order[: len(order) // a.accum * a.accum]):
                d = train[i]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = model(d["tok"].to(dev).float(), d["hand"].to(dev).float(), d["g8"]).float()
                g = d["gain"].to(dev)
                if g.sum() > 0:
                    (-(g / g.sum() * F.log_softmax(s, 0)).sum() / a.accum).backward()
                if (n + 1) % a.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
            if ep % 5 == 0 or ep == a.epochs:
                model.eval()
                v = captured(model, data["val"], dev, a.quota)
                if v > best[0]:
                    best = (v, {k: t.clone() for k, t in model.state_dict().items()}, ep)
        model.load_state_dict(best[1])
        model.eval()
        print(f"fold {f}: val weighted gain captured @{a.quota:g} = {best[0]:.4f} (epoch {best[2]}; chance {a.quota:g})", flush=True)
        for d, fo in zip(data["train"], fold):
            if fo == f:
                heldout[d["id"]] = scores_of(model, d, dev)
        for sp in ("val", "test"):
            for d in data[sp]:
                s = scores_of(model, d, dev)
                ens[d["id"]] = s if d["id"] not in ens else ens[d["id"]] + s
        del model

    for sp in split:
        for d in data[sp]:
            s = heldout[d["id"]] if sp == "train" else ens[d["id"]] / a.folds
            np.save(out / f"{d['id']}.npy", s.view(*d["g8"]).numpy().astype(np.float16))
            if a.oracle_out:
                Path(a.oracle_out).mkdir(parents=True, exist_ok=True)
                np.save(Path(a.oracle_out) / f"{d['id']}.npy", d["gain"].view(*d["g8"]).numpy().astype(np.float16))
    # diagnostics on the volumes that choose nothing: held-out train folds and the validation ensemble
    ho = [d for d in data["train"]]
    for name, ds, sc in (("train (held-out folds)", ho, heldout), ("val (ensemble)", data["val"], {d["id"]: ens[d["id"]] for d in data["val"]})):
        sel = tot = 0.0
        for d in ds:
            s, g = sc[d["id"]], d["gain"]
            if g.sum() > 0:
                sel, tot = sel + float(g[s.argsort(descending=True)[: round(a.quota * len(s))]].sum()), tot + float(g.sum())
        print(f"{name}: weighted gain captured @{a.quota:g} = {sel / max(tot, 1e-9):.4f}", flush=True)
    print(f"maps -> {out}", flush=True)


if __name__ == "__main__":
    main()
