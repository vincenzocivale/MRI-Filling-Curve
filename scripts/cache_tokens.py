"""Per-volume cache for the split ranker (train_ranker.py): the frozen encoder's tokens at 8 and 16 mm and everything the
labels say about splitting each 8 mm cell into its 8 children (4 mm). One .pt per TotalSeg-MRI volume of the segdec cache:

  tok8 [N1, 512], tok16 [N0, 512] (fp16)   the encoder read at 8 and 16 mm (pe.patch_mm set to the patch), grids g1 = 2 g0
  hand [N1, 4]                              mean / std / range / entropy of the cell's voxels
  valid1, mixed1 [N1]                       the cell holds voxels / holds >= 2 labels (background included)
  lab1 [N1], lab2mixed [N1], ntok2 [N1]     labelled voxels (y > 0) of the cell; of its mixed children; its valid children
  gain [N1], gain_w [N1]                    voxels (class-weighted: 1/sqrt(class voxels in the scan)) that stop being
                                            outside the majority class when the cell is read as its 8 children
  coo (cell, cls, val)                      the same gain per class, sparse (signed; sums to `gain` over classes)
Splitting costs the same 7 tokens for every cell, so ranking the cells by gain is the best choice for any budget.

    python scripts/cache_tokens.py <run dir> --out $SCRATCH/ranker_cache [--shard 0/2] [--limit N]
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from prune_eval import CACHE, LEVELS, level_stats

import sfc_gdn2.fm.sfc as m
from sfc_gdn2.apt import base_dims
from sfc_gdn2.bench import segdec as bs
from sfc_gdn2.data.volume import patch_grid, patchify
from sfc_gdn2.io import load_yaml


def hist(y, k, B):
    """y [D,H,W] uint8 (255 = no label), cells of k voxels -> [N, B] int32 voxel counts of the labels 0 .. B - 1."""
    p = patchify(y, k).long()
    p = torch.where(p == 255, torch.full_like(p, B), p).clamp_max(B)
    N = p.shape[0]
    idx = (p + torch.arange(N, device=y.device)[:, None] * (B + 1)).flatten()
    h = torch.zeros(N * (B + 1), dtype=torch.int32, device=y.device)
    h.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
    return h.view(N, B + 1)[:, :B]


def block_sum(t, g1):
    """[g2x*g2y*g2z, ...] (cells of the 4 mm grid, raster) -> [N1, ...] sums over each 8 mm cell's 2x2x2 children."""
    t = t.view(g1[0], 2, g1[1], 2, g1[2], 2, *t.shape[1:])
    return t.sum((1, 3, 5)).flatten(0, 2)


def read(net, x, grid_k, sp, mm, dev):
    pe = net.encoder.patch_embed
    pe.patch_mm = float(mm)
    g = patch_grid(tuple(x.shape), grid_k).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bi = net.extract({"patches": [patchify(x, grid_k)], "grid": g[None], "k": torch.tensor([grid_k], device=dev),
                          "spacing": sp[None].float().to(dev)})["bi"][0]
    return bi[: int(g.prod()), : net.encoder.dim].half().cpu(), g.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    dev = torch.device("cuda")
    w = m.SFC({"checkpoint": a.run, "args": {"patch_mm": 8, "chunk": 131072}}, "cuda")
    net = w.seg_net(51, pretrained=True)[0].to(dev).eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bench = load_yaml("configs/bench/bench.yaml")
    tasks = {"totalseg": load_yaml(bench["probe"]["tasks"])["totalseg"]}
    vols = pd.read_csv(Path(bench["out_dir"]) / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(Path(bench["out_dir"]) / "splits.csv")
    items = [i for i in bs.task_items(tasks, vols, splits, 0)["totalseg"] if (CACHE / f"{i['id']}.pt").exists()]
    i0, n = (int(v) for v in a.shard.split("/"))
    items = items[i0::n][: a.limit]
    print(f"{len(items)} volumes", flush=True)
    for j, it in enumerate(items):
        dst = out / f"{it['id']}.pt"
        if dst.exists():
            continue
        r = torch.load(CACHE / f"{it['id']}.pt", map_location="cpu", weights_only=False)
        x, y = r["x"][0].float().to(dev), r["y"].to(dev)
        sp = r["ctx"][:3]
        k0 = base_dims(sp.tolist(), 16.0, LEVELS)
        pad = [p for n_, q in zip(reversed(x.shape), reversed(k0)) for p in (0, -n_ % q)]
        ones = F.pad(torch.ones_like(x), pad)
        x, y = F.pad(x, pad), F.pad(y, pad, value=255)
        st = level_stats(x, y, ones, k0)
        k1, k2 = tuple(q // 2 for q in k0), tuple(q // 4 for q in k0)
        tok8, g1 = read(net, x, k1, sp, 8, dev)
        tok16, g0 = read(net, x, k0, sp, 16, dev)
        B = len(r["classes"]) + 1
        H2 = hist(y, k2, B).float()                                   # 4 mm cells
        H1 = block_sum(H2, g1)                                        # 8 mm cells
        n_cls = H1.sum(0)
        wc = torch.where(n_cls > 0, n_cls.clamp_min(1).rsqrt(), torch.zeros_like(n_cls))

        def err(H, wt=None):   # voxels outside the cell's majority class (weighted by wt per class)
            Hw = H if wt is None else H * wt
            return Hw.sum(1) - Hw.max(1).values

        gain = err(H1) - block_sum(err(H2)[:, None], g1)[:, 0]
        gain_w = err(H1, wc) - block_sum(err(H2, wc)[:, None], g1)[:, 0]
        keep1 = H1.clone().scatter_(1, H1.argmax(1, keepdim=True), 0)   # H1 without its majority class
        keep2 = H2.clone().scatter_(1, H2.argmax(1, keepdim=True), 0)
        per_cls = keep1 - block_sum(keep2, g1)                        # [N1, B], signed
        cell, cls = per_cls.nonzero(as_tuple=True)
        s1, s2 = st[1], st[2]
        torch.save({
            "id": it["id"], "split": it["split"], "g1": g1, "g0": g0, "classes": r["classes"],
            "tok8": tok8, "tok16": tok16,
            "hand": torch.stack([s1["mean"], s1["std"], s1["rng"], s1["ent"]], -1).flatten(0, 2).half().cpu(),
            "valid1": s1["valid"].flatten().cpu(), "mixed1": s1["mixed"].flatten().cpu(),
            "lab1": s1["nlab"].flatten().int().cpu(),
            "lab2mixed": block_sum((s2["nlab"] * s2["mixed"]).flatten()[:, None].int(), g1)[:, 0].cpu(),
            "ntok2": block_sum(s2["valid"].flatten()[:, None].int(), g1)[:, 0].cpu(),
            "gain": gain.cpu(), "gain_w": gain_w.cpu(),
            "coo": (cell.int().cpu(), cls.short().cpu(), per_cls[cell, cls].int().cpu()),
        }, dst.with_suffix(".tmp"))
        dst.with_suffix(".tmp").rename(dst)
        if (j + 1) % 20 == 0:
            print(f"{j + 1} / {len(items)}", flush=True)
        del x, y, H1, H2, per_cls
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
