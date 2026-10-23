"""Token pruning / merging policies judged by the labels, no model: what would each one throw away?

Real TotalSeg-MRI volumes (cached segdec inputs: normalised intensities + labels on the native grid). The finest cell is
~4 mm (apt.base_dims with 3 levels: 16 -> 8 -> 4 mm cells, native voxels, nothing resampled). A policy picks leaves:
- uni16 / uni8 / uni4: the uniform grids;
- fgT: uni4, leaves whose mean intensity <= T dropped (background pruning); fgT+dil: also keep the 3x3x3 neighbours of a
  kept cell (air next to tissue, lungs);
- oracle: uni4, only the cells holding a label (>0): the least a pruning could keep without losing any target;
- oracle-edge: octree splitting a cell iff it holds >= 2 labels (the bound for any boundary-driven criterion);
- ent / std / rng @ F: octree, F = share of the cells still alive that split at each level (apt.partition `frac`), score = APT's
  entropy of the cell's histogram / intensity std / intensity range; rnd@F: the same number of splits chosen at random;
  `+fg`: leaves with mean <= 0.05 dropped as well.
Per policy (volume-pooled): tokens as a share of uni4's, labelled voxels (y > 0) inside kept leaves (`recall`), labelled
voxels in kept leaves that hold >= 2 labels incl. background (`mixed`: a leaf the decoder must split without seeing the
voxels), and for the pruning policies the classes with the lowest recall.

    python scripts/prune_eval.py --n 100 --out prune_eval.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sfc_gdn2.apt import base_dims, entropy, partition
from sfc_gdn2.data.volume import patchify

CACHE = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/segdec_bench/inputs/sfc_gdn2")
LEVELS, BINS, FG = 3, 16, 0.05


def level_stats(x, y, ones, k0):
    """Per level l (cells of k0 / 2^l voxels): mean, std, entropy (l < LEVELS - 1), labelled voxels, mixed, valid: [g_l]."""
    out = []
    for l in range(LEVELS):
        k = tuple(q // 2 ** l for q in k0)
        g = [s // q for s, q in zip(x.shape, k)]
        px, py, pv = patchify(x, k), patchify(y, k).short(), patchify(ones, k)
        mean = px.mean(1)
        s = {"mean": mean.view(g), "std": (px - mean[:, None]).pow(2).mean(1).sqrt().view(g),
             "nlab": ((py > 0) & (py < 255)).sum(1).view(g), "valid": (pv.sum(1) > 0).view(g)}
        lo, hi = py.amin(1), torch.where(py == 255, torch.zeros_like(py), py).amax(1)
        s["mixed"] = ((lo != hi) & (lo < 255)).view(g)
        s["rng"] = (px.amax(1) - px.amin(1)).view(g)
        s["ent"] = entropy(px, BINS).view(g) if l < LEVELS - 1 else None
        out.append(s)
    return out


def flat(c, g):
    return (c[:, 0] * g[1] + c[:, 1]) * g[2] + c[:, 2]


def score(st, name):
    """The per-level score lists `partition` takes (levels 0 .. LEVELS - 2)."""
    return [st[l][name] for l in range(LEVELS - 1)]


def evaluate(leaves, st, y_vox, k0, keep_fn=None):
    """leaves per level -> counts. keep_fn(l, idx, st) -> bool mask of the leaves kept (default all valid ones)."""
    tok = lab = mixed = 0
    mask = []
    for l, c in enumerate(leaves):
        g = st[l]["mean"].shape
        idx = flat(c, g) if len(c) else c.new_zeros(0)
        ok = st[l]["valid"].flatten()[idx]
        if keep_fn is not None:
            ok = ok & keep_fn(l, idx, st)
        tok += int(ok.sum())
        n = st[l]["nlab"].flatten()[idx][ok]
        lab += int(n.sum())
        mixed += int(n[st[l]["mixed"].flatten()[idx][ok]].sum())
        m = torch.zeros(g, dtype=torch.bool, device=c.device)
        m.view(-1)[idx[ok]] = True
        for a in range(3):   # to the finest cell grid
            m = m.repeat_interleave(2 ** (LEVELS - 1 - l), a)
        mask.append(m)
    keep = torch.stack(mask).any(0)
    return tok, lab, mixed, keep


def classes_kept(keep, y, k0, k_classes):
    """Labelled voxels per class inside the kept finest cells."""
    kf = tuple(q // 2 ** (LEVELS - 1) for q in k0)
    v = keep
    for a in range(3):
        v = v.repeat_interleave(kf[a], a)
    y = y[: v.shape[0], : v.shape[1], : v.shape[2]].long()
    return torch.bincount(y[v & (y < 255)], minlength=k_classes + 1)[: k_classes + 1].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="prune_eval.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    files = sorted(CACHE.glob("TotalSegmentator-MRI__*.pt"))
    files = random.Random(0).sample(files, min(a.n, len(files)))
    tot, classes = {}, None
    for i, f in enumerate(files):
        r = torch.load(f, map_location="cpu", weights_only=False)
        classes = classes or r["classes"]
        k_cls = len(r["classes"])
        x, y = r["x"][0].float().to(dev), r["y"].to(dev)
        k0 = base_dims(r["ctx"][:3].tolist(), 16.0, LEVELS)
        pad = [p for n, q in zip(reversed(x.shape), reversed(k0)) for p in (0, -n % q)]
        ones = F.pad(torch.ones_like(x), pad)
        x, y = F.pad(x, pad), F.pad(y, pad, value=255)
        st = level_stats(x, y, ones, k0)
        g0 = list(st[0]["mean"].shape)
        total_lab = int(((y > 0) & (y < 255)).sum())
        if total_lab == 0:
            continue
        gen = torch.Generator(dev).manual_seed(i)
        fg = lambda l, idx, s: s[l]["mean"].flatten()[idx] > FG                      # noqa: E731
        base = {}

        def run(name, arm=None, keep=None, counts=None, leaves=None, dil=False):
            if leaves is None:
                sc = score(st, arm["s"]) if arm.get("s") else None
                lv, split = partition(sc or [torch.zeros_like(st[l]["mean"]) for l in range(LEVELS - 1)], g0, LEVELS,
                                      arm["a"], counts, gen)
            else:
                lv, split = leaves, None
            kf = keep
            if dil:   # keep a 4 mm cell next to a kept one: uni4 only
                m = st[-1]["mean"] > FG
                m = F.max_pool3d(m[None, None].float(), 3, 1, 1)[0, 0] > 0
                kf = lambda l, idx, s: m.flatten()[idx]                           # noqa: E731
            tok, lab, mixed, kept = evaluate(lv, st, None, k0, kf)
            row = tot.setdefault(name, {"tok": 0.0, "lab": 0, "mixed": 0, "all_lab": 0, "vols": 0, "cls": None})
            row["tok"] += tok
            row["lab"] += lab
            row["mixed"] += mixed
            row["all_lab"] += total_lab
            row["vols"] += 1
            if keep is not None or dil or name in ("oracle", "uni4"):
                c = classes_kept(kept, y, k0, k_cls)
                row["cls"] = c if row["cls"] is None else row["cls"] + c
            return split

        uni = lambda t: {"a": {"tau": [t] * (LEVELS - 1)}}                            # noqa: E731
        run("uni16", uni(1e9))
        run("uni8", {"a": {"tau": [-1.0, 1e9]}})
        run("uni4", uni(-1.0))
        run("fg0.05", uni(-1.0), keep=fg)
        run("fg0.05+dil", uni(-1.0), dil=True)
        lab_keep = lambda l, idx, s: s[l]["nlab"].flatten()[idx] > 0                  # noqa: E731
        run("oracle", uni(-1.0), keep=lab_keep)
        # oracle-edge: split a cell exactly when it holds >= 2 labels - the best any boundary-driven criterion can do
        run("oracle-edge", {"s": "mixed", "a": {"tau": [0.5] * (LEVELS - 1)}})
        for frac in (0.3, 0.15):
            for crit in ("ent", "std", "rng"):
                split = run(f"{crit}@{frac}", {"s": crit, "a": {"frac": [frac] * (LEVELS - 1)}})
                run(f"{crit}@{frac}+fg", {"s": crit, "a": {"frac": [frac] * (LEVELS - 1)}}, keep=fg)
                run(f"rnd@{frac}(match {crit})", {"a": {"score": "random", "frac": None, "tau": None}}, counts=split)
        if (i + 1) % 20 == 0:
            print(f"{i + 1} volumes", flush=True)
    uni4 = tot["uni4"]["tok"]
    print(f"\n{'policy':<26}{'tokens/uni4':>12}{'recall':>9}{'mixed':>9}")
    res = {}
    for name, r in tot.items():
        res[name] = {"tok_share": r["tok"] / uni4, "recall": r["lab"] / r["all_lab"], "mixed": r["mixed"] / r["all_lab"],
                     "volumes": r["vols"]}
        if r["cls"] is not None:
            rec = r["cls"][1:].float() / tot["uni4"]["cls"][1:].float().clamp_min(1) if tot["uni4"]["cls"] is not None else None
            res[name]["worst_classes"] = sorted((round(float(v), 3), classes[j]) for j, v in enumerate(rec))[:8]
        print(f"{name:<26}{res[name]['tok_share']:>12.3f}{res[name]['recall']:>9.4f}{res[name]['mixed']:>9.4f}")
    for name in ("fg0.05", "fg0.05+dil"):
        if name in res:
            print(name, "lowest-recall classes:", res[name].get("worst_classes"))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
