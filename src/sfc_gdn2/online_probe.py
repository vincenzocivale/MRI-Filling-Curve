"""Express probes of a pretraining snapshot (final run, 25/10), run beside the training on its GPU 0 (a few minutes,
a few GB): is the run on track, not a benchmark number. Frozen encoder, one curve (raster), no test subject read.

- global, IDH: UCSF-PDGM, tune subjects (bench.probe.tune_subjects), FLAIR + T1c; the mean token over the foreground
  cells at 16 mm of blocks 4 / 8 / 12 / 16 -> bench.probe.tune_task (subject folds, the candidate chosen on one half
  of the subjects and scored on the other: the screens' protocol, on a smaller menu) -> skill (permutation 0, as the
  screens). Only 35 of the 163 subjects are IDH-mutant (+-0.08 skill per point, 25/10), so also the skill of one
  fixed head (FIXED: no selection jitter, read from the same fit: no cost), the curve to read. Over 10 subject
  permutations it moved by +-0.018 only (step 1000), so one permutation is enough. The features are saved beside <out>
  (<out>.idh.pt), and reused if there.
- local, WMH: WMH2017 FLAIR, the segdec split of seed 0 (train 43 / val 17 volumes). Per level (16 / 8 mm) and per
  block, a ridge (closed form) from each cell's token to its lesion fraction on a 4^3 sub-grid of the cell, resized
  trilinearly to the voxels -> val Dice (mean per volume, class 2 and the ignore label left out) at the best
  threshold and ridge strength on val. A real segmentation score, not a cell AUROC.

UCSF inputs are preprocessed once into <cache> (the run's preprocessing: canonical axes, native voxels, scaled).
    python -m sfc_gdn2.online_probe <encoder_step*.pt> <out.json> [--tasks wmh idh] [--mm 16 8] [--cache dir]
        [--idh-mm 16] [--idh-part I N]   # IDH at a finer grain; part: features of volumes I::N, then a run without
                                         # --idh-part merges the N parts and fits
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

MV = Path("/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2")
SEG_INPUTS = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/segdec_bench/inputs/sfc_gdn2")
CACHE = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/online_probe/ucsf")
TAPS = (1, 3, 5, 7, 9, 11, 13)   # + the last block (15): every other block for WMH; IDH reads 3 / 7 / 11 / 15
IDH_BLOCKS = (3, 7, 11, 15)
FIXED = {"head": "linear", "feature": "mean_b12", "l2": 1.0}   # chosen at 2 of the 3 first snapshots (25/10)
SUB = 4                          # sub-grid per cell edge (16 mm: 4 mm, 8 mm: 2 mm)
ALPHAS = (1e-3, 1e-2, 1e-1)      # ridge strength, relative to the mean diagonal of X^T X
THRESH = torch.linspace(0.02, 0.6, 30)


def _tokens(enc, seed, x, sp, mm):
    """[1, *S] voxels -> [cells, blocks, d] tokens at `mm` (rows: last ++ taps), the grid, k and the voxels read
    (ablation B: SFC_RESPACE resamples them to 1 mm or calls them 1 mm, data.volume.respace)."""
    from .data.volume import patch_dims, patch_grid, respace
    from .fm.token_cache import read

    x, s = respace(x, sp)
    sp = sp.new_tensor(s)
    k = patch_dims(s, mm)
    a = read(enc, ["raster"], seed, x, sp, k, mm, 30_000, 32, 8, taps=True, want=["raster"])["raster"]
    g = patch_grid(tuple(x.shape[1:]), k).tolist()
    # on the CPU, moved one block at a time (2 mm: up to ~1.6M cells x 8 blocks, ~26 GB in fp16)
    return torch.from_numpy(a).reshape(-1, 1 + len(enc.taps), enc.dim), g, k, x


def _block(X, bi):
    """Block bi of the CPU tokens on the GPU, fp32, with a bias column: [cells, d+1]."""
    x = X[:, bi].cuda().float()
    return torch.cat([x, x.new_ones(len(x), 1)], -1)


def _pad(v, g, k):
    return F.pad(v, [p for n, gg, kk in zip(reversed(v.shape), reversed(g), reversed(k)) for p in (0, gg * kk - n)])


POS = {"wmh_seg": lambda y: y == 1, "cow_seg": lambda y: (y > 0) & (y != 255),   # Willis: any vessel class
       "prostate_lesion": lambda y: (y > 0) & (y != 255), "heart_seg": lambda y: (y > 0) & (y != 255)}   # heart: any class
KEEP = {"wmh_seg": lambda y: (y != 255) & (y != 2), "cow_seg": lambda y: y != 255, "prostate_lesion": lambda y: y != 255, "heart_seg": lambda y: y != 255}


def wmh(enc, seed, mms, limit: int = 0, task: str = "wmh_seg") -> dict:
    from .bench.segdec import task_items
    from .io import load_yaml

    vols = pd.read_csv(MV / "volumes.csv", low_memory=False, dtype={"session": str})
    tasks = load_yaml("configs/bench/probe_tasks_v2.yaml")
    items = task_items({task: tasks[task]}, vols, pd.read_csv(MV / "splits.csv"), 0)[task]
    ids = {s: sorted(i["id"] for i in items if i["split"] == s and (task != "wmh_seg" or "FLAIR" in i["id"]))
           for s in ("train", "val")}
    ids = {s: v[:limit] if limit else v for s, v in ids.items()}   # limit: a smoke test on a few volumes
    # ablation B (02/11): SFC_PROBE_LAST=1 reads the last block only; SFC_PROBE_SUBMM=<mm> sets the output sub-grid of each
    # cell to that voxel size (default: 4 per edge, i.e. a quarter of the cell)
    order = [15] if os.environ.get("SFC_PROBE_LAST") else [15, *TAPS]
    nb, out = len(order), {}
    load = lambda i: torch.load(SEG_INPUTS / f"{i}.pt", map_location="cpu", weights_only=False)
    for mm in mms:
        t0, xtx = time.time(), None
        SB = max(1, round(mm / float(os.environ["SFC_PROBE_SUBMM"]))) if os.environ.get("SFC_PROBE_SUBMM") else SUB
        for i in ids["train"]:   # pass 1: X^T X and X^T Y per block, nothing stored
            r = load(i)
            x, sp, y = r["x"].cuda().float(), r["ctx"][:3].float().cuda(), r["y"].cuda()
            X, g, k, xr = _tokens(enc, seed, x, sp, mm)
            t = POS[task](y).float()
            if xr.shape[1:] != y.shape:   # resampled input: the native target straight onto the sub-grid (no 1 mm step)
                t = F.interpolate(t[None, None], size=[round(n * SB / kk) for n, kk in zip(xr.shape[1:], k)],
                                  mode="trilinear", align_corners=False)[0, 0]
                T = _pad(t, g, [SB] * 3)
            else:
                T = F.adaptive_avg_pool3d(_pad(t, g, k)[None, None], [gg * SB for gg in g])[0, 0]
            T = T.reshape(g[0], SB, g[1], SB, g[2], SB).permute(0, 2, 4, 1, 3, 5).reshape(-1, SB ** 3)
            if xtx is None:
                xtx = torch.zeros(nb, X.shape[-1] + 1, X.shape[-1] + 1, device="cuda")
                xty = torch.zeros(nb, X.shape[-1] + 1, SB ** 3, device="cuda")
            for bi in range(nb):
                Xb = _block(X, bi)
                xtx[bi] += Xb.T @ Xb
                xty[bi] += Xb.T @ T
        eye = torch.eye(xtx.shape[-1], device="cuda")
        scale = xtx.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        W = torch.stack([torch.linalg.solve(xtx + al * scale * eye, xty) for al in ALPHAS], 1)   # [nb, alphas, d+1, S^3]
        dice = torch.zeros(nb, len(ALPHAS), len(THRESH), device="cuda")
        th = THRESH.cuda()[:, None]
        def vol_dice(x, sp, y, mm=mm, dice=dice, W=W, SB=SB, th=th):   # Dice of one val volume per (block, alpha, threshold), on the native voxels of y
            X, g, k, xr = _tokens(enc, seed, x, sp, mm)
            keep = KEEP[task](y)
            gt = POS[task](y)[keep]
            dv = torch.zeros_like(dice)
            for bi in range(nb):
                Xb = _block(X, bi)
                for ai in range(len(ALPHAS)):
                    p = (Xb @ W[bi, ai]).reshape(g[0], g[1], g[2], SB, SB, SB).permute(0, 3, 1, 4, 2, 5).reshape(*(gg * SB for gg in g))
                    if xr.shape[1:] == y.shape:
                        p = F.interpolate(p[None, None], size=[gg * kk for gg, kk in zip(g, k)], mode="trilinear",
                                          align_corners=False)[0, 0][: y.shape[0], : y.shape[1], : y.shape[2]]
                    else:   # resampled or thickened input: from the sub-grid straight to the native voxels (Dice there)
                        e = [round(n * SB / kk) for n, kk in zip(xr.shape[1:], k)]
                        p = F.interpolate(p[: e[0], : e[1], : e[2]][None, None], size=list(y.shape), mode="trilinear",
                                          align_corners=False)[0, 0]
                    p = p[keep]
                    pos = p[None] > th
                    tp = (pos & gt).sum(1).float()
                    dv[bi, ai] = 2 * tp / (pos.sum(1) + gt.sum()).clamp_min(1)
            return dv
        dvs = []   # per val volume [blocks, alphas, thresholds]: bootstrap intervals downstream (02/11)
        for i in ids["val"]:     # pass 2: predicted per volume, Dice per (block, alpha, threshold)
            r = load(i)
            dv = vol_dice(r["x"].cuda().float(), r["ctx"][:3].float().cuda(), r["y"].cuda())
            dice += dv
            dvs.append(dv.flatten().cpu())
        dice /= len(ids["val"])
        ib = int(dice.flatten().argmax())   # the val-best (block, alpha, threshold): its Dice per volume
        best = dice.flatten(1).amax(1)   # per block, over ridge strength and threshold (chosen on val)
        out[str(int(mm))] = {"best": round(float(best.max()), 4), "block": order[int(best.argmax())],
                             "last": round(float(best[0]), 4),
                             "per_block": {str(b): round(float(v), 4) for b, v in zip(order, best)},
                             "vols": [round(float(d[ib]), 4) for d in dvs], "ids": ids["val"]}
        for f in [float(v) for v in os.environ.get("SFC_PROBE_THICK", "").split(",") if v]:
            # (02/11) the same head (block, ridge, threshold chosen above) on val slices made f times thicker: the thickest axis
            # averaged (a thicker acquisition, nothing interpolated), Dice on the original mask
            vt = []
            for i in ids["val"]:
                r = load(i)
                x, sp, y = r["x"].cuda().float(), r["ctx"][:3].float().cuda(), r["y"].cuda()
                a = int(sp.argmax()); size = list(x.shape[1:]); size[a] = max(1, round(size[a] / f))
                sp2 = sp.clone(); sp2[a] = sp[a] * x.shape[1 + a] / size[a]
                vt.append(float(vol_dice(F.adaptive_avg_pool3d(x[None], size)[0], sp2, y).flatten()[ib]))
            out[f"{int(mm)}_t{f:g}"] = {"best": round(sum(vt) / len(vt), 4), "vols": [round(v, 4) for v in vt]}
            print(f"[online] {task} {mm:g} mm, slices x{f:g}: Dice {sum(vt) / len(vt):.3f} (same head)", flush=True)
        print(f"[online] {task} {mm:g} mm: Dice best {out[str(int(mm))]['best']:.3f} (block {out[str(int(mm))]['block']}), "
              f"last block {out[str(int(mm))]['last']:.3f}, {time.time() - t0:.0f} s", flush=True)
    out["volumes"] = {s: len(v) for s, v in ids.items()}
    return out


def idh(w, enc, seed, cache: Path, save: Path, mm: float = 16.0, part: tuple[int, int] | None = None) -> dict | None:
    from .bench.probe import task_rows, tune_subjects, tune_task
    from .io import load_yaml

    vols = pd.read_csv(MV / "volumes.csv", low_memory=False, dtype={"session": str})
    splits = pd.read_csv(MV / "splits.csv")
    cfg = load_yaml("configs/bench/probe_tasks_v2.yaml")["idh"]
    rows = task_rows(vols[vols["subject"].isin(tune_subjects(splits))], splits, cfg, 0)
    rows = rows[rows["modality"].isin(["FLAIR", "T1c"])].reset_index(drop=True)
    cache.mkdir(parents=True, exist_ok=True)
    order = [15, *TAPS]
    cols = [order.index(b) for b in IDH_BLOCKS]
    parts = sorted(save.parent.glob(f"{save.stem}.part*.pt"))
    if not save.exists() and part is None and parts and len(parts) == torch.load(parts[0])["of"]:   # merge the parts
        r = [torch.load(f) for f in parts]
        F_ = torch.zeros(len(rows), len(IDH_BLOCKS), r[0]["feats"].shape[-1], dtype=torch.half)
        for q in r:
            assert q["ids"] == list(rows["id"].iloc[q["idx"]]), save
            F_[q["idx"]] = q["feats"]
        torch.save({"ids": list(rows["id"]), "blocks": IDH_BLOCKS, "feats": F_}, save)
    sel = range(len(rows)) if part is None else range(part[0], len(rows), part[1])
    feats = []
    for i, path in zip(rows["id"].iloc[list(sel)], rows["input"].iloc[list(sel)]) if not save.exists() else ():
        f = cache / f"{i}.pt"
        if not f.exists():   # the run's preprocessing, once (written to a temporary name, then renamed)
            p = w.preprocess(path)
            a, c = p["value"]
            torch.save({"x": (p["volume"].float() * a + c)[None].half(), "sp": p["spacing"]}, f.with_suffix(".tmp"))
            f.with_suffix(".tmp").rename(f)
        r = torch.load(f, map_location="cpu")
        x, sp = r["x"].cuda().float(), r["sp"].float().cuda()
        X, g, k, xr = _tokens(enc, seed, x, sp, mm)
        fg = F.avg_pool3d(_pad(xr[0], g, k)[None, None], k)[0, 0].flatten() > 0.05
        feats.append(X[fg.cpu() if fg.any() else slice(None)][:, cols].float().mean(0).cpu())
    if part is not None:   # one share of the volumes (several GPUs): features only, the fit runs on the merge
        torch.save({"ids": list(rows["id"].iloc[list(sel)]), "idx": list(sel), "of": part[1],
                    "feats": torch.stack(feats).half()}, save.with_suffix(f".part{part[0]}.pt"))
        return None
    if feats:
        F_ = torch.stack(feats)   # [volumes, blocks, d]
        torch.save({"ids": list(rows["id"]), "blocks": IDH_BLOCKS, "feats": F_.half()}, save)
    else:
        r = torch.load(save)
        assert r["ids"] == list(rows["id"]) and tuple(r["blocks"]) == IDH_BLOCKS, save
        F_ = r["feats"].float()
    menu = {("mean" if b == 15 else f"mean_b{b + 1}"): F_[:, j] for j, b in enumerate(IDH_BLOCKS)}
    with torch.enable_grad():   # the probe heads are fitted by L-BFGS
        r = tune_task("idh", cfg, rows, menu, {i: n for n, i in enumerate(rows["id"])}, "cuda", keep=True)
    fix = sum(next(c for c in r["cands"] if all(c[k] == v for k, v in FIXED.items()))["skill"]) / 2
    print(f"[online] IDH skill {r['skill']:.3f}, fixed head {fix:.3f} ({r['subjects']} subjects, {len(rows)} volumes)",
          flush=True)
    return {"skill": round(float(r["skill"]), 4), "fixed": FIXED | {"skill": round(float(fix), 4)},
            "subjects": r["subjects"], "volumes": len(rows), "chosen": r["chosen"]}


def main() -> None:
    from .fm.api import load_config
    from .fm.sfc import SFC

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--tasks", nargs="+", default=["wmh", "idh"])
    ap.add_argument("--mm", nargs="+", type=float, default=[16.0, 8.0])
    ap.add_argument("--idh-mm", type=float, default=16.0)
    ap.add_argument("--idh-part", nargs=2, type=int, metavar=("I", "N"))   # features of volumes I::N only
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--limit", type=int, default=0)   # WMH volumes per split (smoke test)
    ap.add_argument("--mem-frac", type=float, default=0.15)   # beside the training: fail the probe, never the training
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    cfg = load_config("configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"], cfg["args"] = a.ckpt, {"curves": ["raster"]}
    w = SFC(cfg, "cuda")
    enc = w._encoder().cuda().eval().requires_grad_(False)
    enc.taps, enc.depth = (() if os.environ.get("SFC_PROBE_LAST") else TAPS), None   # last block only: no taps (Small has 8 blocks)
    seed = int(w.run_cfg["seed"])
    res = {"ckpt": str(a.ckpt), "step": int(Path(a.ckpt).stem.split("step")[-1]) if "step" in Path(a.ckpt).stem else None}
    with torch.no_grad():
        if "wmh" in a.tasks:
            res["wmh"] = wmh(enc, seed, a.mm, a.limit)
        if "cow" in a.tasks:   # ablation B (02/11): Willis, vessel vs not, same ridge probe as WMH
            res["cow"] = wmh(enc, seed, a.mm, a.limit, "cow_seg")
        if "prostate" in a.tasks:   # (02/11) prostate lesions (ADC), same probe
            res["prostate"] = wmh(enc, seed, a.mm, a.limit, "prostate_lesion")
        if "heart" in a.tasks:   # (02/11, W6) HVSMR, heart (any class) vs not, same probe
            res["heart"] = wmh(enc, seed, a.mm, a.limit, "heart_seg")
        if "idh" in a.tasks:
            res["idh"] = idh(w, enc, seed, Path(a.cache), Path(a.out).with_suffix(".idh.pt"), a.idh_mm,
                             a.idh_part and tuple(a.idh_part))
            if a.idh_part:
                print(f"[online] IDH part {a.idh_part[0]} of {a.idh_part[1]}: {time.time() - t0:.0f} s", flush=True)
                return
    res["seconds"] = round(time.time() - t0)
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[online] {a.out}: {res['seconds']} s, peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
