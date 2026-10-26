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
  threshold and ridge strength on val. A real segmentation score, not a cell AUROC (fm/block_plan).

UCSF inputs are preprocessed once into <cache> (the run's preprocessing: canonical axes, native voxels, scaled).
    python -m sfc_gdn2.online_probe <encoder_step*.pt> <out.json> [--tasks wmh idh] [--mm 16 8] [--cache dir]
        [--idh-mm 16] [--idh-part I N]   # IDH at a finer grain; part: features of volumes I::N, then a run without
                                         # --idh-part merges the N parts and fits
"""
from __future__ import annotations

import argparse
import json
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
    """[1, *S] voxels -> [cells, blocks, d] tokens at `mm` (rows: last ++ taps), the grid and k."""
    from .data.volume import patch_dims, patch_grid
    from .fm.token_cache import read

    k = patch_dims(sp.tolist(), mm)
    a = read(enc, ["raster"], seed, x, sp, k, mm, 30_000, 32, 8, taps=True, want=["raster"])["raster"]
    g = patch_grid(tuple(x.shape[1:]), k).tolist()
    return torch.from_numpy(a).cuda().reshape(-1, 1 + len(enc.taps), enc.dim), g, k


def _pad(v, g, k):
    return F.pad(v, [p for n, gg, kk in zip(reversed(v.shape), reversed(g), reversed(k)) for p in (0, gg * kk - n)])


def wmh(enc, seed, mms) -> dict:
    from .bench.segdec import task_items
    from .io import load_yaml

    vols = pd.read_csv(MV / "volumes.csv", low_memory=False, dtype={"session": str})
    tasks = load_yaml("configs/bench/probe_tasks_v2.yaml")
    items = task_items({"wmh_seg": tasks["wmh_seg"]}, vols, pd.read_csv(MV / "splits.csv"), 0)["wmh_seg"]
    ids = {s: sorted(i["id"] for i in items if i["split"] == s and "FLAIR" in i["id"]) for s in ("train", "val")}
    order = [15, *TAPS]
    nb, out = len(order), {}
    load = lambda i: torch.load(SEG_INPUTS / f"{i}.pt", map_location="cpu", weights_only=False)
    for mm in mms:
        xtx = xty = None
        for i in ids["train"]:   # pass 1: X^T X and X^T Y per block, nothing stored
            r = load(i)
            x, sp, y = r["x"].cuda().float(), r["ctx"][:3].float().cuda(), r["y"].cuda()
            X, g, k = _tokens(enc, seed, x, sp, mm)
            T = F.adaptive_avg_pool3d(_pad((y == 1).float(), g, k)[None, None], [gg * SUB for gg in g])[0, 0]
            T = T.reshape(g[0], SUB, g[1], SUB, g[2], SUB).permute(0, 2, 4, 1, 3, 5).reshape(-1, SUB ** 3)
            Xb = torch.cat([X.float(), X.new_ones(len(X), nb, 1).float()], -1).transpose(0, 1)   # [nb, cells, d+1]
            a, b = Xb.transpose(1, 2) @ Xb, Xb.transpose(1, 2) @ T
            xtx, xty = (a, b) if xtx is None else (xtx + a, xty + b)
        eye = torch.eye(xtx.shape[-1], device="cuda")
        scale = xtx.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None]
        W = torch.stack([torch.linalg.solve(xtx + al * scale * eye, xty) for al in ALPHAS], 1)   # [nb, alphas, d+1, S^3]
        dice = torch.zeros(nb, len(ALPHAS), len(THRESH), device="cuda")
        th = THRESH.cuda()[:, None]
        for i in ids["val"]:     # pass 2: predicted per volume, Dice per (block, alpha, threshold)
            r = load(i)
            x, sp, y = r["x"].cuda().float(), r["ctx"][:3].float().cuda(), r["y"].cuda()
            X, g, k = _tokens(enc, seed, x, sp, mm)
            Xb = torch.cat([X.float(), X.new_ones(len(X), nb, 1).float()], -1).transpose(0, 1)
            keep = (y != 255) & (y != 2)
            gt = (y == 1)[keep]
            for bi in range(nb):
                for ai in range(len(ALPHAS)):
                    p = (Xb[bi] @ W[bi, ai]).reshape(g[0], g[1], g[2], SUB, SUB, SUB).permute(0, 3, 1, 4, 2, 5)
                    p = F.interpolate(p.reshape(1, 1, *(gg * SUB for gg in g)), size=[gg * kk for gg, kk in zip(g, k)],
                                      mode="trilinear", align_corners=False)[0, 0]
                    p = p[: y.shape[0], : y.shape[1], : y.shape[2]][keep]
                    pos = p[None] > th
                    tp = (pos & gt).sum(1).float()
                    dice[bi, ai] += 2 * tp / (pos.sum(1) + gt.sum()).clamp_min(1)
        dice /= len(ids["val"])
        best = dice.flatten(1).amax(1)   # per block, over ridge strength and threshold (chosen on val)
        out[str(int(mm))] = {"best": round(float(best.max()), 4), "block": order[int(best.argmax())],
                             "last": round(float(best[0]), 4),
                             "per_block": {str(b): round(float(v), 4) for b, v in zip(order, best)}}
        print(f"[online] WMH {mm:g} mm: Dice best {out[str(int(mm))]['best']:.3f} (block {out[str(int(mm))]['block']}), "
              f"last block {out[str(int(mm))]['last']:.3f}", flush=True)
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
        X, g, k = _tokens(enc, seed, x, sp, mm)
        fg = F.avg_pool3d(_pad(x[0], g, k)[None, None], k)[0, 0].flatten() > 0.05
        feats.append(X[fg if fg.any() else slice(None)][:, cols].float().mean(0).cpu())
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
    ap.add_argument("--mem-frac", type=float, default=0.15)   # beside the training: fail the probe, never the training
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    cfg = load_config("configs/fm/sfc_gdn2.yaml")
    cfg["checkpoint"], cfg["args"] = a.ckpt, {"curves": ["raster"]}
    w = SFC(cfg, "cuda")
    enc = w._encoder().cuda().eval().requires_grad_(False)
    enc.taps, enc.depth = TAPS, None
    seed = int(w.run_cfg["seed"])
    res = {"ckpt": str(a.ckpt), "step": int(Path(a.ckpt).stem.split("step")[-1]) if "step" in Path(a.ckpt).stem else None}
    with torch.no_grad():
        if "wmh" in a.tasks:
            res["wmh"] = wmh(enc, seed, a.mm)
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
