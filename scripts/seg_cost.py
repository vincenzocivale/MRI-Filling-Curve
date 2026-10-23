"""Cost of the segmentation path (fm/sfc.py SegNet) at several patch_mm on cached TotalSeg-MRI inputs.

Per (patch_mm, scan): tokens of one 160^3-voxel training tile (batch 1), the share of foreground tokens, time of the
patch embedding alone (weights + matmul), of the frozen encoder read, of the decoder, forward + backward step time and
peak GPU memory, forward-only (inference tile) time, and the tiles one scan takes in the sliding window (step 0.5).

    python scripts/seg_cost.py <run dir or encoder_step*.pt> [--mm 16 8 4] [--n 4] [--out file.json]
"""
import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sfc_gdn2.data.volume import patch_dims, patch_grid, patchify
from sfc_gdn2.fm.segrun import loss_fn
import sfc_gdn2.fm.sfc as sfc_mod
from sfc_gdn2.fm.sfc import SEG_PATCH, SFC

CACHE = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/segdec_bench/inputs/sfc_gdn2")
FG = 0.05   # the run's fg_threshold


def sync():
    torch.cuda.synchronize()


def timed(fn, acc: dict, key: str):
    def run(*a, **k):
        sync()
        t = time.time()
        out = fn(*a, **k)
        sync()
        acc[key] = acc.get(key, 0.0) + time.time() - t
        return out
    return run


def tile(r: dict, dev):
    """The training tile of a cached scan: its centre 160^3 voxels (right-padded with 0 / 255 if smaller)."""
    x, y = r["x"].float(), r["y"]
    sl = [slice(max((s - p) // 2, 0), max((s - p) // 2, 0) + p) for s, p in zip(x.shape[1:], SEG_PATCH)]
    x, y = x[(slice(None), *sl)], y[tuple(sl)]
    pad = [q for s, p in zip(reversed(x.shape[1:]), reversed(SEG_PATCH)) for q in (0, max(p - s, 0))]
    ctx = torch.cat([r["ctx"].float(), torch.tensor(r["x"].shape[1:], dtype=torch.float)])
    return F.pad(x, pad).to(dev), F.pad(y, pad, value=255).to(dev), ctx.to(dev)


def tiles_in(shape, patch=SEG_PATCH) -> int:
    n = 1
    for s, p in zip(shape, patch):
        n *= len(np.unique(np.round(np.linspace(0, s - p, math.ceil((s - p) / (p / 2)) + 1)).astype(int))) if s > p else 1
    return n


def measure(net, x, y, ctx, k: int, reps: int = 2) -> dict:
    dev = x.device
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)   # noqa: E731
    res = {}
    # patch embedding alone
    sp, kk = ctx[:3].cpu(), patch_dims(ctx[:3].tolist(), net.patch_mm)
    pe = net.encoder.patch_embed
    xp = patchify(x[0], kk)
    res |= {"tokens": len(xp), "k": list(kk), "fg_frac": float((xp.mean(-1) > FG).float().mean())}
    ts = []
    for _ in range(3):
        sync()
        t = time.time()
        with torch.no_grad():
            pe(xp, pe.weights(sp.to(dev), kk))
        sync()
        ts.append(time.time() - t)
    res["embed_s"] = st.median(ts[1:])
    # train step (decoder trained, encoder frozen) and forward-only, split by stage
    acc = {}
    net.extract = timed(net.extract, acc, "enc")
    dec_in = net.decode.register_forward_pre_hook(lambda m, a: sync() or acc.__setitem__("t0", time.time()))
    dec_out = net.decode.register_forward_hook(lambda m, a, o: sync() or acc.__setitem__("dec", acc.get("dec", 0.0) + time.time() - acc["t0"]))
    try:
        for mode in ("train", "infer"):
            torch.cuda.empty_cache()
            runs = []
            for i in range(1 + reps):   # first: Triton compile / allocator warm-up
                acc.clear()
                torch.cuda.reset_peak_memory_stats()
                sync()
                t = time.time()
                if mode == "train":
                    with amp():
                        logits = net(x[None], ctx[None])
                    loss_fn(logits, y[None], k).backward()
                    net.zero_grad(set_to_none=True)
                else:
                    with torch.no_grad(), amp():
                        o = net(x[None], ctx[None])
                    res["finite"] = bool(torch.isfinite(o).all())
                    del o
                sync()
                runs.append({"step_s": time.time() - t, "enc_s": acc.get("enc", 0.0), "dec_fwd_s": acc.get("dec", 0.0),
                             "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
            res[mode] = {key: st.median(r[key] for r in runs[1:]) for key in runs[0]}
    finally:
        dec_in.remove(), dec_out.remove()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--mm", type=float, nargs="+", default=[16, 8, 4])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--decode", type=int, nargs="+", default=[16], help="PatchDecode cells per patch edge (fm/sfc.py DECODE)")
    ap.add_argument("--out", default="seg_cost.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    files = sorted(CACHE.glob("TotalSegmentator-MRI__*.pt"))
    files = [files[int(i)] for i in np.linspace(0, len(files) - 1, a.n)]
    out = []
    for mm, dc in [(m, d) for m in a.mm for d in a.decode]:
        sfc_mod.DECODE = dc
        w = SFC({"checkpoint": a.checkpoint, "args": {"patch_mm": mm}}, "cuda")
        for f in files:
            r = torch.load(f, map_location="cpu", weights_only=False)
            k = len(r["classes"]) + 1
            net = w.seg_net(k, pretrained=True)[0].to(dev)
            for n, p in net.named_parameters():
                p.requires_grad_(not n.startswith("encoder."))
            net.encoder.eval()
            x, y, ctx = tile(r, dev)
            row = {"patch_mm": mm, "decode": dc, "scan": f.stem, "shape": list(r["x"].shape[1:]), "spacing": [round(v, 2) for v in r["ctx"].tolist()],
                   "classes": k, "tiles": tiles_in(r["x"].shape[1:])}
            try:
                row |= measure(net, x, y, ctx, k)
            except torch.cuda.OutOfMemoryError as e:
                row |= {"error": "OOM", "msg": str(e).splitlines()[0][:120]}
            except Exception as e:   # noqa: BLE001 -- recorded, the other configurations still run
                row |= {"error": type(e).__name__, "msg": str(e)[:200]}
            print(json.dumps(row), flush=True)
            out.append(row)
            del net, x, y
            torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
