"""Encoder read in pieces (model.py `flat_chunked`): is it the same function, and what does it cost on the worst tile?

1. equivalence: a real cached TotalSeg tile, logits of SegNet with chunk=0 against chunk=<piece> (and a second chunk=0
   run, the noise of the kernels themselves);
2. the worst tile of the pool (172x172x612 at 4.17x4.17x2.03 mm, ~2e6 tokens at 4 mm, random voxels): time and peak
   memory of the forward (inference tile) and of a training step, patch_mm 4, decode 4, no token limit.

    python scripts/chunk_test.py <run dir> [--piece 8192] [--mm 8] [--skip-extreme]
"""
import argparse
import time
from pathlib import Path

import torch

import sfc_gdn2.fm.sfc as m
from sfc_gdn2.data.volume import patch_dims, patch_grid
from sfc_gdn2.fm.segrun import loss_fn

CACHE = Path("/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/segdec_bench/inputs/sfc_gdn2")


def build(run, mm, decode, chunk):
    w = m.SFC({"checkpoint": run, "args": {"patch_mm": mm, "decode": decode, "chunk": chunk}}, "cuda")
    net = w.seg_net(51, pretrained=True)[0].cuda().eval()
    for n, p in net.named_parameters():
        p.requires_grad_(not n.startswith("encoder."))
    return net


def tile(r):
    x = r["x"].float()
    sl = tuple(slice(max((s - 160) // 2, 0), max((s - 160) // 2, 0) + 160) for s in x.shape[1:])
    x = x[(slice(None), *sl)]
    pad = [q for s in reversed(x.shape[1:]) for q in (0, max(160 - s, 0))]
    ctx = torch.cat([r["ctx"].float(), torch.tensor(r["x"].shape[1:], dtype=torch.float)])
    return torch.nn.functional.pad(x, pad)[None].cuda(), ctx[None].cuda()


def fwd(net, x, ctx):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(x, ctx)
    torch.cuda.synchronize()
    return out.float(), time.time() - t, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--piece", type=int, default=8192)
    ap.add_argument("--mm", type=float, default=8.0)
    ap.add_argument("--skip-extreme", action="store_true")
    a = ap.parse_args()

    files = sorted(CACHE.glob("TotalSegmentator-MRI__*.pt"))
    for f in (files[0], files[len(files) // 2]):
        x, ctx = tile(torch.load(f, map_location="cpu", weights_only=False))
        net = build(a.run, a.mm, 4, 0)
        ref, t0, g0 = fwd(net, x, ctx)
        ref2, _, _ = fwd(net, x, ctx)
        n_tok = int(patch_grid(tuple(min(s, 160) for s in ctx[0, 3:].long().tolist()),
                               patch_dims(ctx[0, :3].tolist(), a.mm)).prod())
        net.encoder.chunk = a.piece
        out, t1, g1 = fwd(net, x, ctx)
        sd = ref.std().item()
        print(f"{f.stem[-26:]} patch {a.mm} mm, {n_tok} tokens, piece {a.piece}: |chunked - whole| max {(out - ref).abs().max():.4f} "
              f"mean {(out - ref).abs().mean():.5f} | |whole - whole(2nd run)| max {(ref2 - ref).abs().max():.4f} | logits std {sd:.3f} "
              f"| whole {t0:.2f}s {g0:.1f}GB, chunked {t1:.2f}s {g1:.1f}GB | finite {bool(torch.isfinite(out).all())}", flush=True)
        del net

    if a.skip_extreme:
        return
    sp, shape = torch.tensor([4.17, 4.17, 2.03]), (172, 172, 612)
    x = torch.rand(1, 1, 160, 160, 160, device="cuda")
    ctx = torch.cat([sp, torch.tensor(shape, dtype=torch.float)])[None].cuda()
    V = tuple(min(s, 160) for s in shape)
    print("worst tile: tokens", int(patch_grid(V, patch_dims(sp.tolist(), 4.0)).prod()), flush=True)
    net = build(a.run, 4, 4, a.piece)
    for rep in range(2):
        try:
            out, t, g = fwd(net, x, ctx)
            print(f"  forward rep {rep}: {t:.1f}s peak {g:.1f} GB finite {bool(torch.isfinite(out).all())}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print("  forward OOM:", str(e).splitlines()[0][:150], flush=True)
            break
    y = torch.randint(0, 51, (1, 160, 160, 160), device="cuda")
    for rep in range(2):
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x, ctx)
            loss_fn(logits, y, 51).backward(); torch.cuda.synchronize()
            net.zero_grad(set_to_none=True)
            print(f"  train step rep {rep}: {time.time() - t:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print("  train step OOM:", str(e).splitlines()[0][:150], flush=True)
            break


if __name__ == "__main__":
    main()
