"""flat_chunked against flat on the encoder alone, same tokens: bf16 autocast (as in use) and fp32 (no autocast).
A correct carry of the states gives ~1e-6 relative in fp32; the bf16 gap is rounding."""
import sys

import torch

import sfc_gdn2.fm.sfc as m

w = m.SFC({"checkpoint": sys.argv[1], "args": {"patch_mm": 8}}, "cuda")
enc = w._encoder().cuda().eval()
torch.manual_seed(0)
x = torch.randn(3000, enc.dim, device="cuda")
cu = torch.tensor([0, len(x)], device="cuda", dtype=torch.int32)
for name, ac in (("bf16 autocast", True), ("fp32", False)):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        ref = enc.flat(x, cu).float()
        ref2 = enc.flat(x, cu).float()
        res = {}
        for ch in (256, 1000):
            res[ch] = enc.flat_chunked(x, ch).float()
    s = ref.abs().mean().item()
    print(f"{name}: |whole-whole2| {(ref2 - ref).abs().max():.2e}", *(
        f"| chunk {ch}: max {(o - ref).abs().max():.2e} mean {(o - ref).abs().mean():.2e} (mean|h| {s:.3f})" for ch, o in res.items()), flush=True)
