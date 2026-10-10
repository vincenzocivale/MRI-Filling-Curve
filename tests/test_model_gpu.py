"""Needs CUDA and the official GatedDeltaNet-2 (skipped otherwise)."""
import pytest
import torch

pytest.importorskip("lit_gpt.gdn2")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sfc_gdn2.lejepa import LeJEPA
from sfc_gdn2.model import Encoder

DEV = torch.device("cuda")


SPACING, K = torch.tensor([4.0, 4.0, 4.0], device=DEV), (4, 4, 4)     # 64-voxel patches


def encoder():
    return Encoder(d_model=64, depth=2, head_dim=32, num_heads=2).to(DEV)


def run(enc, x, mask=None):
    t = enc.tokens(x, enc.patch_embed.weights(SPACING, K), mask)
    return enc.packed(t, torch.ones(x.shape[:2], dtype=torch.bool, device=DEV))


def vols(xs, grid):
    return {"patches": xs, "grid": grid, "k": torch.tensor([K] * len(xs), device=DEV),
            "spacing": SPACING.expand(len(xs), 3)}


def test_encoder_is_causal():
    torch.manual_seed(0)
    enc = encoder().eval()
    x = torch.rand(2, 512, 64, device=DEV)
    y = x.clone()
    y[:, 300:] = torch.rand_like(y[:, 300:])              # change only the future of position 300
    with torch.no_grad():
        a, b = run(enc, x), run(enc, y)
    assert torch.allclose(a[:, :300], b[:, :300], atol=1e-4)
    assert not torch.allclose(a[:, 300:], b[:, 300:], atol=1e-4)


def test_packed_sequences_do_not_mix():
    """cu_seqlens packing (used for the views) must equal running each sequence on its own."""
    torch.manual_seed(0)
    enc, lens = encoder(), [300, 37, 129]
    xs = [torch.randn(1, n, 64, device=DEV) for n in lens]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0)], device=DEV, dtype=torch.int32)
    with torch.no_grad():
        packed = enc.run(torch.cat(xs, 1), cu)
        alone = torch.cat([enc.run(x) for x in xs], 1)
    assert torch.allclose(packed, alone, atol=1e-3)


def test_masked_patch_content_is_invisible():
    torch.manual_seed(0)
    enc = encoder().eval()
    mask = torch.zeros(512, dtype=torch.bool, device=DEV)
    mask[400] = True
    x = torch.rand(1, 512, 64, device=DEV)
    y = x.clone()
    y[:, 400] = 5.0
    with torch.no_grad():
        assert torch.equal(run(enc, x, mask), run(enc, y, mask))


def test_lejepa_step_trains():
    torch.manual_seed(0)
    model = LeJEPA(encoder(), ["hilbert", "raster"], local_edge=(2, 4), thick_prob=0.5, proj_hidden=128,
                   proj_dim=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator(DEV).manual_seed(0)
    grid = torch.tensor([[8, 8, 8], [5, 12, 6], [8, 8, 8], [3, 9, 18]], device=DEV)
    xs = [torch.rand(int(n), 64, device=DEV) + 0.1 for n in grid.prod(1)]
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(vols(xs, grid), g)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        assert torch.isfinite(out["loss"])
    assert all(p.grad is not None for p in model.parameters())


def test_bidirectional_patch_feature_sees_both_sides():
    from sfc_gdn2.probe import encoder_extractor
    torch.manual_seed(0)
    from sfc_gdn2.curves import grid_coords, keys
    enc, grid = encoder().eval(), torch.tensor([[8, 8, 8]], device=DEV)
    ext = encoder_extractor(enc, ["hilbert"], "patch", 17, 0.05)
    perm = keys("hilbert", grid_coords(grid, 512)[0], grid[0]).argsort()
    x = torch.rand(1, 512, 64, device=DEV)
    y = x.clone()
    p = perm[300].item()
    y[:, perm[400]] = 5.0                          # the future of patch p in the forward order
    with torch.no_grad():
        a, b = ext(vols(list(x), grid)), ext(vols(list(y), grid))
    d = enc.dim
    assert torch.allclose(a["bi"][:, p, :d], b["bi"][:, p, :d], atol=1e-4)  # forward half, causal: unchanged
    assert not torch.allclose(a["bi"][:, p, d:], b["bi"][:, p, d:], atol=1e-4)  # backward half sees it


def test_recomputed_embedding_matches_direct_autograd():
    """A scan's token chunk through checkpoint (recomputed in backward, noise from its seed) = plain autograd."""
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(0)
    model = LeJEPA(encoder(), ["hilbert"], thick_prob=1.0, proj_hidden=64, proj_dim=16).to(DEV)
    x, pid, vid = torch.rand(50, 64, device=DEV), torch.arange(10, 40, device=DEV), torch.arange(30, device=DEV) % 2
    aug = (torch.tensor([1.2, 0.8], device=DEV), torch.tensor([0.9, 1.1], device=DEV), torch.tensor([0.03, -0.02], device=DEV),
           torch.tensor([0.02, 0.01], device=DEV), torch.tensor([True, False], device=DEV), torch.tensor([2, 0], device=DEV),
           torch.tensor([8.0, 8.0], device=DEV))
    grads = []
    for ckpt in (False, True):
        w = model.encoder.patch_embed.weights(SPACING, K)
        args = (x, pid, vid, w, aug, list(K), SPACING.tolist(), 123)
        t = checkpoint(model._embed, *args, use_reentrant=False) if ckpt else model._embed(*args)
        model.zero_grad()
        (t.square().sum()).backward()
        grads.append((t.detach(), model.encoder.patch_embed.G.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0]) and torch.allclose(grads[0][1], grads[1][1], rtol=1e-5, atol=1e-8)
