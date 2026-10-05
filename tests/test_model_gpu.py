"""Needs CUDA and the official GatedDeltaNet-2 (skipped otherwise)."""
import pytest
import torch

pytest.importorskip("lit_gpt.gdn2")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sfc_gdn2.curves import CurveViews
from sfc_gdn2.lejepa import LeJEPA
from sfc_gdn2.model import Encoder

DEV = torch.device("cuda")


def encoder(grid=8):
    return Encoder(64, grid, d_model=64, depth=2, head_dim=32, num_heads=2).to(DEV)


def test_encoder_is_causal_in_curve_order():
    torch.manual_seed(0)
    enc, perm = encoder().eval(), CurveViews("hilbert", 8).perms[3].to(DEV)
    x = torch.rand(2, 512, 64, device=DEV)
    y = x.clone()
    y[:, perm[300:]] = torch.rand_like(y[:, perm[300:]])  # change only the future of position 300
    with torch.no_grad():
        a, b = enc(x, perm), enc(y, perm)
    assert torch.allclose(a[:, :300], b[:, :300], atol=1e-4)
    assert not torch.allclose(a[:, 300:], b[:, 300:], atol=1e-4)


def test_masked_patch_content_is_invisible():
    torch.manual_seed(0)
    enc, perm = encoder().eval(), CurveViews("raster", 8).perms[0].to(DEV)
    mask = torch.zeros(512, dtype=torch.bool, device=DEV)
    mask[400] = True
    x = torch.rand(1, 512, 64, device=DEV)
    y = x.clone()
    y[:, 400] = 5.0
    with torch.no_grad():
        assert torch.equal(enc(x, perm, mask), enc(y, perm, mask))


def test_lejepa_step_trains():
    torch.manual_seed(0)
    model = LeJEPA(encoder(), ["hilbert", "raster"], local_edge=(2, 4), proj_hidden=128, proj_dim=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator(DEV).manual_seed(0)
    x = torch.rand(4, 512, 64, device=DEV) + 0.1
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, g)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        assert torch.isfinite(out["loss"])
    assert all(p.grad is not None for p in model.parameters())


def test_bidirectional_patch_feature_sees_both_sides():
    from sfc_gdn2.probe import encoder_extractor
    torch.manual_seed(0)
    enc, perm = encoder().eval(), CurveViews("hilbert", 8).perms[0].to(DEV)
    ext = encoder_extractor(enc, perm, "patch")
    x = torch.rand(1, 512, 64, device=DEV)
    y = x.clone()
    p = perm[300].item()
    y[:, perm[400]] = 5.0                          # the future of patch p in the forward order
    with torch.no_grad():
        a, b = ext(x), ext(y)
    assert torch.allclose(a["token"][:, p], b["token"][:, p], atol=1e-4)    # causal: unchanged
    assert not torch.allclose(a["bi"][:, p], b["bi"][:, p], atol=1e-4)      # backward half sees it
