"""The continuous-kernel patch embedding: same physical content -> same token, whatever the spacing."""
import torch

from sfc_gdn2.model import KernelEmbed


def emb():
    torch.manual_seed(0)
    return KernelEmbed(32, rank=8, hidden=32)


def tok(e, values, spacing, k):
    return e(values.reshape(1, -1), e.weights(torch.tensor(spacing), k))[0]


def test_constant_patch_gives_the_same_token_at_any_spacing():
    e = emb()
    ref = tok(e, torch.ones(8, 8, 8), (2.0, 2.0, 2.0), (8, 8, 8))
    for spacing, k in [((1.0, 1.0, 1.0), (16, 16, 16)), ((0.5, 0.5, 4.0), (32, 32, 4)), ((16.0, 0.25, 2.0), (1, 64, 8))]:
        assert torch.allclose(tok(e, torch.ones(k), spacing, k), ref, rtol=1e-3, atol=1e-4)


def test_replicated_voxels_give_the_same_token():
    """A 2 mm scan and the same scan with each voxel copied into 2x2x2 voxels of 1 mm: identical token
    (the embedding integrates the image; it does not care how finely it is sampled)."""
    e = emb()
    coarse = torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(1))
    fine = coarse.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
    assert torch.allclose(tok(e, coarse, (2.0,) * 3, (8,) * 3), tok(e, fine, (1.0,) * 3, (16,) * 3), rtol=1e-3, atol=1e-4)


def test_every_native_voxel_reaches_the_token():
    e = emb()
    for spacing, k in [((0.156, 0.156, 2.75), (103, 103, 6)), ((28.0, 1.0, 1.0), (1, 16, 16))]:
        w = e.weights(torch.tensor(spacing), k)
        assert w.shape[0] == k[0] * k[1] * k[2] and (w.abs().sum(1) > 0).all()
