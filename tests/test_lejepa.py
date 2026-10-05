import torch

from sfc_gdn2.lejepa import LeJEPA, Projector, SIGReg, per_volume_centred


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_sigreg_separates_gaussian_from_collapse():
    sig = SIGReg()
    gauss = torch.randn(2, 2048, 32, generator=g())
    assert sig(gauss, g(1)) < 2.0
    assert sig(torch.zeros(2, 2048, 32), g(1)) > 100 * sig(gauss, g(1))
    assert sig(3 * gauss, g(1)) > 100 * sig(gauss, g(1))


def test_per_volume_centring_kills_the_global_code_shortcut():
    """One code per volume, shared by all its anchors: Gaussian across volumes, so raw SIGReg is
    fooled; after per-volume centring it is all zero and SIGReg is maximal."""
    sig, b, k, d = SIGReg(), 256, 16, 32
    z = torch.randn(b, 1, d, generator=g()).expand(b, k, d).flatten(0, 1).expand(2, -1, -1)
    vol = torch.arange(b).repeat_interleave(k)
    assert sig(per_volume_centred(z, vol, b), g(1)) > 50 * sig(z, g(1))


def test_projector_is_the_same_function_in_train_and_eval():
    p, x = Projector(16, 32, 8), torch.randn(64, 16, generator=g())
    p.train()
    a = p(x)
    p.eval()
    assert torch.allclose(a, p(x))


def test_groups_views_and_serialization():
    class Enc(torch.nn.Module):
        dim, grid = 8, 16
    m = LeJEPA(Enc(), ["raster", "hilbert"], proj_hidden=8, proj_dim=4)
    xyz = m.xyz
    fg = ((xyz >= 2) & (xyz < 12)).all(-1).expand(4, -1)          # foreground: a 10^3 box
    corner, edge = m.boxes(fg, g())
    assert (corner >= 0).all() and (corner + edge <= 16).all()
    assert (edge[:, :2] >= 6).all()                                # globals: >= 0.3 of the bbox volume
    assert ((edge[:, 2:] >= 3) & (edge[:, 2:] <= 8)).all()         # locals
    vc, ve = m.jittered(corner, edge, g())
    assert (vc >= 0).all() and (vc + ve <= 16).all()
    lo, hi = vc.amax(2), (vc + ve).amin(2)                         # intersection of the two views
    assert (hi > lo).all(-1).all() and not torch.equal(vc[:, :, 0], vc[:, :, 1])
    c, e = vc.reshape(-1, 3), ve.reshape(-1, 3)
    idx, valid = m.serialize(c, e, g())
    for i in range(len(c)):
        seq = idx[i][valid[i]]
        assert len(seq) == e[i].prod() and ((xyz[seq] >= c[i]) & (xyz[seq] < c[i] + e[i])).all()
        assert ((m.ranks[:, seq].diff(dim=1) > 0).all(1)).any()     # the order of some curve view
