import torch

from sfc_gdn2.curves import grid_coords
from sfc_gdn2.lejepa import LeJEPA, Projector, SIGReg


def g(seed=0):
    return torch.Generator().manual_seed(seed)


def test_sigreg_separates_gaussian_from_collapse():
    sig = SIGReg()
    gauss = torch.randn(2, 2048, 32, generator=g())
    assert sig(gauss, g(1)) < 2.0
    assert sig(torch.zeros(2, 2048, 32), g(1)) > 100 * sig(gauss, g(1))
    assert sig(3 * gauss, g(1)) > 100 * sig(gauss, g(1))


def test_projector_is_the_same_function_in_train_and_eval():
    p, x = Projector(16, 32, 8), torch.randn(64, 16, generator=g())
    p.train()
    a = p(x)
    p.eval()
    assert torch.allclose(a, p(x))


class Enc(torch.nn.Module):
    """Identity encoder (token = patch content, no mixing)."""
    dim = 4
    tokens = staticmethod(lambda v, w, mask: v)
    apply_mask = staticmethod(lambda t, mask: t)
    patch_embed = type("PE", (), {"weights": staticmethod(lambda spacing, k: None),
                                  "__call__": lambda self, v, w: v})()

    grad_checkpoint = False

    @staticmethod
    def packed(x, valid, ckpt=None):
        return x * valid[..., None]


def vols(patches, grid):
    """Scans with 1 x 2 x 2-voxel patches (P = 4 = Enc.dim) of 4 mm voxels."""
    b = len(grid)
    return {"patches": patches, "grid": grid, "k": torch.tensor([[1, 2, 2]] * b), "spacing": torch.full((b, 3), 4.0),
            "k_list": [[1, 2, 2]] * b, "spacing_list": [[4.0] * 3] * b}


def test_groups_views_and_serialization():
    m = LeJEPA(Enc(), ["raster", "hilbert"], proj_hidden=8, proj_dim=4, max_view_tokens=600)
    grid = torch.tensor([[16, 16, 16], [20, 9, 14], [16, 16, 16], [6, 30, 12]])
    xyz = grid_coords(grid, 16 ** 3)
    fg = ((xyz >= 2) & (xyz < torch.tensor([12, 8, 12]))).all(-1)  # foreground: a box inside every grid
    corner, edge = m.boxes(fg, xyz, grid, g())
    gb = grid[:, None]
    assert (corner >= 0).all() and (corner + edge <= gb).all()
    assert ((edge[:, 2:] >= 3) & (edge[:, 2:] <= 8)).all()         # locals
    vc, ve = m.jittered(corner, edge, grid, g())
    assert (vc >= 0).all() and (vc + ve <= gb[:, :, None]).all() and (ve.prod(-1) <= 600).all()
    lo, hi = vc.amax(2), (vc + ve).amin(2)                         # intersection of the two views
    assert (hi > lo).all(-1).all() and not torch.equal(vc[:, :, 0], vc[:, :, 1])
    vol = torch.arange(4)[:, None, None].expand(-1, vc.shape[1], 2).reshape(-1)
    c, e = vc.reshape(-1, 3), ve.reshape(-1, 3)
    idx, valid = m.serialize(xyz[vol], c, e, g())
    for i in range(len(c)):
        p = xyz[vol[i]][idx[i][valid[i]]]
        assert len(p) == e[i].prod() and len(p.unique(dim=0)) == len(p)
        assert ((p >= c[i]) & (p < c[i] + e[i])).all()


def test_thick_slices_average_native_slices_along_one_axis():
    m = LeJEPA(Enc(), ["raster"], proj_hidden=8, proj_dim=4, thick_prob=1.0, thick_mm=(4.0,))
    v = torch.rand(30, 64, generator=g())
    on, axis, mm = torch.ones(30, dtype=torch.bool), torch.arange(30) % 3, torch.full((30,), 4.0)
    t = m.thick(v.clone(), [4, 4, 4], [2.0, 2.0, 2.0], on, axis, mm).view(30, 4, 4, 4)
    for x, a in zip(t, axis.tolist()):                               # 2 mm slices -> pairs (4 mm slabs)
        assert torch.equal(x.select(a, 0), x.select(a, 1)) and torch.equal(x.select(a, 2), x.select(a, 3))
    assert torch.allclose(t.mean((-1, -2, -3)), v.mean(-1), atol=1e-6)
    thick_z = m.thick(v.clone(), [4, 4, 4], [2.0, 2.0, 6.0], on, torch.full((30,), 2), mm)
    assert torch.equal(thick_z, v)                                   # already thicker than 4 mm: untouched


def test_token_targets_are_the_same_patch_in_the_other_view():
    """Identity encoder (token = patch content, no mixing): A's masked patches must be matched to
    the same patches in B, through jitter, different curves and the backward pass -> zero loss."""
    m = LeJEPA(Enc(), ["raster", "hilbert"], gamma=0, scale=0, shift=0, noise=0, proj_hidden=8, proj_dim=4,
               mask_ratio=(0.5, 0.5))
    m.tok = torch.nn.Identity()
    grid = torch.tensor([[16, 16, 16], [12, 20, 10], [8, 8, 30]])
    patches = [torch.rand(int(n), 4, generator=g()) * 0.5 + 0.2 for n in grid.prod(1)]
    out = m(vols(patches, grid), g())
    assert out["inv_token"] == 0 and torch.isfinite(out["loss"])


def test_token_term_ignores_background():
    """Masking (hence token pairs) touches foreground patches only; background stays in the views."""
    m = LeJEPA(Enc(), ["raster", "hilbert"], proj_hidden=8, proj_dim=4, mask_ratio=(1.0, 1.0))
    grid = torch.tensor([[16, 16, 16]] * 2)
    patches = torch.rand(2, 4096, 4, generator=g()) * 0.5 + 0.2
    patches[:, ::2] = 0                                            # every other patch is air
    xyz = grid_coords(grid, 4096)
    fg = patches.mean(-1) > m.fg_threshold
    v = vols(list(patches), grid)
    corner, edge = m.jittered(*m.boxes(fg, xyz, grid, g()), grid, g())
    vol = torch.arange(2)[:, None, None].expand(-1, corner.shape[1], 2).reshape(-1)
    c, e = corner.reshape(-1, 3), edge.reshape(-1, 3)
    masked = torch.ones(len(c), dtype=torch.bool)
    _, valid, fgv, mask, _ = m.read(v, [None, None], fg, xyz, vol, c, e, masked, g())
    assert (valid & ~fgv).any()                                    # background is still read
    assert torch.equal(mask, fgv)                                  # ratio 1: all foreground, nothing else
    assert torch.isfinite(m(v, g())["loss"])


def test_masks_are_aligned_cubes_and_cell_targets_match():
    """Masks cover whole aligned cubes; with no jitter A and B hold the same patches, so through the identity
    encoder every fully masked cell of A equals B's: zero cell loss, with pairs at every level."""
    m = LeJEPA(Enc(), ["raster", "hilbert"], gamma=0, scale=0, shift=0, noise=0, jitter=0, proj_hidden=8, proj_dim=4,
               mask_ratio=(0.5, 0.5), levels=(2, 4), mask_cells=(2,))
    m.cell = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    grid = torch.tensor([[16, 16, 16], [12, 20, 10]])
    patches = [torch.rand(int(n), 4, generator=g()) * 0.5 + 0.2 for n in grid.prod(1)]
    v = vols(patches, grid)
    xyz = grid_coords(grid, 16 ** 3)
    fg = torch.ones(2, 16 ** 3, dtype=torch.bool)
    corner, edge = m.jittered(*m.boxes(fg, xyz, grid, g()), grid, g())
    vol = torch.arange(2)[:, None, None].expand(-1, corner.shape[1], 2).reshape(-1)
    c, e = corner.reshape(-1, 3), edge.reshape(-1, 3)
    idx, valid, _, mask, _ = m.read(v, [None, None], fg, xyz, vol, c, e, torch.ones(len(c), dtype=torch.bool), g())
    cube = m.cell_ids(xyz[vol[:, None], idx], grid[vol], torch.full((len(c),), 2))
    for i in range(len(c)):
        ids, mk = cube[i][valid[i]], mask[i][valid[i]]
        for u in ids.unique():
            assert len(mk[ids == u].unique()) == 1                     # a cube is masked whole or not at all
    assert mask.any() and not mask[valid].all()
    out = m(v, g())
    assert out["pairs_cell2"] > 0 and out["pairs_cell4"] >= 0
    assert out["inv_cell2"] < 1e-10 and out["inv_cell4"] < 1e-10 and torch.isfinite(out["loss"])
