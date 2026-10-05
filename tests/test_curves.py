import numpy as np
import pytest
import torch

from sfc_gdn2.curves import CURVES, CurveViews, grid_coords, order


@pytest.mark.parametrize("name", CURVES)
def test_curve_is_permutation(name):
    assert np.array_equal(np.sort(order(name, 8)), np.arange(8 ** 3))


def test_hilbert_requires_power_of_two():
    with pytest.raises(ValueError):
        order("hilbert", 6)


@pytest.mark.parametrize("name", ["snake", "hilbert"])
def test_unit_step_curves_are_continuous(name):
    c = grid_coords(8)[order(name, 8)]
    assert (np.abs(np.diff(c, axis=0)).sum(1) == 1).all()


def test_morton_visits_octants_contiguously():
    c = grid_coords(8)[order("morton", 8)]
    octant = (c // 4) @ np.array([1, 2, 4])
    assert (np.diff(octant) >= 0).all()


@pytest.mark.parametrize("name", CURVES)
def test_views_are_distinct_permutations_with_identity_first(name):
    v = CurveViews(name, 4)
    assert len(v) == 48
    assert torch.equal(v.perms[0], torch.from_numpy(order(name, 4)))
    assert len({tuple(p.tolist()) for p in v.perms}) == 48
    ar = torch.arange(64)
    for p, r in zip(v.perms, v.ranks):
        assert torch.equal(p.sort().values, ar) and torch.equal(p[r], ar)


def test_views_preserve_step_lengths():
    v = CurveViews("hilbert", 8)
    c = torch.from_numpy(grid_coords(8))
    steps = [(c[p][1:] - c[p][:-1]).abs().sum(1) for p in v.perms]
    assert all(torch.equal(s, steps[0]) for s in steps)
