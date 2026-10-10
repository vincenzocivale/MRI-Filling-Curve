import pytest
import torch

from sfc_gdn2.curves import CURVES, keys, symmetries, transform


def coords(*dims):
    return torch.stack(torch.meshgrid(*[torch.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)


def walk(name, *dims):
    c = coords(*dims)
    return c[keys(name, c, torch.tensor(dims)).argsort()]


@pytest.mark.parametrize("name", CURVES)
@pytest.mark.parametrize("dims", [(8, 8, 8), (5, 3, 7)])
def test_keys_are_distinct(name, dims):
    c = coords(*dims)
    assert len(keys(name, c, torch.tensor(dims)).unique()) == len(c)


@pytest.mark.parametrize("name", ["snake", "hilbert"])
def test_unit_step_curves_are_continuous(name):
    assert ((walk(name, 8, 8, 8).diff(dim=0).abs().sum(1)) == 1).all()


def test_snake_is_continuous_on_any_box():
    assert ((walk("snake", 5, 3, 7).diff(dim=0).abs().sum(1)) == 1).all()


def test_morton_visits_octants_contiguously():
    octant = (walk("morton", 8, 8, 8) // 4) @ torch.tensor([1, 2, 4])
    assert (octant.diff() >= 0).all()


def test_symmetries_are_48_distinct_traversals_with_identity_first():
    perms, flips = symmetries()
    assert len(perms) == 48 and perms[0].tolist() == [0, 1, 2] and not flips[0].any()
    c, d = coords(4, 4, 4), torch.tensor([4, 4, 4])
    orders = {tuple(keys("hilbert", *transform(c, d, p, f)).argsort().tolist()) for p, f in zip(perms, flips)}
    assert len(orders) == 48


def test_transform_stays_in_the_box_and_preserves_steps():
    perms, flips = symmetries()
    d = torch.tensor([5, 3, 7])
    path = walk("hilbert", 5, 3, 7)
    for p, f in zip(perms, flips):
        t, td = transform(path, d, p, f)
        assert ((t >= 0) & (t < td)).all() and sorted(td.tolist()) == [3, 5, 7]
        assert torch.equal(t.diff(dim=0).abs().sum(1), path.diff(dim=0).abs().sum(1))
