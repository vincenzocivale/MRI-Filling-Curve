import numpy as np
import torch
import torch.nn.functional as F

from sfc_gdn2.fm import segrun


def _labels():
    lab = np.zeros((20, 18, 16), dtype=np.uint8)
    lab[4:12, 5:14, 3:9] = 1
    lab[13:18, 2:6, 10:15] = 2
    return lab


def test_identity_grid_round_trip():
    lab = _labels()
    y = segrun.pull_labels(lab, np.eye(4), lab.shape, "cpu")
    assert torch.equal(y, torch.from_numpy(lab))
    probs = F.one_hot(y.long(), 3).permute(3, 0, 1, 2).float()
    d, per = segrun.dice_full_res(probs, np.eye(4), lab, 3)
    assert d == 1.0 and per == [1.0, 1.0]


def test_coarser_grid_and_outside():
    lab = _labels()
    half = np.diag([0.5, 0.5, 0.5, 1.0])
    half[:3, 3] = -0.25  # GT voxel centres -> grid of half the size (pixel-centre convention)
    y = segrun.pull_labels(lab, half, (10, 9, 8), "cpu")
    assert set(y.unique().tolist()) <= {0, 1, 2} and (y == 1).any() and (y == 2).any()
    probs = F.one_hot(y.long(), 3).permute(3, 0, 1, 2).float()
    d, _ = segrun.dice_full_res(probs, half, lab, 3)
    assert d > 0.7
    shifted = np.eye(4)
    shifted[0, 3] = 15  # grid voxel i <- GT voxel i - 15: most of the grid falls outside the GT
    assert (segrun.pull_labels(lab, shifted, lab.shape, "cpu")[:15] == 255).all()


def test_sliding_window_covers_volume():
    x = torch.randn(1, 21, 17, 9)
    net = torch.nn.Conv3d(1, 3, 1)
    probs = segrun.sliding(net, x, (8, 8, 8), 3, torch.no_grad)
    assert probs.shape == (3, 21, 17, 9)
    torch.testing.assert_close(probs.sum(0), torch.ones(21, 17, 9))
    torch.testing.assert_close(probs, net(x[None])[0].softmax(0), atol=1e-5, rtol=1e-5)


def test_patches_and_loss():
    rng = np.random.default_rng(0)
    lab = torch.from_numpy(_labels())
    p = segrun.Patches([{"x": torch.randn(1, *lab.shape), "y": lab}], (8, 8, 8), 3, rng)
    x, y = p.sample(4)
    assert x.shape == (4, 1, 8, 8, 8) and y.shape == (4, 8, 8, 8)
    y[0, 0] = 255
    assert torch.isfinite(segrun.loss_fn(torch.randn(4, 3, 8, 8, 8), y, 3))
