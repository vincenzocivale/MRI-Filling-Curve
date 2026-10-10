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
    torch.testing.assert_close(probs.float().sum(0), torch.ones(21, 17, 9), atol=5e-3, rtol=0)
    torch.testing.assert_close(probs.float(), net(x[None])[0].softmax(0), atol=5e-3, rtol=0)  # fp16 accumulator


def test_slab_crop_matches_whole_grid():
    lab = _labels()
    probs = torch.rand(3, 12, 10, 9).half().float()  # the same values the fp16 accumulator holds
    m = np.eye(4)
    m[:3, :3] = [[0.5, 0.2, 0.0], [-0.1, 0.6, 0.1], [0.05, 0.0, 0.7]]
    m[:3, 3] = [-2.0, 1.5, -1.0]  # oblique, partly outside the grid on every side
    M = torch.tensor(m, dtype=torch.float32)
    ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in lab.shape], indexing="ij"))
    c = M[:3, :3] @ ijk.reshape(3, -1) + M[:3, 3:]
    G = torch.tensor(probs.shape[1:], dtype=torch.float32)
    g = (2 * (c + 0.5) / G[:, None] - 1).flip(0).T.reshape(1, 1, 1, -1, 3)
    pred = F.grid_sample(probs[None], g, padding_mode="border", align_corners=False)[0, :, 0, 0].argmax(0)
    t = torch.from_numpy(lab).reshape(-1).long()
    ref = [2 * ((pred == k) & (t == k)).sum().item() / ((pred == k).sum() + (t == k).sum()).item() for k in (1, 2)]
    _, per = segrun.dice_full_res(probs.half(), m, lab, 3, "cpu")
    np.testing.assert_allclose(per, ref, atol=1e-2)  # random probs: any misalignment moves Dice far more


def test_patches_and_loss():
    rng = np.random.default_rng(0)
    lab = torch.from_numpy(_labels())
    p = segrun.Patches([{"x": torch.randn(1, *lab.shape), "y": lab}], (8, 8, 8), 3, rng)
    x, y = p.sample(4)
    assert x.shape == (4, 1, 8, 8, 8) and y.shape == (4, 8, 8, 8)
    y[0, 0] = 255
    assert torch.isfinite(segrun.loss_fn(torch.randn(4, 3, 8, 8, 8), y, 3))


def test_empty_input_is_background():
    _, per = segrun.dice_full_res(torch.zeros(3, 0, 0, 0), np.eye(4), _labels(), 3, "cpu")
    assert per == [0.0, 0.0]
