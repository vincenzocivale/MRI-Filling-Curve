import pytest
import torch
from torch import nn

from sfc_gdn2.data.volume import patchify, unpatchify
from sfc_gdn2.fm import MODELS, build
from sfc_gdn2.fm.base import FoundationEncoder, load_checked, pick, strip_prefix
from sfc_gdn2.fm.extract import fm_extractor, to_grid


def test_unpatchify_inverts_patchify():
    vol = torch.rand(16, 16, 16)
    assert torch.equal(unpatchify(patchify(vol, 4)[None], 4)[0], vol)


def test_to_grid_keeps_canonical_raster_order():
    fmap = torch.arange(8.0).view(1, 1, 2, 2, 2)           # coarser than the grid -> upsampled
    tok = to_grid(fmap, 2)
    assert torch.equal(tok[0, :, 0], fmap.flatten())
    fine = torch.rand(1, 3, 8, 8, 8)                         # finer -> average pooled, patch (i,j,k) = block
    assert torch.allclose(to_grid(fine, 2)[0, 5], fine[0, :, 4:, :4, 4:].mean((1, 2, 3)))


class Toy(FoundationEncoder):
    input_size, in_channels = 8, 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 5, 2, stride=2)

    def features(self, x):
        f = self.conv(x)
        return {"map": f, "global": f.mean((2, 3, 4))}


def test_extractor_shapes_and_levels():
    m, x = Toy().eval(), torch.rand(2, 64, 8)                # 4^3 grid of 2^3 patches -> 8^3 cube
    assert fm_extractor.__name__ and m.preprocess(unpatchify(x, 4)).shape == (2, 2, 8, 8, 8)
    out = m(unpatchify(x, 4))
    assert out["map"].shape == (2, 5, 4, 4, 4) and out["global"].shape == (2, 5)


def test_preprocess_zscores_foreground_and_keeps_background_zero():
    cube = torch.zeros(1, 8, 8, 8)
    cube[:, 2:6, 2:6, 2:6] = torch.rand(4, 4, 4) + 0.5
    x = Toy().normalize(cube[:, None])
    fg = cube[:, None] > 0
    assert x[~fg].abs().max() == 0 and abs(x[fg].mean()) < 1e-5 and abs(x[fg].std(unbiased=False) - 1) < 1e-4


def test_load_checked_raises_on_any_mismatch_but_allows_ignored_prefixes():
    net = nn.Sequential(nn.Linear(2, 2))
    sd = {k: v.clone() for k, v in net.state_dict().items()}
    assert load_checked(net, sd | {"decoder.w": torch.zeros(1)}, ignore_unexpected=("decoder.",))["loaded"] == 2
    with pytest.raises(RuntimeError, match="unexpected"):
        load_checked(net, sd | {"decoder.w": torch.zeros(1)})
    with pytest.raises(RuntimeError, match="missing"):
        load_checked(net, {"0.weight": sd["0.weight"]})


def test_checkpoint_container_helpers():
    assert pick({"state_dict": {"a": 1}, "epoch": 3}, ["model", "state_dict"]) == {"a": 1}
    assert pick({"a": 1}, ["state_dict"]) == {"a": 1}
    assert strip_prefix({"module.a": 1, "b": 2}, "module.") == {"a": 1, "b": 2}


def test_registry_names_and_unknown_model():
    assert {"brainiac", "brainsegfounder", "openmind", "nnfoundation", "amaes", "fomo26", "brainmvp",
            "brainfm", "mome", "mome_plus", "medicalnet", "nnunet"} == set(MODELS)
    with pytest.raises(KeyError, match="available"):
        build({"model": "nope"})
