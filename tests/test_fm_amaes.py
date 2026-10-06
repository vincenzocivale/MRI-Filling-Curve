"""AMAES / FOMO baseline adapter. The pure-torch encoders are compared with the original repo's
networks when `AMAES_SRC` points at its `src/` dir (needs yucca's MedNeXt blocks -> timm); the
checkpoint-format tests need nothing."""
import os
import sys
import types

import pytest
import torch
from torch import nn

from sfc_gdn2.fm import build
from sfc_gdn2.fm.amaes import ARCHS


def _perturb(net: nn.Module) -> nn.Module:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return net.eval()


class Lightning(nn.Module):
    """Stands in for AMAES' LightningModule: the network lives in `self.model`."""

    def __init__(self, net):
        super().__init__()
        self.model = net


@pytest.fixture
def amaes_networks(monkeypatch):
    src = os.environ.get("AMAES_SRC")
    if not src:
        pytest.skip("set AMAES_SRC=<amaes>/src to compare with the original networks")
    pytest.importorskip("yucca.modules.networks.blocks_and_layers.conv_blocks")
    stub = types.ModuleType("yucca.modules.networks.networks.YuccaNet")  # real one drags in the whole yucca stack
    stub.YuccaNet = nn.Module
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.syspath_prepend(src)
    from models import networks
    yield networks
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]


@pytest.mark.parametrize("arch,orig", [("unet_b_lw_dec", "unet_b_lw_dec"), ("unet_xl_lw_dec", "unet_xl_lw_dec"), ("mednext_l3_lw_dec", "mednext_l3_lw_dec"),
                                       ("mednext_m3_lw_dec", "mednext_m3_lw_dec")])
@pytest.mark.parametrize("wrap", ["bare", "compiled", "lightning_ckpt"])
def test_matches_original_encoder(amaes_networks, tmp_path, arch, orig, wrap):
    kw = {"num_classes": 1} if arch.startswith("mednext") else {"output_channels": 1, "prediction": False}
    net = getattr(amaes_networks, orig)(input_channels=1, reconstruction=True, **kw)
    mod = Lightning(_perturb(net))
    sd = mod.state_dict()                                  # `model.encoder.*` + `model.rec_head.*`
    if wrap == "compiled":
        sd = {k.replace("model.", "model._orig_mod.", 1): v for k, v in sd.items()}
    ckpt = {"state_dict": sd, "epoch": 1} if wrap == "lightning_ckpt" else sd
    path = tmp_path / "ckpt.pth"
    torch.save(ckpt, path)
    fm = build({"model": "amaes", "model_args": {"arch": arch, "input_size": 32}})
    info = fm.load_checkpoint(path)
    assert any(k.startswith("model.rec_head.") for k in info["ignored"])
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        want = net.encoder(x)
        got = fm.encoder(x)
    assert len(got) == len(want) == 5
    for g, w in zip(got, want):
        assert g.shape == w.shape and torch.allclose(g, w, atol=1e-4)


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_state_dict_roundtrip_in_amaes_layout(tmp_path, arch):
    src = build({"model": "amaes", "model_args": {"arch": arch}})
    _perturb(src)
    path = tmp_path / "c.pth"
    torch.save({f"model.encoder.{k}": v for k, v in src.encoder.state_dict().items()}
               | {"model.rec_head.out_conv.weight": torch.zeros(1)}, path)
    dst = build({"model": "amaes", "model_args": {"arch": arch}})
    dst.load_checkpoint(path)
    for a, b in zip(src.encoder.state_dict().values(), dst.encoder.state_dict().values()):
        assert torch.equal(a, b)


def test_missing_or_renamed_key_raises(tmp_path):
    fm = build({"model": "amaes", "model_args": {"arch": "unet_b"}})
    sd = {f"model.encoder.{k}": v for k, v in fm.encoder.state_dict().items()}
    k = "model.encoder.encoder_conv2.conv1.conv.weight"
    bad = dict(sd)
    bad[k.replace("conv2", "conv9")] = bad.pop(k)
    torch.save(bad, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="does not match"):
        fm.load_checkpoint(tmp_path / "bad.pth")
    torch.save({"model.rec_head.x": torch.zeros(1)}, tmp_path / "noenc.pth")
    with pytest.raises(RuntimeError, match="no `model.encoder"):
        fm.load_checkpoint(tmp_path / "noenc.pth")


def test_wrong_architecture_checkpoint_raises(tmp_path):
    big = build({"model": "amaes", "model_args": {"arch": "unet_xl"}})
    torch.save({f"model.encoder.{k}": v for k, v in big.encoder.state_dict().items()}, tmp_path / "xl.pth")
    with pytest.raises(RuntimeError):                      # shape mismatch (64 vs 32 filters)
        build({"model": "amaes", "model_args": {"arch": "unet_b"}}).load_checkpoint(tmp_path / "xl.pth")


def test_conventions_differ_between_amaes_and_fomo():
    cube = torch.rand(2, 16, 16, 16) * (torch.rand(2, 16, 16, 16) > 0.5)
    a = build({"model": "amaes", "model_args": {"input_size": 16}}).preprocess(cube)
    f = build({"model": "fomo26", "model_args": {"input_size": 16}}).preprocess(cube)
    assert torch.equal(a[:, 0], cube)                      # [0,1] kept
    fg = cube > 0
    assert abs(f[:, 0][fg].mean().item()) < 1e-3 and (f[:, 0][~fg] == 0).all()
    assert build({"model": "amaes"}).input_size == 128 and build({"model": "fomo26"}).input_size == 96
    assert build({"model": "fomo26"}).arch == "unet_b"


def test_features_shape():
    fm = build({"model": "amaes", "model_args": {"arch": "unet_b", "input_size": 32}})
    assert fm(torch.rand(1, 32, 32, 32))["map"].shape == (1, 512, 2, 2, 2)
