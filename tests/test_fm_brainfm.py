import pytest
import torch

from sfc_gdn2.fm import build
from sfc_gdn2.fm.brainfm import BrainFMEncoder

ARGS = {"f_maps": 8, "num_levels": 3, "input_size": 16}  # tiny: 8-16-32 channels, 16^3 input


def _save(enc: BrainFMEncoder, path, prefix=""):
    sd = {f"{prefix}backbone.{k}": v for k, v in enc.backbone.state_dict().items()}
    sd["head.0.weight"] = torch.zeros(1)  # task heads live in the same joiner state_dict
    torch.save({"model": sd, "epoch": 3}, path)


def test_keys_follow_repo_layout():
    keys = BrainFMEncoder(**ARGS).backbone.state_dict().keys()
    assert "encoders.0.basic_module.SingleConv1.groupnorm.weight" in keys
    assert "encoders.2.basic_module.SingleConv2.conv.weight" in keys
    assert "decoders.1.basic_module.SingleConv1.groupnorm.bias" in keys
    assert not any("pool" in k or "bias" in k and "conv" in k for k in keys)  # conv has no bias with GroupNorm


@pytest.mark.parametrize("prefix", ["", "module."])
def test_checkpoint_roundtrip(tmp_path, prefix):
    torch.manual_seed(0)
    src, dst = build({"model": "brainfm", "model_args": ARGS}), build({"model": "brainfm", "model_args": ARGS})
    _save(src, tmp_path / "ckpt.pth", prefix)
    x = torch.rand(1, 16, 16, 16)
    assert not torch.allclose(src(x)["map"], dst(x)["map"])
    dst.load_checkpoint(tmp_path / "ckpt.pth")
    out = dst(x)["map"]
    assert out.shape == (1, 8, 16, 16, 16)
    assert torch.allclose(out, src(x)["map"])
    assert torch.allclose(out.norm(dim=1), torch.ones(1, 16, 16, 16), atol=1e-4)  # unit_feat


def test_mismatching_checkpoint_raises(tmp_path):
    src, dst = BrainFMEncoder(**ARGS), BrainFMEncoder(**ARGS)
    sd = {f"backbone.{k}": v for k, v in src.backbone.state_dict().items()}
    sd["backbone.encoders.0.basic_module.SingleConv1.conv.weigth"] = sd.pop(
        "backbone.encoders.0.basic_module.SingleConv1.conv.weight")
    torch.save({"model": sd}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="does not match"):
        dst.load_checkpoint(tmp_path / "bad.pth")


def test_pool_and_level():
    enc = BrainFMEncoder(**ARGS, feature_level=0, pool_to=2)
    assert enc(torch.rand(1, 16, 16, 16))["map"].shape == (1, 32, 2, 2, 2)
