"""BrainMVP adapter vs the repo's own UniFormer (Downstream/model/uniformer.py). CPU, tiny input."""
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from sfc_gdn2 import fm


def repo_uniformer():
    root = os.environ.get("SFC_EXT_REPOS")
    path = Path(root or "/nonexistent") / "BrainMVP/Downstream/model/uniformer.py"
    if not path.exists():
        pytest.skip("set SFC_EXT_REPOS to a directory containing the BrainMVP clone")
    if importlib.util.find_spec("timm") is None:  # the repo only needs timm for DropPath (a no-op at rate 0)
        layers = types.ModuleType("timm.models.layers")
        layers.DropPath = nn.Identity
        for name, mod in (("timm", types.ModuleType("timm")), ("timm.models", types.ModuleType("timm.models")),
                          ("timm.models.layers", layers)):
            sys.modules.setdefault(name, mod)
    spec = importlib.util.spec_from_file_location("brainmvp_uniformer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def randomize_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)


def adapter():
    return fm.build({"model": "brainmvp", "model_args": {"input_size": 32}}).eval()


def test_pretraining_checkpoint_matches_repo_encoder(tmp_path):
    ref = repo_uniformer().uniformer_small(img_size=32, in_chans=1).eval()
    torch.manual_seed(0)
    randomize_bn(ref)
    # pretraining RecModel layout under DDP: module.encoder.uniformer.* + decoder.* + rep_template
    sd = {"module.encoder.uniformer." + k: v for k, v in ref.state_dict().items()}
    sd |= {"module.decoder.out_1.conv.conv.weight": torch.zeros(1), "module.rep_template": torch.zeros(8, 4, 4, 4)}
    torch.save({"state_dict": sd}, tmp_path / "BrainMVP_uniformer.pt")
    model = adapter()
    model.load_checkpoint(tmp_path / "BrainMVP_uniformer.pt")
    x = torch.rand(2, 1, 32, 32, 32)
    out = model.features(x)["map"]
    # the repo returns x4 in (D, H, W) order; ours is permuted back to the canonical input order
    torch.testing.assert_close(out, ref(x)[4].permute(0, 1, 3, 4, 2), rtol=1e-4, atol=1e-4)
    assert out.shape == (2, 512, 2, 2, 2)


def test_downstream_style_keys_and_roundtrip(tmp_path):
    src = adapter()
    sd = {"encoder." + k: v for k, v in src.uniformer.state_dict().items()}  # UniUnet / transfer layout
    torch.save({"state_dict": sd}, tmp_path / "a.pt")
    dst = adapter()
    dst.load_checkpoint(tmp_path / "a.pt")
    for a, b in zip(src.uniformer.state_dict().values(), dst.uniformer.state_dict().values()):
        assert torch.equal(a, b)


def test_mismatch_raises(tmp_path):
    sd = {"module.encoder.uniformer." + k: v for k, v in adapter().uniformer.state_dict().items()}
    key = "module.encoder.uniformer.blocks3.0.attn.qkv.weight"
    sd[key.replace("qkv", "qkx")] = sd.pop(key)
    torch.save({"state_dict": sd}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="does not match"):
        adapter().load_checkpoint(tmp_path / "bad.pt")


def test_preprocess_percentile_scaling():
    model = adapter()
    cube = torch.zeros(1, 32, 32, 32)
    cube[0, 8:24, 8:24, 8:24] = torch.rand(16, 16, 16)
    x = model.preprocess(cube)
    assert x.shape == (1, 1, 32, 32, 32) and x.min() == 0 and x.max() == 1 and x[0, 0, 0, 0, 0] == 0
