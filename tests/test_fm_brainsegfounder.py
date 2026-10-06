"""BrainSegFounder adapter vs MONAI's SwinTransformer / the repo's own SSLHead. CPU, tiny model."""
import importlib.util
import os
from argparse import Namespace
from pathlib import Path

import pytest
import torch

pytest.importorskip("monai")
from monai.networks.nets.swin_unetr import SwinTransformer

from sfc_gdn2 import fm

ARGS = {"feature_size": 12, "in_channels": 1, "depths": [1, 1, 1, 1], "num_heads": [3, 6, 12, 24], "input_size": 64}


def adapter():
    return fm.build({"model": "brainsegfounder", "model_args": ARGS}).eval()


def reference() -> SwinTransformer:
    torch.manual_seed(0)
    return SwinTransformer(in_chans=1, embed_dim=12, window_size=(7,) * 3, patch_size=(2,) * 3, depths=(1,) * 4,
                           num_heads=(3, 6, 12, 24), spatial_dims=3).eval()


def check(ckpt, tmp_path, ref):
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)
    model = adapter()
    report = model.load_checkpoint(path)
    cube = torch.rand(2, 64, 64, 64)
    assert report["loaded"] > 0
    torch.testing.assert_close(model(cube)["map"], ref(cube[:, None])[4])


def test_ddp_ssl_head_checkpoint(tmp_path):
    """Real SSLHead from the repo (if cloned) behind DDP's `module.` -- the stage-1/2 pretraining format."""
    root = os.environ.get("SFC_EXT_REPOS")
    ssl = Path(root or "/nonexistent") / "BrainSegFounder/downstream/BraTS/ssl/models/ssl_head.py"
    if not ssl.exists():
        pytest.skip("set SFC_EXT_REPOS to a directory containing the BrainSegFounder clone")
    spec = importlib.util.spec_from_file_location("bsf_ssl_head", ssl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = Namespace(spatial_dims=3, bottleneck_depth=16 * 12, in_channels=1, feature_size=12,
                     num_swin_blocks_per_stage=[1] * 4, num_heads_per_stage=[3, 6, 12, 24],
                     dropout_path_rate=0.0, use_checkpoint=False)
    head = mod.SSLHead(args).eval()
    sd = {"module." + k: v for k, v in head.state_dict().items()}
    check({"epoch": 3, "state_dict": sd, "optimizer": {}}, tmp_path, head.swinViT)


def test_original_monai_swin_keys(tmp_path):
    ref = reference()
    check({"state_dict": {"module." + k: v for k, v in ref.state_dict().items()}}, tmp_path, ref)


def test_finetuned_swinunetr_checkpoint_ignores_decoder(tmp_path):
    ref = reference()
    sd = {"swinViT." + k: v for k, v in ref.state_dict().items()}
    sd |= {"encoder1.layer.conv1.conv.weight": torch.zeros(1), "decoder5.transp_conv.conv.weight": torch.zeros(1),
           "out.conv.conv.weight": torch.zeros(1)}
    check({"state_dict": sd}, tmp_path, ref)


def test_bare_state_dict(tmp_path):
    ref = reference()
    check(ref.state_dict(), tmp_path, ref)


def test_mismatched_checkpoint_raises(tmp_path):
    ref = reference()
    sd = {"module." + k: v for k, v in ref.state_dict().items()}
    key = next(k for k in sd if "norm1.weight" in k)
    sd["module." + key.removeprefix("module.").replace("norm1", "normX")] = sd.pop(key)
    torch.save({"state_dict": sd}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="does not match"):
        adapter().load_checkpoint(tmp_path / "bad.pt")
    torch.save({"state_dict": {k: v for k, v in sd.items() if "patch_embed" not in k}}, tmp_path / "bad2.pt")
    with pytest.raises(RuntimeError, match="missing"):
        adapter().load_checkpoint(tmp_path / "bad2.pt")


def test_patch_level_map_shape():
    out = adapter()(torch.rand(1, 64, 64, 64))
    assert out["map"].shape == (1, 192, 2, 2, 2) and out["global"].shape == (1, 192)
