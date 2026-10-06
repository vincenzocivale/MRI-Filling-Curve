import json

import pytest
import torch

pytest.importorskip("dynamic_network_architectures")
from sfc_gdn2.fm import build
from sfc_gdn2.fm.nnunet import network_from_plans

ARCH = {
    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
    "arch_kwargs": {"n_stages": 3, "features_per_stage": [4, 8, 16], "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[3, 3, 3]] * 3, "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [2, 2, 2], "n_conv_per_stage_decoder": [2, 2],
                    "conv_bias": True, "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-5, "affine": True}, "dropout_op": None,
                    "dropout_op_kwargs": None, "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True}},
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}


@pytest.fixture
def model_dir(tmp_path):
    plans = {"configurations": {"3d_fullres": {"patch_size": [16, 16, 16], "architecture": ARCH},
                                "3d_child": {"inherits_from": "3d_fullres", "batch_size": 2}}}
    (tmp_path / "plans.json").write_text(json.dumps(plans))
    (tmp_path / "dataset.json").write_text(json.dumps(
        {"channel_names": {"0": "T1"}, "labels": {"background": 0, "a": 1, "b": 2}}))
    return tmp_path


def _write_ckpt(model_dir, prefix=""):
    torch.manual_seed(0)
    net = network_from_plans(ARCH, 1, 3)
    (model_dir / "fold_0").mkdir()
    # trainer.save_checkpoint layout: network_weights + bookkeeping (pickled python objects)
    torch.save({"network_weights": {prefix + k: v for k, v in net.state_dict().items()},
                "init_args": {"configuration": "3d_fullres"}, "trainer_name": "nnUNetTrainer",
                "current_epoch": 1, "_best_ema": 0.1}, model_dir / "fold_0" / "checkpoint_final.pth")
    return net


@pytest.mark.parametrize("prefix", ["", "_orig_mod.", "module.", "module._orig_mod."])
@pytest.mark.parametrize("via_dir", [False, True])
def test_load_matches_original_encoder(model_dir, prefix, via_dir):
    net = _write_ckpt(model_dir, prefix)
    enc = build({"model": "nnunet", "model_args": {"model_dir": str(model_dir), "configuration": "3d_child"}})
    path = model_dir if via_dir else model_dir / "fold_0" / "checkpoint_final.pth"
    assert enc.load_checkpoint(path)["loaded"] == len(net.state_dict())
    x = torch.randn(1, 1, 16, 16, 16)
    assert enc.input_size == 16
    assert torch.allclose(enc.features(x)["map"], net.encoder(x)[-1])
    assert enc.features(x)["map"].shape == (1, 16, 4, 4, 4)


def test_corrupted_key_raises(model_dir):
    _write_ckpt(model_dir)
    p = model_dir / "fold_0" / "checkpoint_final.pth"
    ck = torch.load(p, weights_only=False)
    k = next(iter(ck["network_weights"]))
    ck["network_weights"]["x." + k] = ck["network_weights"].pop(k)
    torch.save(ck, p)
    enc = build({"model": "nnunet", "model_args": {"model_dir": str(model_dir)}})
    with pytest.raises(RuntimeError, match="does not match"):
        enc.load_checkpoint(p)
