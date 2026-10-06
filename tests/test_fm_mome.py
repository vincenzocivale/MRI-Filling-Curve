"""MoME / MoME+ adapters vs the MoME fork's own nnU-Net classes (PlainConvUNet, ClsDispatchNet). CPU."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from sfc_gdn2 import fm
from sfc_gdn2.fm.mome import arch_from_plans, sibling

ARCH = {"features_per_stage": [4, 8, 8, 8, 8], "kernel_sizes": [3] * 5, "strides": [1, 2, 2, 2, 2],
        "n_conv_per_stage": [2] * 5}
SIZE = 32


def repo_root() -> Path:
    root = os.environ.get("SFC_EXT_REPOS")
    path = Path(root or "/nonexistent") / "MoME"
    if not path.exists():
        pytest.skip("set SFC_EXT_REPOS to a directory containing the MoME clone")
    return path


def ref_unet(cls, cin, **kw):
    return cls(cin, 5, ARCH["features_per_stage"], nn.Conv3d, 3, ARCH["strides"], 2, 2, 2, conv_bias=True,
               norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True}, nonlin=nn.LeakyReLU,
               nonlin_kwargs={"inplace": True}, deep_supervision=True, **kw)


def save(net, path):
    torch.save({"network_weights": net.state_dict(), "init_args": {}, "trainer_name": "nnUNetTrainer"}, path)


def adapter(name, **kw):
    return fm.build({"model": name, "model_args": {"arch": ARCH, "input_size": SIZE, **kw}}).eval()


def test_mome_matches_foundation_fork(tmp_path):
    sys.path.insert(0, str(repo_root() / "MoME_foundation"))
    try:
        from nnunetv2.dynamic_network_architectures.architectures.unet import PlainConvUNet
    finally:
        sys.path.pop(0)
    torch.manual_seed(0)
    experts = [ref_unet(PlainConvUNet, 1).eval() for _ in range(5)]
    agg = ref_unet(PlainConvUNet, 1 + 5 * 4, num_experts=5).eval()
    save(agg, tmp_path / "checkpoint_best.pth")
    for i, e in enumerate(experts, 1):
        save(e, tmp_path / f"checkpoint_best{i}.pth")
    model = adapter("mome")
    report = model.load_checkpoint(tmp_path / "checkpoint_best.pth")
    assert set(report) == {"aggregator", *[f"expert{i}" for i in range(1, 6)]}
    x = model.preprocess(torch.rand(2, SIZE, SIZE, SIZE))
    with torch.no_grad():
        skips = [e.encoder(x)[0] for e in experts]  # `Feature_i[0]` of the trainer
        expected = agg.encoder(torch.cat([x, *skips], 1))[-1]
    out = model.features(x)
    torch.testing.assert_close(out["map"], expected, rtol=1e-4, atol=1e-4)
    assert out["map"].shape == (2, 8, 2, 2, 2) and out["global"].shape == (2, 8)


def test_mome_plus_matches_plus_fork_and_dispatch(tmp_path):
    root = repo_root()
    sys.path.insert(0, str(root / "MoME_plus" / "nnunetv2"))
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    finally:
        sys.path.pop(0)
    spec = importlib.util.spec_from_file_location(
        "mome_plus_dispatch", root / "MoME_plus/nnunetv2/training/nnUNetTrainer/Dispatch_network.py")
    dispatch_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch_mod)

    torch.manual_seed(0)
    experts = [ref_unet(PlainConvUNet, 1, num_experts=4).eval() for _ in range(4)]
    agg = ref_unet(PlainConvUNet, 4 + 4 * 4, num_experts=4, Mod_Prior=True).eval()
    disp = dispatch_mod.ClsDispatchNet(input_dim=4).eval()
    save(agg, tmp_path / "checkpoint_best.pth")
    for i, e in enumerate(experts, 1):
        save(e, tmp_path / f"checkpoint_best{i}.pth")
    save(disp, tmp_path / "checkpoint_best_dispatch.pth")
    model = adapter("mome_plus", modality="t1")
    assert set(model.load_checkpoint(tmp_path / "checkpoint_best.pth")) >= {"aggregator", "dispatch", "expert4"}

    x = model.preprocess(torch.rand(2, SIZE, SIZE, SIZE))
    with torch.no_grad():  # predict_from_raw_data.py:_internal_maybe_mirror_and_predict, MultiMod = [1,0,0,0]
        data = torch.zeros(2, 4, *x.shape[2:])
        data[:, 0] = x[:, 0]
        mask = torch.tensor([[1.0, 0, 0, 0]])
        modality_embading = mask[..., None].repeat(1, 1, 4).permute(0, 2, 1)
        raw = disp(mask)
        mn, mx = raw.min(dim=2, keepdim=True).values, raw.max(dim=2, keepdim=True).values
        sparse = ((raw - mn + 1e-4) / (mx - mn + 1e-4)) * modality_embading
        keep = mask == 1
        sparse[keep, :] = torch.eye(4)[None].repeat(1, 1, 1)[keep, :]
        dispatched = torch.sum(sparse.expand(2, -1, -1)[..., None, None, None] * data.unsqueeze(1), dim=2)
        skips = [e.encoder(dispatched[:, i:i + 1])[0] for i, e in enumerate(experts)]
        expected = agg.encoder(torch.cat([dispatched, *skips], 1))[-1]
    torch.testing.assert_close(model.features(x)["map"], expected, rtol=1e-4, atol=1e-4)


def test_mome_plus_modality_slot_changes_features():
    x = torch.rand(1, 1, SIZE, SIZE, SIZE)
    torch.manual_seed(1)
    a = adapter("mome_plus", modality="t1")
    b = adapter("mome_plus", modality="flair")
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a.features(x)["map"], b.features(x)["map"])


def test_corrupted_expert_key_raises(tmp_path):
    model = adapter("mome")
    sd = model.aggregator.state_dict()
    torch.save({"network_weights": sd}, tmp_path / "checkpoint_best.pth")
    for i in range(1, 6):
        e = dict(model.experts[i - 1].state_dict())
        if i == 3:
            e["encoder.stages.0.0.convs.0.all_modules.0.weights"] = e.pop("encoder.stages.0.0.convs.0.all_modules.0.weight")
        torch.save({"network_weights": e}, tmp_path / f"checkpoint_best{i}.pth")
    with pytest.raises(RuntimeError, match="expert 3"):
        model.load_checkpoint(tmp_path / "checkpoint_best.pth")


def test_checkpoint_key_layout_has_original_duplicates():
    """The fork's blocks register conv/norm twice (`conv.*` and `all_modules.0.*`); real checkpoints carry both."""
    keys = adapter("mome").aggregator.state_dict()
    assert "encoder.stages.0.0.convs.0.conv.weight" in keys
    assert "encoder.stages.0.0.convs.0.all_modules.1.weight" in keys


def test_sibling_naming_rule():
    assert sibling(Path("f/checkpoint_best.pth"), "2").name == "checkpoint_best2.pth"
    assert sibling(Path("f/checkpoint_final.pth"), "_dispatch").name == "checkpoint_final_dispatch.pth"
    with pytest.raises(ValueError):
        sibling(Path("f/weights.pth"), "1")


def test_arch_from_plans_both_layouts(tmp_path):
    old = {"configurations": {"3d_fullres": {"conv_kernel_sizes": [[3, 3, 3]] * 4, "UNet_base_num_features": 32,
                                              "unet_max_num_features": 100, "n_conv_per_stage_encoder": [2] * 4,
                                              "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [1, 2, 2]]}}}
    (tmp_path / "old.json").write_text(json.dumps(old))
    a = arch_from_plans(tmp_path / "old.json")
    assert a["features_per_stage"] == [32, 64, 100, 100] and a["strides"][3] == [1, 2, 2] and a["strides"][0] == 1
    new = {"configurations": {"base": {"architecture": {"arch_kwargs": {
        "n_stages": 2, "features_per_stage": [8, 16], "kernel_sizes": [[3, 3, 3]] * 2, "strides": [[1, 1, 1], [2, 2, 2]],
        "n_conv_per_stage": [2, 2]}}}, "3d_fullres": {"inherits_from": "base"}}}
    (tmp_path / "new.json").write_text(json.dumps(new))
    assert arch_from_plans(tmp_path / "new.json")["features_per_stage"] == [8, 16]
    model = fm.build({"model": "mome", "model_args": {"plans_json": str(tmp_path / "new.json")}})
    assert model.arch["features_per_stage"] == [8, 16]
