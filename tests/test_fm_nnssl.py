"""nnssl (OpenMind / nnFoundation) adapter: checkpoints in nnssl's on-disk format, original networks
built with dynamic_network_architectures (the nnssl package itself needs python>=3.12 and is not used)."""
import pytest
import torch
from torch import nn

pytest.importorskip("dynamic_network_architectures")
pytest.importorskip("timm")
pytest.importorskip("einops")

from dynamic_network_architectures.architectures.primus import PrimusS
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from sfc_gdn2.fm import build
from sfc_gdn2.fm.nnssl import _RESENC_L

CNN_PLAN = {  # AdaptationPlan.serialize() of a BaseMAETrainer ResEncL checkpoint
    "architecture_plans": {"arch_class_name": "ResEncL", "arch_kwargs": None, "arch_kwargs_requiring_import": None},
    "pretrain_plan": {"configurations": {"onemmiso": {"patch_size": [64, 64, 64]}}},
    "pretrain_num_input_channels": 1, "recommended_downstream_patchsize": [64, 64, 64],
    "key_to_encoder": "encoder.stages", "key_to_stem": "encoder.stem",
    "keys_to_in_proj": ["encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"], "key_to_lpe": None,
}
VIT_PLAN = {
    "architecture_plans": {"arch_class_name": "PrimusS", "arch_kwargs": None, "arch_kwargs_requiring_import": None},
    "pretrain_plan": {"configurations": {"onemmiso": {"patch_size": [64, 64, 64]}}},
    "pretrain_num_input_channels": 1, "recommended_downstream_patchsize": [64, 64, 64],
    "key_to_encoder": "eva", "key_to_stem": "down_projection", "keys_to_in_proj": ["down_projection.proj"],
    "key_to_lpe": "eva.pos_embed",
}


def _perturb(net: nn.Module) -> nn.Module:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    return net.eval()


def _unet() -> ResidualEncoderUNet:
    return ResidualEncoderUNet(input_channels=1, num_classes=1, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d,
                               nonlin=nn.LeakyReLU, deep_supervision=False, **_RESENC_L)


def _save(tmp_path, net, plan, **extra):
    path = tmp_path / "checkpoint_final.pth"
    torch.save({"network_weights": net.state_dict(), "nnssl_adaptation_plan": plan, "trainer_name": "T",
                "current_epoch": 3, **extra}, path)
    return path


def test_resenc_checkpoint_matches_original_encoder(tmp_path):
    orig = _perturb(_unet())
    path = _save(tmp_path, orig, CNN_PLAN)
    fm = build({"model": "openmind"})
    info = fm.load_checkpoint(path)
    assert fm.input_size == 64                      # from the plan
    assert any(k.startswith("decoder.") for k in info["ignored"])
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        want = orig.encoder(x)[-1]
        got = fm.features(x)["map"]
    assert got.shape == (1, 320, 2, 2, 2) and torch.allclose(got, want, atol=1e-4)


def test_resenc_wrong_key_raises(tmp_path):
    sd = _perturb(_unet()).state_dict()
    k = next(k for k in sd if k.startswith("encoder.stages.2"))
    sd[k.replace("stages.2", "stages.9")] = sd.pop(k)
    path = tmp_path / "bad.pth"
    torch.save({"network_weights": sd, "nnssl_adaptation_plan": CNN_PLAN}, path)
    with pytest.raises(RuntimeError, match="does not match"):
        build({"model": "openmind"}).load_checkpoint(path)


def test_ddp_and_compile_prefixes_are_stripped(tmp_path):
    orig = _perturb(_unet())
    sd = {"module._orig_mod." + k: v for k, v in orig.state_dict().items()}
    path = tmp_path / "ddp.pth"
    torch.save({"network_weights": sd, "nnssl_adaptation_plan": CNN_PLAN}, path)
    build({"model": "openmind"}).load_checkpoint(path)


def test_primus_checkpoint_matches_original_encoder(tmp_path):
    orig = _perturb(PrimusS(1, 1, (8, 8, 8), (64, 64, 64)))
    path = _save(tmp_path, orig, VIT_PLAN)
    fm = build({"model": "openmind", "model_args": {"arch": "PrimusS", "input_size": 64}})
    info = fm.load_checkpoint(path)
    assert any(k.startswith("up_projection.") for k in info["ignored"])
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        tok = orig.down_projection(x)
        want, _ = orig.eva(tok.flatten(2).transpose(1, 2))
        got = fm.features(x)["map"]
    assert got.shape == (1, 396, 8, 8, 8)
    assert torch.allclose(got.flatten(2).transpose(1, 2), want, atol=1e-4)


def test_primus_wrong_key_raises(tmp_path):
    sd = _perturb(PrimusS(1, 1, (8, 8, 8), (64, 64, 64))).state_dict()
    sd["eva.blocks.99.extra"] = torch.zeros(1)
    path = tmp_path / "bad.pth"
    torch.save({"network_weights": sd, "nnssl_adaptation_plan": VIT_PLAN}, path)
    with pytest.raises(RuntimeError, match="does not match"):
        build({"model": "openmind", "model_args": {"arch": "PrimusS", "input_size": 64}}).load_checkpoint(path)


def test_plan_less_checkpoint_uses_model_args(tmp_path):
    orig = _perturb(_unet())
    path = tmp_path / "bare.pth"
    torch.save(orig.state_dict(), path)             # bare state_dict, no plan
    build({"model": "nnfoundation"}).load_checkpoint(path)


def test_nnfoundation_vit_preset_matches_trainer(monkeypatch):
    from sfc_gdn2.fm import nnssl
    monkeypatch.setattr(nnssl, "_make", lambda spec: nn.Identity())  # 674M parameters: spec only
    fm = build({"model": "nnfoundation", "model_args": {"variant": "vit"}})
    k = fm.spec["kwargs"]
    assert (k["embed_dim"], k["depth"], k["heads"], tuple(k["input_shape"])) == (1056, 40, 16, (192,) * 3)
    assert fm.input_size == 192


def test_preprocess_resizes_and_zscores():
    fm = build({"model": "nnfoundation", "model_args": {"input_size": 32}})
    x = fm.preprocess(torch.rand(2, 16, 16, 16))
    assert x.shape == (2, 1, 32, 32, 32) and abs(x.mean().item()) < 1e-4
