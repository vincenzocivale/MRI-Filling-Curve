import pytest
import torch

monai_nets = pytest.importorskip("monai.networks.nets")

from sfc_gdn2.fm import build

ARGS = {"img_size": 32, "patch_size": 16, "hidden_size": 48, "mlp_dim": 96, "num_layers": 2, "num_heads": 4}


def _original():
    return monai_nets.ViT(in_channels=1, img_size=(32,) * 3, patch_size=(16,) * 3, hidden_size=48, mlp_dim=96,
                          num_layers=2, num_heads=4, save_attn=True).eval()


def _save(tmp_path, ref):
    sd = {f"backbone.{k}": v for k, v in ref.state_dict().items()}
    sd["projection.0.weight"] = torch.zeros(3, 3)       # SimCLR head, must be ignored
    p = tmp_path / "ckpt.ckpt"
    torch.save({"state_dict": sd, "epoch": 1}, p)
    return p


def test_matches_original_forward(tmp_path):
    ref = _original()
    m = build({"model": "brainiac", "model_args": ARGS}).eval()
    with pytest.raises(RuntimeError):                    # head key is not silently accepted as backbone
        m.backbone.load_state_dict({"x": torch.zeros(1)})
    m.load_checkpoint(_save(tmp_path, ref))
    cube = torch.rand(2, 32, 32, 32)
    out = m(cube)
    tokens = ref(m.preprocess(cube))[0]
    assert out["map"].shape == (2, 48, 2, 2, 2)
    assert torch.allclose(out["map"].flatten(2).transpose(1, 2), tokens, atol=1e-5)


def test_renamed_key_raises(tmp_path):
    ref = _original()
    sd = {f"backbone.{k}": v for k, v in ref.state_dict().items()}
    k = next(iter(sd))
    sd[k + "_bad"] = sd.pop(k)
    p = tmp_path / "bad.ckpt"
    torch.save({"state_dict": sd}, p)
    with pytest.raises(RuntimeError, match="does not match"):
        build({"model": "brainiac", "model_args": ARGS}).load_checkpoint(p)
