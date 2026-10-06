import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from sfc_gdn2.fm import build

REPO = Path("/tmp/claude-5235/-home-vcivale-MRI-Filling-Curve/660599b5-db2d-42e7-9eef-1171ce8969ec/"
            "scratchpad/ext/MedicalNet")


def _original(depth, shortcut):
    """The upstream resnet.py, loaded from a local clone (not vendored); skipped if absent."""
    f = REPO / "models" / "resnet.py"
    if not f.exists():
        pytest.skip("MedicalNet clone not available")
    spec = importlib.util.spec_from_file_location("mn_resnet", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mn_resnet"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, f"resnet{depth}")(sample_input_W=64, sample_input_H=64, sample_input_D=64,
                                           shortcut_type=shortcut, no_cuda=True, num_seg_classes=2).eval()


@pytest.mark.parametrize("depth,shortcut", [(10, "A"), (50, "B")])
def test_matches_original_forward(tmp_path, depth, shortcut):
    ref = _original(depth, shortcut)
    for bn in ref.modules():
        if isinstance(bn, torch.nn.BatchNorm3d):
            bn.running_mean.normal_()
            bn.running_var.uniform_(0.5, 2)
    p = tmp_path / "resnet.pth"
    torch.save({"epoch": 3, "state_dict": {f"module.{k}": v for k, v in ref.state_dict().items()}}, p)
    m = build({"model": "medicalnet", "model_args": {"depth": depth, "input_size": 32}}).eval()
    m.load_checkpoint(p)
    cube = torch.rand(1, 32, 32, 32)
    x = m.preprocess(cube)
    ref_feat = ref.layer4(ref.layer3(ref.layer2(ref.layer1(ref.maxpool(ref.relu(ref.bn1(ref.conv1(x))))))))
    out = m(cube)["map"]
    assert out.shape == (1, 512 * (1 if depth < 50 else 4), 4, 4, 4)
    assert torch.allclose(out, ref_feat, atol=1e-4, rtol=1e-4)


def test_renamed_key_raises(tmp_path):
    m = build({"model": "medicalnet", "model_args": {"depth": 10}})
    sd = {f"module.{k}": v for k, v in m.net.state_dict().items()}
    sd["module.layer1.0.conv1.weights"] = sd.pop("module.layer1.0.conv1.weight")
    p = tmp_path / "bad.pth"
    torch.save({"state_dict": sd}, p)
    with pytest.raises(RuntimeError, match="does not match"):
        m.load_checkpoint(p)
