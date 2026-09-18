import pytest
import torch
from torch import nn

from sfc_gdn2 import model as models
from sfc_gdn2.engine import evaluate, prediction_batch


class CausalMixer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return (x.cumsum(dim=1),)


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(models, "_gdn2", lambda: CausalMixer)

    def make(curve, objective="masked"):
        torch.manual_seed(17)
        return models.MRIProbe(2, 2, curve, 17, {
            "d_model": 8, "head_dim": 4, "num_heads": 2, "depth": 2,
        }, objective)

    return make


@pytest.mark.parametrize("curve", ["raster", "snake", "morton", "hilbert", "random"])
def test_next_patch_alignment_and_causality(probe, curve):
    model = probe(curve, "next_patch")
    x = torch.randn(2, 8, 2)
    pred, target, mask = prediction_batch(model, x, 1.0, x.device, None)
    torch.testing.assert_close(target, x[:, model.perm][:, 1:])
    assert pred.shape == (2, 7, 2)
    assert mask.all()
    assert all(not block.bidirectional for block in model.blocks)
    changed = x.clone()
    changed[:, model.perm[4:]] += torch.randn(2, 4, 2) * 10
    torch.testing.assert_close(model(changed)[:, :4], pred[:, :4])
    ((pred - target) ** 2).mean().backward()
    assert model.head.weight.grad.abs().sum() > 0
    rows = evaluate(model, [{"patches": x, "dataset": ["test", "test"]}],
                    0.0, x.device, 17)
    assert rows[0]["mse"] == pytest.approx(((pred - target) ** 2).mean().item())
    assert rows[0]["mae"] == pytest.approx((pred - target).abs().mean().item())


@pytest.mark.parametrize("curve", ["raster", "snake", "morton", "hilbert", "random"])
def test_masked_default_and_hidden_values(probe, curve):
    model = probe(curve)
    x = torch.randn(2, 8, 2)
    g = torch.Generator().manual_seed(11)
    pred, target, mask = prediction_batch(model, x, 0.5, x.device, g)
    assert model.objective == "masked"
    assert all(block.bidirectional for block in model.blocks)
    torch.testing.assert_close(target, x)
    changed = x.clone()
    changed[mask] += 10
    torch.testing.assert_close(model(changed, mask), pred)
    assert pred.shape == x.shape


def test_invalid_objective(probe):
    with pytest.raises(ValueError, match="Unknown objective"):
        probe("raster", "invalid")
