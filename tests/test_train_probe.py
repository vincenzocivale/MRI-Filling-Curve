import torch
from torch import nn

from sfc_gdn2.pretrain import param_groups
from sfc_gdn2.probe import (
    bootstrap_ci,
    classification_metrics,
    fit_logistic,
    fit_ridge,
    position_extractor,
    regression_metrics,
    roc_auc,
    segmentation_metrics,
)


def test_param_groups_decay_matrices_only():
    m = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    m.A_log = nn.Parameter(torch.zeros(2, 2))
    m.A_log._no_weight_decay = True
    decay, keep = param_groups(m, 0.05)
    assert [p.shape for p in decay["params"]] == [torch.Size([4, 4])]
    assert len(keep["params"]) == 4 and keep["weight_decay"] == 0.0


def test_roc_auc_extremes():
    y = torch.tensor([0, 0, 1, 1])
    assert roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), y) == 0.0


def test_logistic_probe_separates_and_ci_brackets():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (400,), generator=g)
    x = torch.randn(400, 8, generator=g) + y[:, None] * 1.5
    head = fit_logistic(x, y, 2, 1e-3)
    m = classification_metrics(head(x), y)
    lo, hi = bootstrap_ci(head(x), y, torch.arange(400), classification_metrics, "roc_auc", n=200)
    assert m["roc_auc"] > 0.95 and lo <= m["roc_auc"] <= hi


def test_ridge_recovers_a_linear_target_in_both_regimes():
    gen = torch.Generator().manual_seed(0)
    for n, d in ((300, 10), (50, 200)):                 # primal and dual solve
        x = torch.randn(n, d, generator=gen)
        y = x[:, 0] * 3 + 1
        m = regression_metrics(fit_ridge(x, y, 1e-4)(x), y)
        assert m["r2"] > 0.99 and m["pearson_r"] > 0.99


def test_segmentation_macro_ap_ignores_background_and_absent_classes():
    y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    perfect = torch.nn.functional.one_hot(y, 4).float() * 5
    m = segmentation_metrics(perfect, y)
    assert m["classes"] == 2 and abs(m["macro_ap"] - 1) < 1e-6 and abs(m["chance_ap"] - 0.25) < 1e-6
    swapped = perfect[:, [0, 2, 1, 3]]          # classes 1 and 2 confused
    assert segmentation_metrics(swapped, y)["macro_ap"] < 0.5


def test_bootstrap_resamples_whole_volumes():
    y = torch.tensor([0, 1] * 50)
    out = torch.nn.functional.one_hot(y, 2).float()
    groups = torch.arange(100) // 10
    lo, hi = bootstrap_ci(out, y, groups, classification_metrics, "accuracy", n=50)
    assert lo == hi == 1.0


def test_position_features_depend_only_on_location():
    ext, grid = position_extractor(n_features=16), torch.tensor([[4, 4, 4]] * 2)
    a, b = ext({"grid": grid, "patches": [torch.rand(64, 8)] * 2})["rff"], ext({"grid": grid})["rff"]
    assert a.shape == (2, 64, 32) and torch.equal(a, b) and torch.equal(a[0], a[1])
    assert len(torch.unique(a[0], dim=0)) == 64                  # every grid cell distinct
