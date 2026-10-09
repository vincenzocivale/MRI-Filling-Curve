import numpy as np
import pandas as pd
import pytest
import torch

from sfc_gdn2 import bench_probe as bp


@pytest.fixture
def archive(tmp_path):
    """200 volumes / 100 subjects; feature `emb` carries every label, `noise` none; a map stored as [C, 2, 2, 2]."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    rows = pd.DataFrame({"id": [f"v{i:03d}" for i in range(n)], "subject": [f"s{i // 2}" for i in range(n)],
                         "dataset": np.where(np.arange(n) % 4 == 0, "A", "B"),
                         "sex": np.where(z > 0, "M", "F"), "grade": np.digitize(z, [-0.7, 0.7]) + 2,
                         "age": 50 + 10 * z, "os_days": np.exp(5 - z), "os_event": (np.arange(n) % 5 != 0) * 1.0})
    d = tmp_path / "m"
    d.mkdir()
    for i, zi in zip(rows["id"], z):
        emb = torch.tensor([zi, *rng.normal(size=3)], dtype=torch.float32)
        torch.save({"features": {"emb": emb, "noise": torch.randn(4), "map": torch.randn(4, 2, 2, 2)}}, d / f"{i}.pt")
    subjects = sorted(rows["subject"].unique())
    split = np.array(["train"] * 70 + ["val"] * 10 + ["test"] * 20)[np.random.default_rng(1).permutation(100)]
    splits = pd.DataFrame({"subject": subjects, "split_s0": split})
    return d, rows, splits


@pytest.mark.parametrize("cfg,key,good", [
    ({"type": "class", "label": "sex"}, "auroc", 0.9),
    ({"type": "ordinal", "label": "grade"}, "qwk", 0.7),
    ({"type": "dex", "label": "age", "bin": 1.0}, "neg_mae", -3.0),
    ({"type": "cox", "label": ["os_days", "os_event"]}, "c_index", 0.85),
])
def test_probe_recovers_the_signal_with_both_heads(archive, cfg, key, good):
    d, rows, splits = archive
    ids, feats = bp.load_globals(d)
    assert set(feats) == {"emb", "noise", "map"} and (d / "_globals.pt").exists()
    task = bp.task_rows(rows, splits, cfg, 0)
    res = bp.run_task("t", cfg, task, feats, {i: k for k, i in enumerate(ids)}, device="cpu")
    for head in ("linear", "mlp"):
        assert res[head]["feature"] == "emb" and res[head]["test"][key] > good, (head, res[head])
        lo, hi = res[head]["test_ci95"]
        assert lo <= res[head]["test"][key] <= hi
