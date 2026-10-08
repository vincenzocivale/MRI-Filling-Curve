"""`sfc fm` plumbing: config expansion, the wrapper contract and the in-env worker (with a fake model).
Each real wrapper is checked against its original pipeline by `tests/fm/<family>_parity.py`, run in
that model's own env."""
import json
import sys
import types

import pytest
import torch

from sfc_gdn2 import fm
from sfc_gdn2.fm import api, base, worker


class Fake(base.Wrapper):
    name = "fake"
    modalities = ("t1", "t2")

    def preprocess(self, image):
        return torch.ones(1, 2, 2, 2) * (2 if isinstance(image, dict) else 1)

    def features(self, x):
        return {"features": {"map": x, "embedding": x.flatten(1).mean(1)}, "canonical": "embedding",
                "meta": {"shape": list(x.shape)}}


@pytest.fixture
def fake_model(monkeypatch):
    mod = types.ModuleType("sfc_gdn2.fm.fake")
    mod.build = lambda cfg, device: Fake(cfg, device)
    monkeypatch.setitem(sys.modules, "sfc_gdn2.fm.fake", mod)
    monkeypatch.setitem(fm.MODELS, "fake", "fake")


def test_config_expands_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SFC_FM_ENVS", "/envs")
    p = tmp_path / "m.yaml"
    p.write_text("model: fake\nenv: ${SFC_FM_ENVS}/fm-fake\nrepo: /r\nargs: {paths: [$SFC_FM_ENVS/a]}\n")
    cfg = api.load_config(p)
    assert cfg["env"] == "/envs/fm-fake" and cfg["args"]["paths"] == ["/envs/a"]
    monkeypatch.delenv("SFC_FM_ENVS")
    with pytest.raises(KeyError, match="unset environment variable"):
        api.load_config(p)


def test_image_ids():
    assert api.image_id("/d/sub-01_T1w.nii.gz") == "sub-01_T1w"
    assert api.image_id({"t2": "/d/b_T2.nii", "t1": "/d/a_T1.nii.gz"}) == "a_T1"


def test_worker_writes_contract_output(tmp_path, fake_model):
    job = {"cfg": {"model": "fake", "repo": None}, "out": str(tmp_path / "o"), "device": "cpu",
           "items": [{"id": "a", "image": "/x/a.nii.gz"}, {"id": "b", "image": {"t1": "/x/b1", "t2": "/x/b2"}}]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    worker.main(str(tmp_path / "job.json"))
    a, b = (torch.load(tmp_path / "o" / f"{i}.pt") for i in "ab")
    assert a["canonical"] == "embedding" and a["features"]["embedding"].item() == 1
    assert b["features"]["embedding"].item() == 2 and b["derived"] == [] and b["model"] == "fake"


def test_contract_violations_raise(fake_model):
    w = Fake({"model": "fake"}, "cpu")
    with pytest.raises(ValueError, match="unknown modalities"):
        w({"flair": "/x"})
    w.features = lambda x: {"features": {"map": x}, "canonical": "embedding", "meta": {}}
    with pytest.raises(RuntimeError, match="contract"):
        w("/x")


def test_registry_modules_exist():
    from pathlib import Path
    pkg = Path(fm.__file__).parent
    missing = sorted({m for m in fm.MODELS.values() if not (pkg / f"{m}.py").exists()})
    assert not missing, f"registry points at missing wrapper modules: {missing}"


def test_save_trims_storage_only(tmp_path, fake_model):
    job = {"cfg": {"model": "fake", "repo": None, "save": {"features": [], "dtype": "float16"}},
           "out": str(tmp_path), "device": "cpu", "items": [{"id": "a", "image": "/x/a.nii.gz"}]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    worker.main(str(tmp_path / "job.json"))
    res = torch.load(tmp_path / "a.pt")
    assert list(res["features"]) == ["embedding"] and res["features"]["embedding"].dtype == torch.float16


def test_worker_group_shares_preprocessing_and_trims(tmp_path, fake_model):
    save = {"features": ["embedding"], "dense": ["map_pool2"], "dtype": "float16",
            "pool": [{"from": "map", "factor": 2, "to": "map_pool2"}]}
    cfgs = [{"model": "fake", "repo": None, "save": save}, {"model": "fake", "repo": None}]
    job = {"cfgs": cfgs, "outs": [str(tmp_path / "a"), str(tmp_path / "b")], "device": "cpu", "verify_shared": True,
           "items": [{"id": "x", "image": "/x.nii.gz", "dense": True}, {"id": "y", "image": "/y.nii.gz"}]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    worker.main(str(tmp_path / "job.json"))
    x, y, xb = (torch.load(tmp_path / d / f"{i}.pt") for d, i in (("a", "x"), ("a", "y"), ("b", "x")))
    assert set(x["features"]) == {"embedding", "map_pool2"} and x["features"]["map_pool2"].shape == (1, 1, 1, 1)
    assert set(y["features"]) == {"embedding"} and y["features"]["embedding"].dtype == torch.float16
    assert set(xb["features"]) == {"embedding", "map"} and "map_pool2" in x["derived"]
