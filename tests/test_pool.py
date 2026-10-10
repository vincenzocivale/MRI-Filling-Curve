"""Pretraining pool (data/pool.py), cache-free loading of zip members, and our benchmark wrapper's geometry."""
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from sfc_gdn2.bench.geom import source_to_input
from sfc_gdn2.data import pool
from sfc_gdn2.data.volume import VolumeStore, patch_grid


def _zip(tmp_path, members: dict) -> str:
    a = tmp_path / "ses-01.zip"
    with zipfile.ZipFile(a, "w") as zf:
        for name, img in members.items():
            nib.save(img, tmp_path / name)
            zf.write(tmp_path / name, f"ses-01/anat/{name}")
    return str(a)


def test_select_keeps_pool_datasets_phase1_and_refuses_downstream_subjects():
    cat = pd.DataFrame({"dataset": ["MR-RATE", "MR-RATE", "FOMO300K", "IXI"], "subject": ["m:1", "m:2", "f:1", "i:1"],
                        "path": ["a", "b", "c", "d"], "use_phase1": [True, False, np.nan, np.nan]})
    df = pool.select(cat, ["MR-RATE", "FOMO300K"], ["i:1"])
    assert df["path"].tolist() == ["a", "c"] and (df["split"] == "pretrain").all() and df["sample_id"].is_unique
    with pytest.raises(RuntimeError, match="downstream"):
        pool.select(cat, ["MR-RATE", "FOMO300K"], ["f:1"])


def test_with_patches_reads_zip_headers_and_drops_multi_frame_scans(tmp_path):
    vol = nib.Nifti1Image(np.zeros((40, 33, 10), np.int16), np.diag([1.0, 1.0, 4.0, 1.0]))
    one = nib.Nifti1Image(np.zeros((8, 8, 8, 1), np.int16), np.eye(4))
    dwi = nib.Nifti1Image(np.zeros((8, 8, 8, 3), np.int16), np.eye(4))
    a = _zip(tmp_path, {"a_T1w.nii.gz": vol, "b_T1w.nii.gz": one, "c_dwi.nii.gz": dwi})
    df = pd.DataFrame({"path": [f"zip://{a}::ses-01/anat/{n}" for n in ("a_T1w.nii.gz", "b_T1w.nii.gz", "c_dwi.nii.gz")]
                       + [str(tmp_path / "missing.nii.gz")]})
    out = pool.with_patches(df, 16.0, workers=1)
    assert out["path"].str.endswith(("a_T1w.nii.gz", "b_T1w.nii.gz")).all() and len(out) == 2
    assert out["n_patch"].tolist() == [int(patch_grid((40, 33, 10), (16, 16, 4)).prod()), 1]


def test_store_without_cache_reads_zip_members_in_memory(tmp_path):
    raw = np.random.default_rng(0).integers(1, 3000, (6, 5, 4, 1)).astype(np.int16)
    a = _zip(tmp_path, {"x.nii.gz": nib.Nifti1Image(raw, np.diag([-0.7, 0.7, 5.0, 1.0]))})
    path = f"zip://{a}::ses-01/anat/x.nii.gz"
    before = sorted(tmp_path.rglob("*"))
    arr, _, aff = VolumeStore(None).load(path)
    assert sorted(tmp_path.rglob("*")) == before                     # nothing extracted or cached
    cached = VolumeStore(tmp_path / "cache").load(path)
    assert np.array_equal(arr, raw[::-1, :, :, 0]) and np.array_equal(arr, cached[0]) and aff == pytest.approx(cached[2])


def test_sfc_wrapper_maps_source_voxels_to_its_canonical_grid(tmp_path):
    from sfc_gdn2.fm.sfc import SFC

    run = tmp_path / "out" / "lejepa-abc"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"seed": 0, "data": {"patch_mm": 2.0},
                                                     "objective": {"view_curves": ["raster", "hilbert"]}}))
    for s in (0, 5):
        (run / f"encoder_step{s:06d}.pt").touch()
    raw = np.arange(4 * 6 * 5, dtype=np.int16).reshape(4, 6, 5) + 1
    affine = np.array([[0, 0, -1.0, 0], [1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1]])  # permuted + flipped axes
    nib.save(nib.Nifti1Image(raw, affine), tmp_path / "x.nii.gz")
    w = SFC({"model": "sfc_gdn2", "checkpoint": str(tmp_path / "out")}, "cpu")
    assert w.ckpt.name == "encoder_step000005.pt" and w.curves == ["raster", "hilbert"]
    p = w.preprocess(str(tmp_path / "x.nii.gz"))
    m = source_to_input("sfc_gdn2", p["meta"])
    vol = p["volume"].numpy()
    for i, j, k in [(0, 0, 0), (3, 5, 4), (1, 2, 3)]:
        c = (m @ [i, j, k, 1])[:3].round().astype(int)
        assert vol[tuple(c)] == raw[i, j, k]
    assert torch.equal(p["grid"], patch_grid(vol.shape, (2, 2, 1)))  # RAS: 1, 1, 2 mm
