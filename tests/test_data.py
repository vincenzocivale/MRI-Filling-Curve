
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from sfc_gdn2.data.dataset import BudgetBatches, PatchDataset, collate, to_device
from sfc_gdn2.data.volume import VolumeStore, patch_dims, patch_grid, patchify


def frame(n_cohorts=3, subjects=20, sessions=2):
    return pd.DataFrame([{"dataset": "d", "cohort": f"c{c}", "subject": f"c{c}_{s:03d}",
                          "sample_id": f"c{c}_{s:03d}_{ses}", "session": ses, "path": f"c{c}_{s:03d}_{ses}",
                          "sex": "F" if s % 2 else "M"}
                         for c in range(n_cohorts) for s in range(subjects) for ses in range(sessions)])


def test_patchify_matches_unfold():
    vol = torch.randn(16, 16, 16)
    ref = vol.unfold(0, 4, 4).unfold(1, 4, 4).unfold(2, 4, 4).reshape(64, 64)
    assert torch.equal(patchify(vol, 4), ref)


def test_patchify_anisotropic_patches_pad_the_end_only():
    vol = torch.arange(5 * 3 * 7.0).view(5, 3, 7) + 1
    p = patchify(vol, (2, 3, 4))
    assert p.shape == (3 * 1 * 2, 24) and patch_grid(vol.shape, (2, 3, 4)).tolist() == [3, 1, 2]
    assert torch.equal(p[0], vol[:2, :, :4].flatten()) and torch.equal(p[1], torch.cat(
        [vol[:2, :, 4:], torch.zeros(2, 3, 1)], -1).flatten())
    assert p.sum() == vol.sum()                                      # every voxel exactly once


def test_patch_dims_are_about_patch_mm_of_native_voxels():
    assert patch_dims((1.2, 0.5, 6.0), 16) == (13, 32, 3) and patch_dims((28.0, 0.156, 2.0), 16) == (1, 103, 8)


class FakeStore(VolumeStore):
    def __init__(self):
        super().__init__(".")

    def load(self, path):
        return np.full((8, 4, 12), hash(path) % 7, dtype=np.int16), (4.0, 4.0, 4.0), (0.5, 1.0)


def test_patch_dataset_keeps_stored_voxels_and_patches_on_device():
    df = frame()
    ds = PatchDataset(df, FakeStore(), 16, target=lambda row: torch.tensor([len(row["subject"])]))
    item = ds[3]
    assert item["volume"].dtype == torch.int16 and item["volume"].shape == (8, 4, 12)
    assert item["grid"].tolist() == [2, 1, 3] and item["k"].tolist() == [4, 4, 4]
    assert item["index"] == 3 and item["target"].item() == len(df.subject[3])
    vols = to_device(collate([ds[0], ds[3]]), "cpu")
    assert [p.shape for p in vols["patches"]] == [(6, 64)] * 2
    assert torch.equal(vols["patches"][1], patchify(item["volume"].float() * 0.5 + 1.0, 4))


def test_volume_store_is_exact_and_canonical(tmp_path):
    raw = np.random.default_rng(0).integers(-300, 3000, (6, 5, 4)).astype(np.int16)
    affine = np.diag([-0.7, 0.7, 5.0, 1.0])                           # x flipped (LAS), thick z
    img = nib.Nifti1Image(raw, affine)
    img.header.set_slope_inter(0.25, 10.0)
    nib.save(img, tmp_path / "x.nii.gz")
    store = VolumeStore(tmp_path / "cache")
    for _ in range(2):                                                # decode, then the cached copy
        arr, spacing, (a, c) = store.load(str(tmp_path / "x.nii.gz"))
        assert arr.dtype == np.int16 and spacing == pytest.approx((0.7, 0.7, 5.0))
        ref = nib.as_closest_canonical(nib.load(tmp_path / "x.nii.gz"))
        val = np.asarray(ref.dataobj, dtype=np.float64)
        assert np.array_equal(arr, raw[::-1]) and np.allclose(arr * a + c, val / np.percentile(val[val > 0], 99))


def test_budget_batches_keep_scans_whole_and_respect_the_budget():
    sizes = [3000] * 20 + [50000, 9000]
    batches = list(BudgetBatches(sizes, 16384, 6, seed=0))
    seen = [i for b in batches for i in b]
    assert len(seen) == len(set(seen)) and len(sizes) - len(seen) <= 6       # once each; last partial batch dropped
    for b in batches:
        assert len(b) <= 6 and (sum(sizes[i] for i in b) <= 16384 or len(b) == 1)
    assert all(b == [20] for b in batches if 20 in b)                          # bigger than the budget: alone
