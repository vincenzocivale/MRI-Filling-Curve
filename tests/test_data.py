import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from sfc_gdn2.data.dataset import PatchDataset
from sfc_gdn2.data.splits import assign_splits
from sfc_gdn2.data.volume import VolumeStore, fit_cube, patchify


def frame(n_cohorts=3, subjects=20, sessions=2):
    return pd.DataFrame([{"dataset": "d", "cohort": f"c{c}", "subject": f"c{c}_{s:03d}",
                          "sample_id": f"c{c}_{s:03d}_{ses}", "session": ses, "path": f"c{c}_{s:03d}_{ses}",
                          "sex": "F" if s % 2 else "M"}
                         for c in range(n_cohorts) for s in range(subjects) for ses in range(sessions)])


COUNTS = {"pretrain": 15, "train": 24, "val": 3, "test": 12}


def test_splits_have_exact_counts_one_scan_per_subject():
    df = frame()
    df["split"] = assign_splits(df, 0, COUNTS, "sex")
    used = df[df.split != "unused"]
    assert used.split.value_counts().to_dict() == COUNTS
    assert used.subject.is_unique


def test_splits_are_subject_disjoint_and_stratified():
    df = frame()
    df["split"] = assign_splits(df, 0, COUNTS, "sex")
    used = df[df.split != "unused"]
    groups = {k: set(g.subject) for k, g in used.groupby("split")}
    for a, b in itertools.combinations(groups, 2):
        assert not groups[a] & groups[b]
    assert (used[used.split == "test"].cohort.value_counts() == 4).all()


def test_probe_splits_only_draw_labelled_subjects():
    df = frame()
    df.loc[df.subject.isin([f"c0_{i:03d}" for i in range(10)]), "sex"] = None
    df["split"] = assign_splits(df, 0, COUNTS, "sex")
    probe = df[df.split.isin(["train", "val", "test"])]
    assert probe.sex.notna().all()


def test_pretrain_all_takes_every_scan_outside_the_probe_splits():
    df = frame()
    probe_only = {k: v for k, v in COUNTS.items() if k != "pretrain"}
    ref = assign_splits(df, 0, probe_only, "sex")
    out = assign_splits(df, 0, {**probe_only, "pretrain": "all"}, "sex")
    assert (out[ref != "unused"] == ref[ref != "unused"]).all()       # probe splits unchanged
    probe_subjects = set(df.subject[ref != "unused"])
    assert (out[~df.subject.isin(probe_subjects)] == "pretrain").all()  # all sessions included
    assert not (out[df.subject.isin(probe_subjects)] == "pretrain").any()


def test_splits_reject_impossible_counts():
    with pytest.raises(ValueError):
        assign_splits(frame(), 0, {"pretrain": 0, "train": 500, "val": 1, "test": 1}, "sex")


def test_patchify_matches_unfold():
    vol = torch.randn(16, 16, 16)
    ref = vol.unfold(0, 4, 4).unfold(1, 4, 4).unfold(2, 4, 4).reshape(64, 64)
    assert torch.equal(patchify(vol, 4), ref)


class FakeStore(VolumeStore):
    def __init__(self):
        super().__init__(".", (8, 8, 8))

    def load(self, path):
        return np.full(self.shape, hash(path) % 7, dtype=np.float16)


def test_patch_dataset_returns_index_and_worker_target():
    df = frame()
    ds = PatchDataset(df, FakeStore(), 4, target=lambda row: torch.tensor(len(row["subject"])))
    item = ds[3]
    assert item["patches"].shape == (8, 64) and item["patches"].dtype == torch.float16
    assert item["index"] == 3 and item["target"].item() == len(df.subject[3])


def test_fit_cube_keeps_aspect_ratio():
    vol = np.ones((40, 10, 20), np.float32)
    cube = fit_cube(vol, (1.0, 2.0, 1.0), 16)
    filled = np.argwhere(cube > 0.5)
    assert cube.shape == (16, 16, 16)
    assert (filled.max(0) - filled.min(0) + 1).tolist() == [16, 8, 8]


def test_volume_store_caches_decoded_cube(tmp_path, monkeypatch):
    store = VolumeStore(tmp_path, (8, 8, 8))
    calls = []
    monkeypatch.setattr(store, "_decode", lambda p: calls.append(p) or np.ones((8, 8, 8), np.float16))
    a, b = store.load("x.nii.gz"), store.load("x.nii.gz")
    assert calls == ["x.nii.gz"] and np.array_equal(a, b)
