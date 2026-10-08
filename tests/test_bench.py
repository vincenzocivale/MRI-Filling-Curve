from pathlib import Path

import pandas as pd

from sfc_gdn2 import bench


def test_splits_are_subject_level_stratified_and_keep_totalseg_test(tmp_path):
    rows = [{"dataset": "IXI", "subject": f"IXI:{i}", "session": s, "modality": "T1w", "variant": "T1",
             "path": f"/d/{i}_{s}.nii.gz", "seg": None, "age": 20 + i, "sex": "MF"[i % 2]}
            for i in range(200) for s in ("1", "2")]
    rows += [{"dataset": "TotalSegmentator-MRI", "subject": f"TotalSegmentator-MRI:s{i}", "session": "",
              "modality": "other", "variant": "SE", "path": f"/t/{i}.nii.gz", "seg": "/t/seg", "age": None,
              "sex": None} for i in range(20)]
    cat = pd.DataFrame(rows).reindex(columns=[*pd.DataFrame(rows).columns, "seg_source", *bench.LABELS])
    cat = cat.loc[:, ~cat.columns.duplicated()]
    pd.DataFrame({"image_id": [f"s{i}" for i in range(20)], "split": ["test"] * 4 + ["train"] * 16}).to_csv(
        tmp_path / "meta.csv", sep=";", index=False)
    vols = bench.volumes(cat)
    spl = bench.splits(vols, tmp_path / "meta.csv", seeds=2)
    assert vols["dense"].sum() == 20 and spl["subject"].is_unique
    ixi = spl[spl.dataset == "IXI"]["split_s0"].value_counts()
    assert (ixi["train"], ixi["val"], ixi["test"]) == (140, 20, 40)
    ts = spl[spl.dataset == "TotalSegmentator-MRI"].set_index("subject")
    assert (ts.loc[[f"TotalSegmentator-MRI:s{i}" for i in range(4)], "split_s1"] == "test").all()
    assert (spl["split_s0"] != spl["split_s1"]).any()


def test_stage_converts_analyze_and_links_spaced_paths(tmp_path):
    import nibabel as nib
    import numpy as np
    a = np.arange(24, dtype=np.int16).reshape(2, 3, 4, 1)
    nib.save(nib.AnalyzeImage(a, np.eye(4)), tmp_path / "x.img")
    (tmp_path / "d d").mkdir()
    nib.save(nib.Nifti1Image(a[..., 0], np.eye(4)), tmp_path / "d d" / "y.nii.gz")
    nib.save(nib.Nifti1Image(a[..., 0], np.eye(4)), tmp_path / "z.nii.gz")
    vols = pd.DataFrame({"id": ["x", "y", "z"], "path": [str(tmp_path / "x.img"), str(tmp_path / "d d" / "y.nii.gz"),
                                                         str(tmp_path / "z.nii.gz")]})
    out = bench.stage(vols, tmp_path / "in").set_index("id")
    x = nib.load(out.at["x", "input"])
    assert x.shape == (2, 3, 4) and (np.asarray(x.dataobj) == a[..., 0]).all() and out.at["x", "shape"] == "2x3x4"
    assert " " not in out.at["y", "input"] and Path(out.at["y", "input"]).stat().st_nlink == 2
    assert out.at["z", "input"] == str(tmp_path / "z.nii.gz")
