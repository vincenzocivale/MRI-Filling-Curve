"""Downstream benchmark: which volumes get features, and the fixed subject-level splits.

`volumes(catalog)` keeps one volume per (subject, session, modality) of every downstream dataset (repeated runs
and reconstructions of the same acquisition are dropped, UCSF-PDGM `_bias` copies too) and flags the `dense`
ones, whose stride-8 maps are stored for segmentation. `splits(volumes)` assigns every subject to train / val /
test, 70 / 10 / 20, stratified within its dataset, for `seeds` independent draws (seed 0 is the primary one);
TotalSegmentator keeps its official test split and draws val from its official train. A subject is one unit
everywhere: all its sessions and sequences, in every dataset it appears in, share the split.

Both are written by `sfc bench-split <config>` and are fixed once for every model (ours included).
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOWNSTREAM = ("ABIDE-I", "AOMIC-ID1000", "Calgary", "DLBS", "IXI", "Long579", "OASIS-1", "OASIS-2", "Pixar",
              "SOOP", "UCSF-PDGM", "UPENN-GBM", "TotalSegmentator-MRI")
# tumour segmentation on the four structural sequences only; SOOP / UPENN: volumes with a mask
DENSE_UCSF = ("FLAIR", "T1c", "T1w", "T2w")
# position i of a dataset's stratum-ordered subject list -> split, a 10-cycle at 70/10/20
CYCLE = np.array(["train", "train", "test", "train", "val", "train", "train", "test", "train", "train"])
LABELS = ["age", "sex", "dx_group", "cdr", "mmse", "nihss", "mrs90", "who_grade", "idh", "mgmt", "codel_1p19q",
          "os_days", "os_event", "kps"]


def _safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(s)).strip("-") or "na"


def volumes(catalog: pd.DataFrame) -> pd.DataFrame:
    df = catalog[catalog["dataset"].isin(DOWNSTREAM)].copy()
    df = df[~((df["dataset"] == "UCSF-PDGM") & df["variant"].astype(str).str.endswith("_bias"))]
    df["session"] = df["session"].fillna("").astype(str)
    # preferred variant: not a re-reconstruction (Calgary rec-PURE), then the dataset's most common one, then name
    freq = df.groupby(["dataset", "variant"])["path"].transform("size")
    df["_rank"] = list(zip(df["variant"].astype(str).str.contains("rec-PURE"), -freq, df["variant"].astype(str)))
    df = (df.sort_values(["subject", "session", "modality", "_rank"])
            .drop_duplicates(["subject", "session", "modality"]).drop(columns="_rank"))
    df["id"] = [f"{_safe(d)}__{_safe(s.split(':', 1)[1])}__{_safe(ses)}__{_safe(v)}"
                for d, s, ses, v in zip(df["dataset"], df["subject"], df["session"], df["variant"])]
    if df["id"].duplicated().any():
        raise ValueError(f"duplicate volume ids: {df.loc[df['id'].duplicated(), 'id'].head().tolist()}")
    has_seg = df["seg"].notna()
    df["dense"] = ((df["dataset"] == "TotalSegmentator-MRI")
                   | ((df["dataset"] == "UCSF-PDGM") & df["modality"].isin(DENSE_UCSF))
                   | (df["dataset"].isin(["SOOP", "UPENN-GBM"]) & has_seg))
    abide = df["dataset"] == "ABIDE-I"
    df["site"] = ""
    df.loc[abide, "site"] = df.loc[abide, "path"].map(lambda p: Path(p).parts[-4])  # raw/ABIDE-I/<site>/sub-*/anat
    cols = ["id", "dataset", "subject", "session", "modality", "variant", "path", "seg", "seg_source", "dense",
            "site", *LABELS]
    return df[cols].sort_values("id").reset_index(drop=True)


def _txt(x: pd.Series) -> pd.Series:
    return pd.Series([str(v) if pd.notna(v) else "na" for v in x], index=x.index)


def strata(subj: pd.DataFrame) -> pd.Series:
    """One stratum string per subject row (first session's labels), chosen per dataset."""
    out = pd.Series("", index=subj.index)
    for ds, idx in subj.groupby("dataset").groups.items():
        g = subj.loc[idx]
        age = _txt(pd.cut(g["age"].rank(pct=True), [0, .2, .4, .6, .8, 1.0], labels=False))  # quintile in dataset
        cols = {"ABIDE-I": [g["site"], g["dx_group"]],
                "OASIS-1": [g["cdr"].gt(0).where(g["cdr"].notna()), age],
                "SOOP": [pd.cut(g["mrs90"], [-1, 2, 6], labels=["good", "poor"]), g["has_seg"]],
                "UCSF-PDGM": [g["who_grade"], g["idh"]],
                "UPENN-GBM": [g["manual_seg"], g["os_event"]],
                "TotalSegmentator-MRI": []}
        cols["OASIS-2"] = cols["OASIS-1"]
        parts = [_txt(c) for c in cols.get(ds, [age, g["sex"]])]   # default: healthy brain-age / sex cohorts
        out.loc[idx] = ["|".join(t) for t in zip(*parts)] if parts else ""
    return out


def splits(vols: pd.DataFrame, totalseg_meta: str | Path, seeds: int = 5) -> pd.DataFrame:
    """One row per subject: dataset (the first one it appears in), stratum, split_s0..split_s{seeds-1}."""
    v = vols.sort_values(["subject", "dataset", "session", "modality"])
    first = v.drop_duplicates("subject").set_index("subject")
    subj = first[["dataset", "site", *LABELS]].copy()
    subj["has_seg"] = v.groupby("subject")["seg"].apply(lambda x: x.notna().any())
    subj["manual_seg"] = v.groupby("subject")["seg_source"].apply(lambda x: (x == "manual").any())
    subj["stratum"] = strata(subj.reset_index()).to_numpy()
    meta = pd.read_csv(totalseg_meta, sep=";", encoding="utf-8-sig", usecols=["image_id", "split"])
    official = dict(zip("TotalSegmentator-MRI:" + meta["image_id"], meta["split"]))
    out = subj[["dataset", "stratum"]].copy()
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        col = pd.Series("", index=subj.index, dtype=object)
        for ds, g in subj.groupby("dataset"):
            pool = g.index.to_numpy()
            if ds == "TotalSegmentator-MRI":
                off = np.array([official[s] for s in pool])
                col.loc[pool[off == "test"]] = "test"
                train = rng.permutation(pool[off == "train"])
                n_val = round(len(pool) * 0.10)
                col.loc[train[:n_val]] = "val"
                col.loc[train[n_val:]] = "train"
                continue
            order = pool[np.lexsort((rng.random(len(pool)), g.loc[pool, "stratum"].to_numpy()))]  # stratum, then random
            start = int(rng.integers(len(CYCLE)))
            col.loc[order] = CYCLE[(np.arange(len(order)) + start) % len(CYCLE)]
        out[f"split_s{seed}"] = col
    return out.reset_index().sort_values(["dataset", "subject"]).reset_index(drop=True)


def stage(vols: pd.DataFrame, root: str | Path) -> pd.DataFrame:
    """Model input per volume (`input`) + its header (`shape`, `spacing`). The original file, except: Analyze
    .img / singleton-4D (OASIS) -> a 3D NIfTI copy under `root` (nibabel's affine); paths with spaces (UCSF, UPENN;
    MedicalNet splits its image list on spaces) -> a hard link under `root`. Same voxels for every model."""
    import nibabel as nib
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in zip(vols["id"], vols["path"]):
        img = nib.load(p)
        dst = root / f"{i}.nii.gz"
        if not p.endswith((".nii", ".nii.gz")) or len(img.shape) > 3:
            img = nib.squeeze_image(img)
            if len(img.shape) != 3:
                raise ValueError(f"{p}: not a 3D volume {img.shape}")
            if not dst.exists():
                nib.save(nib.Nifti1Image.from_image(img), dst)
        elif " " in p:
            dst = root / (i + (".nii.gz" if p.endswith(".gz") else ".nii"))
            if not dst.exists():
                os.link(p, dst)
        else:
            dst = Path(p)
        rows.append({"input": str(dst), "shape": "x".join(map(str, img.shape[:3])),
                     "spacing": "x".join(f"{float(z):.4g}" for z in img.header.get_zooms()[:3])})
    return vols.assign(**pd.DataFrame(rows, index=vols.index))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
