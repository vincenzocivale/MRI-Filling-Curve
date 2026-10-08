"""One table of every raw MRI volume we have, with standardized labels (NaN = not available).

`python -m sfc_gdn2.data.catalog <raw_root> <mrrate_root> <mrrate_body_region.csv> <out.csv.gz> [brats_duplicates*.csv ...]`

Row = one volume: `dataset`, `cohort`, `subject` (global, `<source>:<id>`), `session`, `modality`
(canonical, see `modality`), `variant` (source file entities), `path` (file or `zip://archive::member`),
`seg` / `brain_mask` (paths), then labels. Shared labels use one encoding everywhere: `age` (years),
`sex` (M/F), `handedness` (R/L/A), `dx` (source diagnosis, lowercase), `dx_group` (control / tumor / stroke / dementia /
parkinson / adhd / autism / epilepsy / psychosis / depression / ms / tbi / other). Task labels:
`cdr`, `mmse`, `nihss`, `mrs90`, `idh`, `mgmt`, `codel_1p19q` (1 = mutated / methylated / co-deleted),
`who_grade`, `os_days`, `os_event`, `kps`; MR-RATE adds `pat_*` (37 official findings), `nvfm_*` (74),
`split_official`, `body_region`, `use_phase1`.

FOMO300K repackages several datasets we also hold in their original form (IXI, OASIS-1/2, CoRR NYU_2,
Calgary, and OpenNeuro Pixar/AOMIC/Long579/DLBS/SOOP). Their FOMO subjects are mapped back to the
original IDs (FOMO `mapping.tsv`), FOMO labels fill what the original lacks, and the FOMO rows are
dropped: one copy per subject. BraTS-GEN / MSD (FOMO) share no IDs with UCSF-PDGM / UPENN-GBM nor with each
other: their copies are found by image fingerprint (datasets-meta/scripts/brats_dedup.py) and dropped here.
Left out: HaN-Seg (head-neck), ATLAS (encrypted).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MODALITY = [  # first match on the lowercased file stem wins
    ("FLAIR", r"flair"), ("T1c", r"t1c|t1ce|t1gd|post|stealth"), ("MP2RAGE", r"mp2rage|unit1"),
    ("SWI", r"swi"), ("T2starw", r"t2star|_gre\b|gre$"), ("PDw", r"(^|_)pdw?$|-pd$"),
    ("ADC", r"adc"), ("DWI", r"dwi|bval|trace"), ("ASL", r"asl|m0scan|cbf|att$"), ("angio", r"angio|mra"),
    ("qmap", r"map$"),
    ("T2w", r"t2"), ("T1w", r"t1|mpr|mprage|anat$"),
]
_DX_GROUP = [  # coarse diagnosis shared across sources; `dx` keeps the source wording
    ("control", (r"^(controls?|normal|healthy( control)?|health control|hc|cn|nh|nondemented|neurotypical|"
                 r"never-depressed control|typically developing|td)$")),
    ("tumor", (r"tumou?r|gliom|glioblastoma|gmb|astrocytom|oligodendro|oligoastro|meningiom|ependymom|medulloblastom|"
               r"lymphoma|metasta|adenoma|dnet|ganglioglioma|neuroectodermal|glioneuronal")),
    ("stroke", r"stroke|infarct"), ("dementia", r"dementia|demented|^ad$|ftd|alzheimer|converted"),
    ("parkinson", r"parkinson|^pd$"), ("adhd", r"adhd|attention"), ("autism", r"autis"),
    ("epilepsy", r"epilep|cortical dysplasia"), ("psychosis", r"schiz|schz|psychos"), ("depression", r"depress"),
    ("ms", r"^ms$|multiple sclerosis"), ("tbi", r"^tbi$|traumatic brain"),
]


def modality(stem: str) -> str:
    s = stem.lower()
    return next((m for m, rx in _MODALITY if re.search(rx, s)), "other")


def sex(v) -> str | float:
    s = str(v).strip().lower()
    return "M" if s in {"m", "male", "man"} else "F" if s in {"f", "female", "woman"} else np.nan


def hand(v) -> str | float:
    s = str(v).strip().lower()
    return {"r": "R", "right": "R", "l": "L", "left": "L", "a": "A", "ambi": "A", "mixed": "A"}.get(s, np.nan)


def dx(v) -> str | float:
    return np.nan if pd.isna(v) else re.sub(r"\s+", " ", str(v).strip().lower()).replace("’", "'")


def dx_group(v) -> str | float:
    return np.nan if pd.isna(v) else next((g for g, rx in _DX_GROUP if re.search(rx, v)), "other")


def dicom_age(v) -> float:
    """'025Y' / '006M' / '010W' / '003D' -> years."""
    m = re.fullmatch(r"(\d+)([YMWD])", str(v).strip())
    return int(m[1]) / {"Y": 1, "M": 12, "W": 52.1775, "D": 365.25}[m[2]] if m else np.nan


def _ses(v) -> str:
    s = str(v).removeprefix("ses-")
    return s.lstrip("0") or "0" if s.isdigit() else s


def _num(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.where(x >= 0)


# ---- BIDS datasets (raw/<name>/[site/]sub-*/[ses-*/]anat/*.nii.gz)

def _bids(root: Path, name: str) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("**/anat/*.nii*")):
        if "derivatives" in p.parts:
            continue
        stem = p.name.split(".nii")[0]
        sub = re.search(r"sub-([A-Za-z0-9]+)", stem)[1]
        ses = re.search(r"ses-([A-Za-z0-9]+)", stem)
        variant = "_".join(e for e in stem.split("_") if not e.startswith(("sub-", "ses-")))
        rows.append({"dataset": name, "cohort": name, "subject": f"{name}:{sub}", "_id": sub,
                     "session": _ses(ses[1]) if ses else "", "modality": modality(variant), "variant": variant,
                     "path": str(p)})
    return pd.DataFrame(rows)


def _join(df: pd.DataFrame, meta: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    return df.merge(meta.drop_duplicates(on), on=on, how="left")


def abide(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "ABIDE-I", "ABIDE-I")
    ph = pd.read_csv(raw / "ABIDE-I/phenotypic/Phenotypic_V1_0b.csv")
    meta = pd.DataFrame({"_id": ph.SUB_ID.astype(str).str.zfill(7), "age": _num(ph.AGE_AT_SCAN),
                         "sex": ph.SEX.map({1: "M", 2: "F"}), "handedness": ph.HANDEDNESS_CATEGORY.map(hand),
                         "dx": ph.DX_GROUP.map({1: "autism", 2: "control"})})
    return _join(df, meta, ["_id"])


def aomic(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "AOMIC-ID1000", "AOMIC-ID1000")
    p = pd.read_csv(raw / "AOMIC-ID1000/participants.tsv", sep="\t")
    meta = pd.DataFrame({"_id": p.participant_id.str.removeprefix("sub-"), "age": _num(p.age),
                         "sex": p.sex.map(sex), "handedness": p.handedness.map(hand)})
    return _join(df, meta, ["_id"])


def calgary(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "Calgary", "Calgary")
    p = pd.read_csv(raw / "Calgary/participants.tsv", sep="\t")
    # sex 1 = M, 0 = F (checked against FOMO300K's copy); handedness coding undocumented -> not used
    meta = pd.DataFrame({"_id": p.participant_id.astype(str), "session": p.session.astype(str),
                         "age": _num(p.age), "sex": p.sex.map({1: "M", 0: "F"}),
                         "dx": np.where(p.dev_disorder == 1, "developmental disorder", "control")})
    return _join(df, meta, ["_id", "session"])


def corr_nyu2(raw: Path) -> pd.DataFrame:  # participants.tsv has handedness "#" only; FOMO300K's copy has age/sex
    return _bids(raw / "CoRR-NYU2", "CoRR")


def dlbs(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "DLBS", "DLBS")
    p = pd.read_csv(raw / "DLBS/participants.tsv", sep="\t")
    meta = pd.concat([pd.DataFrame({"_id": p.participant_id.str.removeprefix("sub-"), "session": f"wave{w}",
                                    "age": _num(p[f"AgeMRI_W{w}"]), "sex": p.Sex.map(sex),
                                    "mmse": _num(p[f"MMSE_W{w}"])}) for w in (1, 2, 3)])
    return _join(df, meta, ["_id", "session"])


def long579(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "Long579", "Long579")
    p = pd.read_csv(raw / "Long579/participants.tsv", sep="\t")
    birth = pd.to_datetime(p.birthdate, errors="coerce")
    meta = pd.concat([pd.DataFrame({"_id": p.participant_id.astype(str), "session": s, "sex": p.sex.map(sex),
                                    "age": (pd.to_datetime(p[f"ses-{s}_date_ST"], errors="coerce") - birth).dt.days / 365.25,
                                    "handedness": pd.Series(np.nan, index=p.index)})
                      for s in ("5", "7", "9")])
    return _join(df, meta, ["_id", "session"])


def pixar(raw: Path) -> pd.DataFrame:
    df = _bids(raw / "Pixar", "Pixar")
    p = pd.read_csv(raw / "Pixar/participants.tsv", sep="\t")
    meta = pd.DataFrame({"_id": p.participant_id.str.removeprefix("sub-"), "age": _num(p.Age),
                         "sex": p.Gender.map(sex), "handedness": p.Handedness.map(hand)})
    return _join(df, meta, ["_id"])


def soop(raw: Path) -> pd.DataFrame:
    """`seg`: lesion mask coregistered to this T1w/FLAIR (datasets-meta/scripts/soop_coreg.py; the released masks
    are in DWI TRACE space), only where its QC is `reliable`."""
    df = _bids(raw / "SOOP", "SOOP")
    p = pd.read_csv(raw / "SOOP/participants.tsv", sep="\t")
    meta = pd.DataFrame({"_id": p.participant_id.str.removeprefix("sub-"), "age": _num(p.age),
                         "sex": p.sex.map(sex), "nihss": _num(p.nihss), "mrs90": _num(p.gs_rankin_6isdeath),
                         "dx": np.where(p.acuteischaemicstroke == 1, "acute ischemic stroke", None)})
    df = _join(df, meta, ["_id"])
    coreg = raw.parent / "derivatives/SOOP-lesion-coreg"
    if (coreg / "qc.csv").exists():
        qc = pd.read_csv(coreg / "qc.csv")
        ok = set(zip(qc.subject[qc.reliable], qc.target[qc.reliable]))
        df["seg"] = [str(coreg / f"sub-{i}/anat/sub-{i}_space-{m}_desc-lesion_mask.nii.gz")
                     if (f"sub-{i}", m) in ok and (coreg / f"sub-{i}/anat/sub-{i}_space-{m}_desc-lesion_mask.nii.gz").exists()
                     else "" for i, m in zip(df._id, df.variant)]
    return df


def unlabelled(raw: Path, name: str) -> pd.DataFrame:  # SALD, Petfrog
    return _bids(raw / name, name)


# ---- non-BIDS datasets

def ixi(raw: Path) -> pd.DataFrame:  # labels: IXI.xls needs xlrd; FOMO300K's copy has age/sex
    rows = [{"dataset": "IXI", "cohort": "IXI", "_id": (i := p.name.split("-")[0]), "subject": f"IXI:{i}",
             "session": "", "modality": modality(p.name.split(".nii")[0].rsplit("-", 1)[1]),
             "variant": p.name.split(".nii")[0].rsplit("-", 1)[1], "path": str(p)}
            for p in sorted((raw / "IXI").glob("IXI-*/*.nii.gz"))]
    return pd.DataFrame(rows)


def oasis1(raw: Path) -> pd.DataFrame:
    rows = []
    for txt in sorted((raw / "OASIS-1").glob("disc*/OAS1_*_MR*/OAS1_*_MR*.txt")):
        f = dict(re.findall(r"^([A-Z/]+):\s*(\S.*?)\s*$", txt.read_text(), re.MULTILINE))
        sid, ses = txt.stem.rsplit("_", 1)
        for img in sorted(txt.parent.glob("RAW/*_mpr-*_anon.img")):
            rows.append({"dataset": "OASIS-1", "cohort": "OASIS-1", "_id": sid, "subject": f"OASIS-1:{sid}",
                         "session": ses, "modality": "T1w", "variant": re.search(r"mpr-\d+", img.name)[0],
                         "path": str(img), "age": f.get("AGE"), "sex": sex(f.get("M/F")),
                         "handedness": hand(f.get("HAND")), "cdr": f.get("CDR"), "mmse": f.get("MMSE")})
    df = pd.DataFrame(rows)
    for c in ("age", "cdr", "mmse"):
        df[c] = _num(df[c])
    return df


def oasis2(raw: Path) -> pd.DataFrame:  # no labels downloaded; FOMO300K's copy has age/sex/group
    rows = [{"dataset": "OASIS-2", "cohort": "OASIS-2", "_id": (s := img.parts[-3].rsplit("_", 1))[0],
             "subject": f"OASIS-2:{s[0]}", "session": s[1], "modality": "T1w",
             "variant": img.name.split(".")[0], "path": str(img)}
            for img in sorted((raw / "OASIS-2").glob("OAS2_RAW_PART*/OAS2_*_MR*/RAW/mpr-*.nifti.img"))]
    return pd.DataFrame(rows)


def ucsf_pdgm(raw: Path) -> pd.DataFrame:
    base = raw / "UCSF-PDGM/PKG - UCSF-PDGM Version 5/UCSF-PDGM-v5"
    keep = re.compile(r"_(T1|T1c|T2|FLAIR|SWI|DWI|ADC|ASL)(_bias)?\.nii\.gz$")
    rows = []
    for case in sorted(base.glob("UCSF-PDGM-*_nifti")):
        cid, _, fu = case.name.split("-")[2].removesuffix("_nifti").partition("_")  # "0429_FU003d": follow-up
        cid = str(int(cid))
        seg = case / f"{case.name.removesuffix('_nifti')}_tumor_segmentation.nii.gz"
        for p in sorted(case.iterdir()):
            if m := keep.search(p.name):
                rows.append({"dataset": "UCSF-PDGM", "cohort": "UCSF-PDGM", "_id": cid, "subject": f"UCSF-PDGM:{cid}",
                             "session": fu, "modality": modality(m[1]), "variant": m[0][1:].split(".")[0],
                             "path": str(p), "seg": str(seg) if seg.exists() else ""})
    df = pd.DataFrame(rows)
    c = pd.read_csv(raw / "UCSF-PDGM/clinical/UCSF-PDGM-metadata_v5.csv")
    meta = pd.DataFrame({
        "_id": c.ID.str.extract(r"UCSF-PDGM-(\d+)")[0].astype(int).astype(str), "session": c.ID.str.extract(r"_(FU\w+)")[0].fillna(""),
        "age": _num(c["Age at MRI"]),
        "sex": c.Sex.map(sex), "dx": c["Final pathologic diagnosis (WHO 2021)"].map(dx),
        "who_grade": _num(c["WHO CNS Grade"]),
        "idh": c.IDH.map(lambda v: np.nan if pd.isna(v) else 0.0 if v == "wildtype" else 1.0),
        "mgmt": c["MGMT status"].map({"positive": 1.0, "negative": 0.0}),
        "codel_1p19q": c["1p/19q"].map({"co-deletion": 1.0, "intact": 0.0}),
        "os_days": _num(c.OS), "os_event": _num(c["1-dead 0-alive"])})
    return _join(df, meta, ["_id", "session"])


def upenn_gbm(raw: Path) -> pd.DataFrame:
    """TCIA NIfTI package (SRI24 240x240x155, skull-stripped, CaPTk = BraTS preprocessing): `<id>_11` baseline,
    `<id>_21` follow-up. `seg`: manual segmentation where released (147), else the automated one (`seg_source`).
    The IDC DICOM series (nifti/, native space, datasets-meta/scripts/convert_upenn.sh) are the same scans."""
    base = raw / "UPENN-GBM/nifti_sri24/PKG - UPENN-GBM-NIfTI/UPENN-GBM/NIfTI-files"
    rows = []
    for d in sorted((base / "images_structural").glob("UPENN-GBM-*")):
        pid, ses = d.name.rsplit("_", 1)
        man, auto = base / f"images_segm/{d.name}_segm.nii.gz", base / f"automated_segm/{d.name}_automated_approx_segm.nii.gz"
        seg, src = (man, "manual") if man.exists() else (auto, "automated") if auto.exists() else ("", "")
        for m in ("T1", "T1GD", "T2", "FLAIR"):
            rows.append({"dataset": "UPENN-GBM", "cohort": "UPENN-GBM", "_id": d.name, "subject": f"UPENN-GBM:{pid}",
                         "session": ses, "modality": {"T1": "T1w", "T1GD": "T1c", "T2": "T2w", "FLAIR": "FLAIR"}[m],
                         "variant": m, "path": str(d / f"{d.name}_{m}.nii.gz"), "seg": str(seg), "seg_source": src})
    df = pd.DataFrame(rows)
    c = pd.read_csv(raw / "UPENN-GBM/clinical/UPENN-GBM_clinical_info_v2.1.csv")
    meta = pd.DataFrame({
        "_id": c.ID, "age": _num(c.Age_at_scan_years), "sex": c.Gender.map(sex),
        "dx": "glioblastoma", "idh": c.IDH1.map({"Mutated": 1.0, "Wildtype": 0.0}),
        "mgmt": c.MGMT.map({"Methylated": 1.0, "Unmethylated": 0.0}),
        "os_days": _num(c.Survival_from_surgery_days_UPDATED),
        "os_event": c.Survival_Status.map(lambda v: 1.0 if str(v).startswith("Deceased") else 0.0 if v == "Alive" else np.nan),
        "kps": _num(c.KPS)})
    return _join(df, meta, ["_id"])


def totalseg(raw: Path) -> pd.DataFrame:
    base = raw / "TotalSegmentator/MRI_v300/Totalsegmentator_dataset_v300"
    m = pd.read_csv(base / "meta.csv", sep=";", encoding="utf-8-sig")
    return pd.DataFrame({"dataset": "TotalSegmentator-MRI", "cohort": "TotalSegmentator-MRI",
                         "subject": "TotalSegmentator-MRI:" + m.image_id, "session": "", "modality": "other",
                         "variant": m.scanning_sequence.astype(str), "path": [str(base / i / "mri.nii.gz") for i in m.image_id],
                         "seg": [str(base / i / "segmentations") for i in m.image_id],
                         "age": _num(m.age), "sex": m.gender.map(sex)})


def mrrate(root: Path) -> pd.DataFrame:
    cols = ["patient_uid", "study_uid", "series_id", "classified_modality", "Patient'sAge", "Patient'sSex"]
    meta = pd.concat([pd.read_csv(f, usecols=cols, dtype=str).assign(_batch=f.name[:7])
                      for f in sorted((root / "metadata").glob("batch*_metadata.csv"))])
    zp = "zip://" + str(root / "mri") + "/" + meta._batch + "/" + meta.study_uid + ".zip::" + meta.study_uid
    df = pd.DataFrame({
        "dataset": "MR-RATE", "cohort": "MR-RATE", "subject": "MR-RATE:" + meta.patient_uid, "session": meta.study_uid,
        "modality": meta.classified_modality.replace({"MRA": "angio"}), "variant": meta.series_id,
        "path": zp + "/img/" + meta.study_uid + "_" + meta.series_id + ".nii.gz",
        "brain_mask": zp + "/seg/" + meta.study_uid + "_" + meta.series_id + "_brain-mask.nii.gz",
        "age": meta["Patient'sAge"].map(dicom_age), "sex": meta["Patient'sSex"].map(sex),
        "patient_uid": meta.patient_uid, "study_uid": meta.study_uid, "series_id": meta.series_id})
    return df


def mrrate_labels(df: pd.DataFrame, root: Path, region_csv: Path) -> pd.DataFrame:
    pat = pd.read_csv(root / "pathology_labels/mrrate_labels.csv")
    pat.columns = ["study_uid"] + ["pat_" + re.sub(r"\W+", "_", c).strip("_").lower() for c in pat.columns[1:]]
    nv = pd.read_csv(root / "pathology_labels_neurovfm/mrrate_neurovfm74_labels.csv")
    nv.columns = ["study_uid"] + ["nvfm_" + c for c in nv.columns[1:]]
    split = pd.read_csv(root / "splits.csv")[["study_uid", "split"]].rename(columns={"split": "split_official"})
    region = pd.read_csv(region_csv, dtype={"patient_uid": str})[["study_uid", "series_id", "body_region", "use_phase1"]]
    for t in (pat, nv, split):
        df = df.merge(t.drop_duplicates("study_uid"), on="study_uid", how="left")
    return df.merge(region.drop_duplicates(["study_uid", "series_id"]), on=["study_uid", "series_id"], how="left") \
             .drop(columns=["patient_uid", "study_uid", "series_id"])


# ---- FOMO300K and deduplication

# FOMO cohort -> (original dataset name, regex on old_path giving (subject, session))
_OVERLAP = {
    "PT021_IXI": ("IXI", r"(IXI\d+)-()"),
    "PT028_OASIS1": ("OASIS-1", r"(OAS1_\d+)_(MR\d)"),
    "PT029_OASIS2": ("OASIS-2", r"(OAS2_\d+)_(MR\d)"),
    "PT014_CoRR": ("CoRR", r"^NYU_2_\w+/(\d+)/session_(\d+)"),
    "PT013_Calgary_Preschool": ("Calgary", r"/(\d{5})/(PS\d+_\d+)?"),  # CL_DEV_* sessions: subject only
    "PT030_OpenNeuro/ds000228": ("Pixar", r"sub-(pixar\d+)()"),
    "PT030_OpenNeuro/ds003097": ("AOMIC-ID1000", r"sub-(\d+)()"),
    "PT030_OpenNeuro/ds003604": ("Long579", r"sub-(\d+)_ses-(\d+)"),
    "PT030_OpenNeuro/ds004856": ("DLBS", r"sub-(\d+)_ses-(wave\d)"),
    "PT030_OpenNeuro/ds004889": ("SOOP", r"sub-(\d+)()"),
}


def fomo300k(root: Path) -> pd.DataFrame:
    m = pd.read_csv(root / "mapping.tsv", sep="\t")
    sub, rest = m.new_path.str.split("/", n=1).str[0], m.new_path.str.split("/", n=1).str[1]
    archive = str(root) + "/" + m.dataset + "/" + sub + "/" + m.session_id + ".zip"
    exists = {a: Path(a).exists() for a in archive.unique()}
    stem = m.new_filename.str.split(".nii").str[0]
    variant = stem.str.split("_").map(lambda e: "_".join(x for x in e if not x.startswith(("sub-", "ses-"))))
    df = pd.DataFrame({"dataset": "FOMO300K", "cohort": m.dataset,
                       "subject": "FOMO300K/" + m.dataset + ":" + m.participant_id.str.removeprefix("sub-"),
                       "session": m.session_id.map(_ses), "modality": variant.map(modality), "variant": variant,
                       "path": "zip://" + archive + "::" + rest, "old_path": m.old_path, "_fsub": m.participant_id,
                       "_fses": m.session_id, "_exists": archive.map(exists)})
    p = pd.read_csv(root / "participants.tsv", sep="\t", dtype=str)
    lab = pd.DataFrame({"cohort": p.dataset, "_fsub": p.participant_id, "_fses": p.session_id, "age": _num(p.age),
                        "sex": p.sex.map(sex), "handedness": p.handedness.map(hand), "dx": p.group.map(dx)})
    return _join(df, lab, ["cohort", "_fsub", "_fses"]).drop(columns=["_fsub", "_fses"])


def dedup(fomo: pd.DataFrame, orig: pd.DataFrame, raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map FOMO copies to their original subject/session, fill the originals' missing shared labels
    from them (downloaded or not), return (downloaded FOMO rows of subjects not held in original form,
    filled originals)."""
    calgary_ses = pd.read_csv(raw / "Calgary/participants.tsv", sep="\t").set_index("study_code").session.astype(str)
    held = set(orig.subject)
    src = pd.Series(pd.NA, index=fomo.index, dtype=object)
    ses = pd.Series(pd.NA, index=fomo.index, dtype=object)
    for cohort, (name, rx) in _OVERLAP.items():
        sel = fomo.cohort == cohort
        ids = fomo.loc[sel, "old_path"].str.extract(rx)
        src[sel] = (name + ":" + ids[0]).where(ids[0].notna())
        ses[sel] = ids[1].map(calgary_ses.to_dict().get) if name == "Calgary" else ids[1].map(lambda s: _ses(s) if isinstance(s, str) and s else "")
    dup = src.isin(held)
    fill = fomo[dup].assign(subject=src[dup], session=ses[dup])[["subject", "session", "age", "sex", "handedness", "dx"]]
    by_ses = fill.groupby(["subject", "session"]).first()
    by_sub = fill.groupby("subject")[["sex", "dx"]].first()  # session-invariant labels only
    key = pd.MultiIndex.from_frame(orig[["subject", "session"]])
    for c in ("age", "sex", "handedness", "dx"):
        if c not in orig:
            orig[c] = np.nan
        orig[c] = orig[c].where(orig[c].notna(), pd.Series(by_ses[c].reindex(key).to_numpy(), index=orig.index))
        if c in by_sub:
            orig[c] = orig[c].where(orig[c].notna(), orig.subject.map(by_sub[c]))
    return fomo[~dup & fomo._exists].drop(columns=["old_path", "_exists"]), orig


def build(raw: Path, mrrate_root: Path, region_csv: Path) -> pd.DataFrame:
    orig = pd.concat([abide(raw), aomic(raw), calgary(raw), corr_nyu2(raw), dlbs(raw), long579(raw), pixar(raw),
                      soop(raw), unlabelled(raw, "SALD"), unlabelled(raw, "Petfrog"), ixi(raw), oasis1(raw),
                      oasis2(raw), ucsf_pdgm(raw), upenn_gbm(raw), totalseg(raw)], ignore_index=True)
    fomo, orig = dedup(fomo300k(raw / "FOMO300K"), orig, raw)
    mr = mrrate_labels(mrrate(mrrate_root), mrrate_root, region_csv)
    df = pd.concat([orig.drop(columns="_id"), fomo, mr], ignore_index=True)
    df["age"] = df.age.where(df.age.between(0, 110))  # MR-RATE has DICOM ages up to 140
    df.insert(df.columns.get_loc("dx") + 1, "dx_group", df.dx.map(dx_group))
    if df.path.duplicated().any():
        raise RuntimeError(f"duplicate paths: {df.path[df.path.duplicated()].head().tolist()}")
    return df


if __name__ == "__main__":
    raw, mr, region, out = (Path(a) for a in sys.argv[1:5])
    df = build(raw, mr, region)
    for f in sys.argv[5:]:  # brats_duplicates*.csv: FOMO BraTS copies found by image fingerprint (no shared IDs)
        df = df[~df.subject.isin(pd.read_csv(f)["drop"])]
    df.to_csv(out, index=False)
    print(df.groupby("dataset").agg(volumes=("path", "size"), subjects=("subject", "nunique")).to_string())
