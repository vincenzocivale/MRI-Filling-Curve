from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

_ENTITY_RE = re.compile(r"sub-(?P<subject>[A-Za-z0-9]+)(?:_ses-(?P<session>[A-Za-z0-9]+))?")
_NIFTI_SUFFIXES = (".nii.gz", ".nii")


def _strip_nifti_suffix(name: str) -> str | None:
    for suf in _NIFTI_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return None


def _match_modality(stem: str, modalities: list[str]) -> str | None:
    if not modalities:
        return None
    for m in modalities:
        if stem.endswith(f"_{m}"):
            return m
    return None


def _find_mask(path: Path, stem: str) -> str | None:
    for cand_name in ("fb_mask.nii.gz", "deface_mask.nii.gz"):
        cand = path.parent / f"{stem}__Data" / cand_name
        if cand.exists():
            return str(cand)
    return None


def _scan_bids_like(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["root"])
    modalities = cfg.get("modalities") or []
    rows = []
    for path in sorted(root.rglob("*.nii*")):
        if "__Data" in path.parts:
            continue  # derivative/mask folders, not raw scans
        stem = _strip_nifti_suffix(path.name)
        if stem is None:
            continue
        modality = _match_modality(stem, modalities)
        if modality is None:
            continue
        m = _ENTITY_RE.search(path.name)
        if not m:
            continue
        subject, session = m.group("subject"), m.group("session")
        mask_path = _find_mask(path, stem)
        sample_id = "_".join(filter(None, [subject, session, modality]))
        rows.append({
            "dataset": cfg["name"], "sample_id": sample_id, "subject": subject,
            "session": session or "", "modality": modality,
            "path": str(path), "mask_path": mask_path or "", "split": cfg.get("split", "train"),
        })
    return pd.DataFrame(rows)


def _scan_mrrate_atlas(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["root"])
    modalities = cfg.get("modalities") or []
    rows = []
    for archive in sorted(root.rglob("*_atlas.zip")):
        subject = archive.stem.replace("_atlas", "")
        with zipfile.ZipFile(archive) as zf:
            members = [n for n in zf.namelist() if n.endswith((".nii", ".nii.gz")) and not n.endswith("/")]
        mask_members = [m for m in members if "mask" in Path(m).name.lower()]
        vol_members = [m for m in members if m not in mask_members]
        for member in sorted(vol_members):
            stem = _strip_nifti_suffix(Path(member).name) or Path(member).stem
            modality = _match_modality(stem, modalities) if modalities else stem
            mask_member = next((mm for mm in mask_members if Path(mm).stem.split(".")[0].startswith(stem)), None)
            sample_id = f"{subject}_{stem}"
            rows.append({
                "dataset": cfg["name"], "sample_id": sample_id, "subject": subject,
                "session": "", "modality": modality or "",
                "path": f"zip://{archive}::{member}",
                "mask_path": f"zip://{archive}::{mask_member}" if mask_member else "",
                "split": cfg.get("split", "train"),
            })
    return pd.DataFrame(rows)


_SCANNERS = {"bids_like": _scan_bids_like, "mrrate_atlas": _scan_mrrate_atlas}


def build(cfg: dict) -> pd.DataFrame:
    kind = cfg.get("kind", "bids_like")
    scanner = _SCANNERS.get(kind, _scan_bids_like)
    df = scanner(cfg)
    if df.empty:
        raise RuntimeError(
            f"No samples found under root={cfg.get('root')!r} for dataset={cfg.get('name')!r} "
            f"(kind={kind!r}, modalities={cfg.get('modalities')!r})."
        )
    df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    max_samples = cfg.get("max_samples")
    if max_samples:
        df = df.iloc[: int(max_samples)].reset_index(drop=True)
    return df
