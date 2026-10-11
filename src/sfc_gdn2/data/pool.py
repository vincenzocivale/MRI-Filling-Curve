"""Pretraining pool from the catalog (datasets-meta/DECISIONS.md, Version 1): every scan of the pretraining datasets
(MR-RATE phase-1 brain series, FOMO300K after dedup, SALD, CoRR, Petfrog), none of a downstream benchmark subject.

`sfc pretrain-pool <pretrain config>` writes `data.manifests[0]` (split `pretrain`) with each scan's header shape,
spacing, `frames` (4D+ scans: DWI directions, ASL, ...; the loader reads a random one per access) and `n_patch` at the
config's patch_mm (a statistic: the loader recomputes it from shape / spacing at its own patch_mm). Unreadable scans
and scans with no positive spacing are dropped. The header pass reads ~1M headers: a SLURM CPU job.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import itertools
import math
import multiprocessing as mp
import zipfile

import nibabel as nib
import pandas as pd

from .volume import patch_dims

COLUMNS = ["dataset", "cohort", "subject", "session", "modality", "variant", "path", "use_phase1"]


def select(catalog: pd.DataFrame, datasets: list[str], exclude_subjects) -> pd.DataFrame:
    """Rows of `datasets`; MR-RATE: phase-1 series only (brain). Raises if a subject of `exclude_subjects`
    (the downstream benchmark's) is in the pool."""
    df = catalog[catalog["dataset"].isin(datasets)]
    if "use_phase1" in df:
        df = df[df["use_phase1"].astype(str) != "False"]
    if leaked := set(df["subject"]) & set(exclude_subjects):
        raise RuntimeError(f"{len(leaked)} downstream subjects in the pretraining pool, e.g. {sorted(leaked)[:3]}")
    df = df.drop(columns=["use_phase1"], errors="ignore").drop_duplicates("path")
    return df.assign(split="pretrain",
                     sample_id=[hashlib.sha1(p.encode()).hexdigest()[:16] for p in df["path"]]).reset_index(drop=True)


def _header(f, name: str) -> tuple[tuple, tuple]:
    raw = io.BytesIO((gzip.GzipFile(fileobj=f) if name.endswith(".gz") else f).read(540))
    try:
        h = nib.Nifti1Header.from_fileobj(raw)
    except nib.spatialimages.HeaderDataError:
        raw.seek(0)
        h = nib.Nifti2Header.from_fileobj(raw)
    return tuple(int(n) for n in h.get_data_shape()), tuple(float(z) for z in h.get_zooms())


def _headers(paths: list[str]) -> list:
    """[(shape, zooms) | None] of files / members of ONE archive (one open per archive)."""
    out = []
    zipped = paths[0].startswith("zip://")
    with zipfile.ZipFile(paths[0][len("zip://"):].split("::", 1)[0]) if zipped else contextlib.nullcontext() as zf:
        for p in paths:
            try:
                if zf is None:
                    with open(p, "rb") as f:
                        out.append(_header(f, p))
                else:
                    member = p.split("::", 1)[1]
                    with zf.open(member) as f:
                        out.append(_header(f, member))
            except Exception:  # noqa: BLE001 -- unreadable scan: dropped from the pool
                out.append(None)
    return out


def n_patch(shape, zooms, patch_mm: float) -> int:
    """Patches of the first 3 axes (one frame); 0 = unusable (fewer than 3 axes, no positive spacing)."""
    if len(shape) < 3 or len(zooms) < 3 or min(zooms[:3]) <= 0:
        return 0
    return math.prod(-(-n // k) for n, k in zip(shape[:3], patch_dims(zooms, patch_mm)))


def with_patches(df: pd.DataFrame, patch_mm: float, workers: int = 8) -> pd.DataFrame:
    """Header shape / spacing / frames / n_patch per row; unusable rows are dropped."""
    archive = df["path"].str.split("::").str[0]
    groups = [g.tolist() for _, g in df["path"].groupby(archive, sort=False)]
    with mp.get_context("spawn").Pool(workers) if workers > 1 else contextlib.nullcontext() as pool:
        hs = map(_headers, groups) if pool is None else pool.imap(_headers, groups, chunksize=64)
        heads = dict(zip(itertools.chain.from_iterable(groups), itertools.chain.from_iterable(hs)))
    h = [heads[p] for p in df["path"]]
    df = df.assign(shape=["x".join(map(str, x[0])) if x else "" for x in h],
                   spacing=["x".join(f"{z:.4g}" for z in x[1][:3]) if x else "" for x in h],
                   frames=[math.prod(x[0][3:]) if x else 0 for x in h],
                   n_patch=[n_patch(*x, patch_mm) if x else 0 for x in h])
    return df[(df["n_patch"] > 0) & (df["frames"] > 0)].reset_index(drop=True)
