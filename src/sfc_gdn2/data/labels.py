from __future__ import annotations

from pathlib import Path

import pandas as pd


def _participants(root: Path, cohort: str) -> pd.DataFrame:
    p = root / cohort / "participants.tsv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, sep="\t")
    df["_subject"] = f"{cohort}_" + df["participant_id"].str.removeprefix("sub-")
    df["_session"] = df["session_id"].str.removeprefix("ses-")
    return df.drop(columns=["participant_id", "session_id"])


def attach_zip_bids(df: pd.DataFrame, root: str | Path, columns: list[str]) -> pd.DataFrame:
    """Join per-cohort participants.tsv onto manifest rows by (subject, session)."""
    root = Path(root)
    keys = df["subject"].astype(str), df["session"].astype(str).str.zfill(2)
    out = df.assign(_subject=keys[0], _session=keys[1])
    tables = [t for c in sorted(df["cohort"].unique()) if not (t := _participants(root, c)).empty]
    if not tables:
        raise RuntimeError(f"No participants.tsv found under {root}")
    meta = pd.concat(tables, ignore_index=True)
    available = [c for c in columns if c in meta.columns]
    if missing := set(columns) - set(available):
        raise ValueError(f"Columns absent from every participants.tsv: {sorted(missing)}")
    out = out.merge(meta[["_subject", "_session", *available]], on=["_subject", "_session"], how="left")
    return out.drop(columns=["_subject", "_session"])


def attach_totalseg(df: pd.DataFrame, root: str | Path, columns: list[str]) -> pd.DataFrame:
    from .totalseg import attach as run
    return run(df, root, columns)


ATTACHERS = {"zip_bids": attach_zip_bids, "totalseg": attach_totalseg}


def attach(df: pd.DataFrame, kind: str, root: str | Path, columns: list[str]) -> pd.DataFrame:
    if kind not in ATTACHERS:
        raise ValueError(f"No label attacher for kind={kind!r}; have {sorted(ATTACHERS)}")
    return ATTACHERS[kind](df, root, columns)
