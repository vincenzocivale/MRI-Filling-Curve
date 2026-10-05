from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import cohort_key

PROBE_SPLITS = ("test", "val", "train")


def _quotas(total: int, pools: pd.Series) -> pd.Series:
    """Largest-remainder allocation of `total` proportional to `pools`, capped by pool size."""
    if total > pools.sum():
        raise ValueError(f"Asked for {total} subjects, only {pools.sum()} available.")
    raw = pools / pools.sum() * total
    q = np.floor(raw).astype(int)
    for c in (raw - q).sort_values(ascending=False).index[:total - q.sum()]:
        q[c] += 1
    return q


def assign_splits(df: pd.DataFrame, seed: int, counts: dict[str, int], label: str) -> pd.Series:
    """Subject-level, cohort-stratified split with exact subject counts, one scan per subject.

    `counts` maps split -> subjects, e.g. {pretrain: 1000, train: 500, val: 100, test: 300}.
    Probe splits (train/val/test) draw only subjects with `label`; pretrain draws from the rest,
    labelled or not. Each chosen subject contributes its first scan (labelled when possible);
    every other row is `unused`. `pretrain: all` instead puts every scan (all sessions) of every
    subject outside the probe splits into pretrain.
    """
    rng = np.random.default_rng(seed)
    scans = (df.assign(_cohort=cohort_key(df), _lab=df[label].notna())
             .sort_values(["subject", "_lab", "session"], ascending=[True, False, True])
             .drop_duplicates("subject"))
    pool = {c: list(rng.permutation(g["subject"].astype(str).to_numpy())) for c, g in scans.groupby("_cohort")}
    labelled = set(scans.loc[scans["_lab"], "subject"].astype(str))
    chosen: dict[str, str] = {}

    def draw(split: str, eligible) -> None:
        avail = {c: [s for s in subs if s not in chosen and eligible(s)] for c, subs in pool.items()}
        for c, k in _quotas(counts[split], pd.Series({c: len(v) for c, v in avail.items()})).items():
            chosen.update((s, split) for s in avail[c][:k])

    for split in PROBE_SPLITS:
        draw(split, labelled.__contains__)
    pretrain_all = counts.get("pretrain") == "all"
    if counts.get("pretrain") and not pretrain_all:
        draw("pretrain", lambda s: True)

    out = pd.Series("unused", index=df.index, dtype=object)
    sub = scans["subject"].astype(str)
    sel = sub.isin(chosen.keys())
    out[scans.index[sel]] = sub[sel].map(chosen).to_numpy()
    if pretrain_all:
        out[~df["subject"].astype(str).isin(chosen.keys())] = "pretrain"
    return out


def summarize(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    rows = []
    for (cohort, split), g in df.groupby([cohort_key(df), "split"], sort=True):
        row = {"cohort": cohort, "split": split, "volumes": len(g), "subjects": g["subject"].nunique()}
        if label_col and label_col in g:
            row["labelled"] = int(g[label_col].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)
