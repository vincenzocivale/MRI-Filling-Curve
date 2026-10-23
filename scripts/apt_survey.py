"""Entropy of the octree cells on real volumes: which thresholds give which token counts (no model, no GPU needed).

    python scripts/apt_survey.py [--per-dataset 3] [--base-mm 16] [--levels 3] [--bins 16]

Volumes: the benchmark's tuning subjects (no test split), a few per dataset of the probe tasks."""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from sfc_gdn2.apt import base_dims, partition, scores
from sfc_gdn2.bench import probe as bp
from sfc_gdn2.data.volume import decode, open_image
from sfc_gdn2.io import load_yaml

OUT = "/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench"
DATASETS = ("IXI", "OASIS-1", "AOMIC-ID1000", "UCSF-PDGM", "UPENN-GBM", "SOOP")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=3)
    ap.add_argument("--base-mm", type=float, default=16)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vols = pd.read_csv(OUT + "/volumes.csv", low_memory=False)
    vols = vols[vols["subject"].isin(bp.tune_subjects(pd.read_csv(OUT + "/splits.csv")))]
    rows = []
    for ds in DATASETS:
        for _, r in vols[vols["dataset"] == ds].sort_values("id").head(a.per_dataset).iterrows():
            arr, sp, (s, c) = decode(open_image(r["input"]))
            vol = torch.from_numpy(arr).to(dev).float() * s + c
            k0 = base_dims(sp, a.base_mm, a.levels)
            vol = torch.nn.functional.pad(vol, [p for n, q in zip(reversed(vol.shape), reversed(k0)) for p in (0, -n % q)])
            g0 = [n // q for n, q in zip(vol.shape, k0)]
            sc = scores(vol, k0, a.levels, a.bins)
            fine = int(torch.tensor(g0).prod()) * 8 ** (a.levels - 1)
            row = {"id": r["id"], "spacing": tuple(round(float(x), 2) for x in sp), "k0": k0, "finest": fine,
                   "base": int(torch.tensor(g0).prod())}
            fg = vol.reshape(g0[0], k0[0], g0[1], k0[1], g0[2], k0[2]).mean((1, 3, 5)) > 0.05
            row["H0 q50/q90 (fg cells)"] = " / ".join(f"{sc[0][fg].quantile(q).item():.2f}" for q in (.5, .9))
            row["H0 q50 (bg cells)"] = f"{sc[0][~fg].median().item():.2f}" if (~fg).any() else "-"
            for t in a.taus:
                leaves, _ = partition(sc, g0, a.levels, {"tau": [t] * (a.levels - 1)})
                row[f"tau {t}"] = round(sum(len(x) for x in leaves) / fine, 3)
            rows.append(row | {"dataset": ds})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    cols = [c for c in df if c.startswith("tau")]
    print("\nleaves / finest-grid cells (levels", a.levels, ", base", a.base_mm, "mm), mean per dataset")
    print(df.groupby("dataset")[cols].mean().round(3).to_string())
