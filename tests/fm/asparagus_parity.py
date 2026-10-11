"""Parity of sfc_gdn2.fm.asparagus with the original FOMO26 pipeline (run INSIDE fm-asparagus, PYTHONPATH=src):

    source configs/fm/leonardo.env
    $SFC_FM_ENVS/fm-asparagus/bin/python tests/fm/asparagus_parity.py [samples...]

Reference = fomo-lp `pipeline/embed_all.py:embed_all` untouched (CSV -> <ptid>.npz) plus its
load_scan/preprocess for the model input. Cases (fomo26):
- reorient_ras=false on the raw files: the pure repo path;
- reorient_ras=true: the reference gets RAS files written independently with nibabel
  `as_closest_canonical` (the organisers' delivery), the wrapper the raw files.
- the 5ch checkpoint must be rejected.
Everything is fp32 on CPU in one process: equality is asserted bitwise.
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from common import SAMPLES, check

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.asparagus import build

ROOT = Path(__file__).resolve().parents[2]


def cfg_for(**args) -> dict:
    cfg = load_config(ROOT / "configs/fm/fomo26.yaml")
    cfg["args"] = {**cfg["args"], **args}
    return cfg


def reference(ea, ckpt: str, files: list[Path], work: Path) -> dict[str, np.ndarray]:
    """The original embed_all.embed_all on a CSV of `files` (one <ptid>.npz each)."""
    out = work / "emb"
    with open(work / "subjects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ptid", "nifti_path"])
        w.writerows([[p.name.split(".")[0], str(p)] for p in files])
    ea.embed_all(work / "subjects.csv", Path(ckpt), out)
    return {p.name.split(".")[0]: np.load(out / f"{p.name.split('.')[0]}.npz")["arr_0"] for p in files}


def main(samples: list[Path]) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ras = []
        for p in samples:  # independent RAS delivery of each sample
            img = nib.as_closest_canonical(nib.load(p))
            dst = tmp / p.name.replace(".nii.gz", ".nii")
            nib.save(nib.Nifti1Image(np.asarray(img.get_fdata(), dtype=np.float32), img.affine), dst)
            ras.append(dst)
            print(f"{p.name}: {nib.load(p).shape} {''.join(nib.aff2axcodes(nib.load(p).affine))}", flush=True)
        for reorient, ref_files in ((False, samples), (True, ras)):
            w = build(cfg_for(reorient_ras=reorient), "cpu")
            ea = w.ea
            work = tmp / f"fomo26_{reorient}"
            work.mkdir()
            ref = reference(ea, w.cfg["checkpoint"], ref_files, work)
            print(f"[fomo26] reorient_ras={reorient}", flush=True)
            for p, rp in zip(samples, ref_files):
                prep = w.preprocess(str(p))
                check(f"{p.name} input", prep["x"], ea.preprocess(ea.load_scan(rp)))
                out = w.features(prep)
                check(f"{p.name} embedding vs embed_all", out["features"]["embedding"], ref[rp.name.split(".")[0]])
                gap = out["features"]["stage5"].mean((1, 2, 3))
                print(f"  stage5 GAP vs embedding max|diff|={(gap - out['features']['embedding']).abs().max():.1e}"
                      f"  meta reoriented={prep['meta']['reoriented_to_ras']} offset={prep['meta']['input_offset']}")
            del w
    five = cfg_for()
    five["checkpoint"] = five["checkpoint"].replace("resenc_unet_b.ckpt", "resenc_unet_b_5ch.ckpt")
    try:
        build(five, "cpu")
        raise AssertionError("5ch checkpoint was accepted")
    except ValueError as e:
        print(f"5ch rejected: {e}")
    print("PARITY OK")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]] or [SAMPLES / "ixi002_T1.nii.gz", SAMPLES / "pdgm0004_T1.nii.gz"])
