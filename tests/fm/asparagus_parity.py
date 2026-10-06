"""Parity of sfc_gdn2.fm.asparagus with the original FOMO26 pipeline (run INSIDE fm-asparagus, PYTHONPATH=src):

    source configs/fm/leonardo.env
    $SFC_FM_ENVS/fm-asparagus/bin/python tests/fm/asparagus_parity.py [samples...]

Reference = fomo-lp `pipeline/embed_all.py:embed_all` untouched (CSV -> <ptid>.npz) plus its
load_scan/preprocess for the model input. Cases, for both checkpoints:
- reorient_ras=false on the raw files: the pure repo path;
- reorient_ras=true: the reference gets RAS files written independently with nibabel
  `as_closest_canonical` (the organisers' delivery), the wrapper the raw files.
- encoder=pretrain vs asparagus `SelfSupervisedModule.predict_step` (self_supervised.py:183-186).
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

from sfc_gdn2.fm.api import expand
from sfc_gdn2.fm.asparagus import build

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
CONFIGS = ["fomo26", "amaes_fomo260k"]


def cfg_for(name: str, **args) -> dict:
    import yaml

    cfg = expand(yaml.safe_load((ROOT / f"configs/fm/{name}.yaml").read_text()))
    cfg["args"] = {**cfg["args"], **args}
    return cfg


def check(tag: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    assert a.shape == b.shape, (tag, a.shape, b.shape)
    d = (a - b).abs().max().item()
    print(f"  {tag:<48} shape={tuple(a.shape)} max|diff|={d:.3e}", flush=True)
    assert torch.equal(a, b), f"{tag}: not bitwise equal"


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
        for name in CONFIGS:
            for reorient, ref_files in ((False, samples), (True, ras)):
                w = build(cfg_for(name, reorient_ras=reorient), "cpu")
                ea = w.ea
                work = tmp / f"{name}_{reorient}"
                work.mkdir()
                ref = reference(ea, w.cfg["checkpoint"], ref_files, work)
                print(f"[{name}] reorient_ras={reorient}", flush=True)
                for p, rp in zip(samples, ref_files):
                    prep = w.preprocess(str(p))
                    check(f"{p.name} input", prep["x"], ea.preprocess(ea.load_scan(rp)))
                    out = w.features(prep)
                    check(f"{p.name} embedding vs embed_all", out["features"]["embedding"], ref[rp.name.split(".")[0]])
                    gap = out["features"]["stage5"].mean((1, 2, 3))
                    print(f"  stage5 GAP vs embedding max|diff|={(gap - out['features']['embedding']).abs().max():.1e}"
                          f"  meta reoriented={prep['meta']['reoriented_to_ras']} offset={prep['meta']['input_offset']}")
                del w
            # opt-in pretraining encoder vs the repo's SelfSupervisedModule.predict_step
            from asparagus.modules.lightning_modules.self_supervised import SelfSupervisedModule
            from asparagus.modules.networks.resenc_unet import resenc_unet_b

            w = build(cfg_for(name, encoder="pretrain"), "cpu")
            sd = torch.load(w.cfg["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
            net = resenc_unet_b(dimensions="3D", input_channels=1, output_channels=1, use_skip_connections=False)
            ssl = SelfSupervisedModule(model=net, weights=sd).eval()
            print(f"[{name}] encoder=pretrain", flush=True)
            for p in samples:
                prep = w.preprocess(str(p))
                with torch.no_grad():
                    ref_deep = ssl.predict_step({"image": prep["x"].unsqueeze(0)}, 0)[0]
                check(f"{p.name} deepest vs predict_step", w.features(prep)["features"]["deepest"], ref_deep)
            del w, ssl
    five = cfg_for("fomo26")
    five["checkpoint"] = five["checkpoint"].replace("resenc_unet_b.ckpt", "resenc_unet_b_5ch.ckpt")
    try:
        build(five, "cpu")
        raise AssertionError("5ch checkpoint was accepted")
    except ValueError as e:
        print(f"5ch rejected: {e}")
    print("PARITY OK")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]] or [SAMPLES / "ixi002_T1.nii.gz", SAMPLES / "pdgm0004_T1.nii.gz"])
