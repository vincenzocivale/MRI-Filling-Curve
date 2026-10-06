"""BrainIAC wrapper vs the original pipeline. Run inside fm-brainiac:
    PYTHONPATH=src $SFC_FM_ENVS/fm-brainiac/bin/python tests/fm/brainiac_parity.py [--skip-e2e]

1. shipped reference: wrapper (preprocessed=true) on the repo's processed sample vs the repo's own
   `inference/features/features.csv` (computed by the authors on GPU -> tolerance 1e-3).
2. the repo's `get_brainiac_features.py` on the repo's processed samples + our SRI24 sample, model inputs
   compared bitwise to its `BrainAgeDataset` items, and features:
   a. its `infer()` called in this process on the wrapper's model, b. the script run untouched as a
   subprocess (same device). Inputs must be bitwise equal; embeddings within 1e-5 (|x| ~ 5): CPU float32
   GEMM kernels are not bitwise stable across call sites (observed 0 and 1.1e-6).
3. end to end: the repo's `preprocessing/mri_preprocess_3d_simple.py` run untouched and the wrapper
   (preprocessed=false) on the repo's unprocessed sample, both vs the shipped processed image. Registration
   samples voxels with a wall-clock seed (mri_preprocess_3d_simple.py:113), so only tolerances apply.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sfc_gdn2.fm.brainiac import build

REPO = Path(os.environ.get("SFC_FM_REPOS", "/leonardo_work/IscrC_SFMRI/fcorrent/repos")) / "BrainIAC"
CKPT = Path(os.environ.get("SFC_FM_MODELS", "/leonardo_work/IscrC_SFMRI/fcorrent/models")) / "BrainIAC/BrainIAC.ckpt"
SRC = REPO / "src"
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the repo script picks cuda:0 when available
FEATS = [f"Feature_{i}" for i in range(768)]


def wrapper(**args):
    return build({"model": "brainiac", "repo": str(REPO), "checkpoint": str(CKPT), "args": args}, DEVICE)


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def shipped_reference(w) -> None:
    ref = pd.read_csv(SRC / "inference/features/features.csv")[FEATS].values[0]
    emb = w(str(SRC / "data/sample/processed/I10307487_0000.nii.gz"))["features"]["embedding"].cpu().numpy()
    d = np.abs(emb - ref).max()
    print(f"[1] shipped features.csv: max|diff|={d:.3e} cos={cos(emb, ref):.10f}")
    assert d < 1e-3, d


def bitwise_vs_script(w, tmp: Path) -> None:
    sys.path.insert(0, str(SRC))
    from dataset import BrainAgeDataset, get_validation_transform

    root = tmp / "root"
    root.mkdir()
    imgs = {"I10307487_0000": SRC / "data/sample/processed/I10307487_0000.nii.gz",
            "00001_t1n": SRC / "data/sample/processed/00001_t1n.nii.gz",
            "pdgm0004_T1": SAMPLES / "pdgm0004_T1.nii.gz"}
    for k, p in imgs.items():
        (root / f"{k}.nii.gz").symlink_to(p)
    csv, out_csv = tmp / "in.csv", tmp / "out.csv"
    pd.DataFrame({"pat_id": list(imgs), "label": 0.0}).to_csv(csv, index=False)
    subprocess.run([sys.executable, "get_brainiac_features.py", "--checkpoint", str(CKPT), "--input_csv", str(csv),
                    "--output_csv", str(out_csv), "--root_dir", str(root)], cwd=SRC, check=True)
    cli = pd.read_csv(out_csv)[FEATS].values.astype(np.float32)  # float32 written with round-trip repr
    ds = BrainAgeDataset(str(csv), str(root), transform=get_validation_transform())
    import get_brainiac_features as g  # module import only sets CUDA_VISIBLE_DEVICES=0 and the seed

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)  # as its main()
    inproc = g.infer(w.model, loader)[FEATS].values
    for i, k in enumerate(imgs):
        prep = w.preprocess(str(root / f"{k}.nii.gz"))
        x_equal = torch.equal(prep["image"].as_tensor(), ds[i]["image"].as_tensor())
        emb = w.features(prep)["features"]["embedding"].cpu().numpy()
        d = np.abs(emb - cli[i]).max()
        di = np.abs(emb - inproc[i]).max()
        print(f"[2] {k}: input bitwise={x_equal} | infer() max|diff|={di:.3e} | script subprocess max|diff|={d:.3e}")
        assert x_equal and di <= 1e-5 and d <= 1e-5, k  # float32 kernel noise only, see docstring


def end_to_end(tmp: Path) -> None:
    raw = SRC / "data/sample/unprocessed/I10307487.nii.gz"
    shipped = SRC / "data/sample/processed/I10307487_0000.nii.gz"
    ref_emb = pd.read_csv(SRC / "inference/features/features.csv")[FEATS].values[0]
    (tmp / "raw").mkdir()
    (tmp / "raw" / raw.name).symlink_to(raw)
    t0 = time.time()
    subprocess.run([sys.executable, "preprocessing/mri_preprocess_3d_simple.py", "--temp_img",
                    "preprocessing/atlases/temp_head.nii.gz", "--input_dir", str(tmp / "raw"), "--output_dir",
                    str(tmp / "script")], cwd=SRC, check=True)
    print(f"[3] repo script preprocessing: {time.time() - t0:.0f}s")
    w = wrapper(preprocessed=False, keep_dir=str(tmp / "wrapper"))
    t0 = time.time()
    res = w(str(raw))
    print(f"[3] wrapper (preprocess+features): {time.time() - t0:.0f}s")
    script_emb = wrapper(preprocessed=True)(str(tmp / "script/I10307487_0000.nii.gz"))["features"]["embedding"]
    s = nib.load(str(shipped))
    sv = np.asarray(s.dataobj, np.float32)
    for name, p in [("script", tmp / "script/I10307487_0000.nii.gz"), ("wrapper", tmp / "wrapper/I10307487_0000.nii.gz")]:
        im = nib.load(str(p))
        v = np.asarray(im.dataobj, np.float32)
        assert v.shape == sv.shape and np.allclose(im.affine, s.affine, atol=1e-4), (name, v.shape, im.affine)
        m, ms = v != 0, sv != 0
        dice = 2 * (m & ms).sum() / (m.sum() + ms.sum())
        u = m | ms
        r = np.corrcoef(v[u], sv[u])[0, 1]
        print(f"[3] {name} vs shipped processed: grid equal, brain-mask Dice={dice:.4f}, intensity r={r:.4f}")
        # voxelwise r is reported only: rigid registration with random MI sampling moves edges by sub-mm
        # amounts (the repo's own script gives r ~0.6 vs its shipped output); the brain mask must agree.
        assert dice > 0.95, (name, dice)
    for name, e in [("script", script_emb), ("wrapper", res["features"]["embedding"])]:
        c = cos(e.cpu().numpy(), ref_emb)
        print(f"[3] {name} embedding vs shipped features.csv: cos={c:.5f}")
    print(f"[3] wrapper vs script embedding cos={cos(res['features']['embedding'].cpu(), script_emb.cpu()):.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-e2e", action="store_true")
    a = ap.parse_args()
    print(f"device={DEVICE} torch={torch.__version__}")
    w = wrapper(preprocessed=True)
    with tempfile.TemporaryDirectory() as t:
        shipped_reference(w)
        bitwise_vs_script(w, Path(t))
        if not a.skip_e2e:
            end_to_end(Path(t))
    print("BRAINIAC PARITY OK")


if __name__ == "__main__":
    main()
