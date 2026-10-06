"""BrainFM wrapper == the original pipeline (run inside fm-brainfm, PYTHONPATH=src).

Original path, untouched: `scripts/demo_test.py:46,51` = `utils.test_utils.prepare_image(path, win_size=None,
zero_crop_first=True, spacing=None, add_bf=False)` then `evaluate_image(im, feature_only=False, ...)`. The
only runtime change is the one `evaluate_image` cannot run without: its module-level default config paths
(`test_utils.py:28-31`, pointing at a non-existent `cfg/defaults/`) are set to the repo's real `cfgs/` files,
and the demo configs are passed as absolute paths (`utils/process_cfg.py:60`).
Asserts bitwise equality of the preprocessed input, every feature level and every trained task output
(cuDNN deterministic for both paths), and that the wrapper's corrected affine maps the crop correctly.

usage: python tests/fm/brainfm_parity.py [--device cuda|cpu] [images...]
"""
from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.brainfm import GEN_CFGS, TRAIN_CFGS, UNTRAINED_OUTPUTS, build

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
DEFAULT = [SAMPLES / "pdgm0004_T1.nii.gz", SAMPLES / "pdgm0004_FLAIR.nii.gz", SAMPLES / "ixi002_T1.nii.gz"]


def diff(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return (a.float() - b.float()).abs().max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", default=[str(p) for p in DEFAULT])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    cfg = load_config(ROOT / "configs/fm/brainfm.yaml") if "SFC_FM_ENVS" in os.environ else None
    cfg = cfg or {"model": "brainfm", "repo": "/leonardo_work/IscrC_SFMRI/fcorrent/repos/BrainFM",
                  "checkpoint": "/leonardo_work/IscrC_SFMRI/fcorrent/models/BrainFM/assets/brainfm_pretrained.pth"}
    cfg = cfg | {"args": {"levels": True, "task_outputs": True}}
    repo = Path(cfg["repo"])
    wrapper = build(cfg, args.device)   # puts the repo on sys.path, imports test_utils from the repo root
    tu = wrapper.tu
    tu.default_gen_cfg_file = str(repo / GEN_CFGS[0])
    tu.default_train_cfg_file, tu.default_val_file = str(repo / TRAIN_CFGS[0]), str(repo / TRAIN_CFGS[1])
    gen_cfg, model_cfg = str(repo / GEN_CFGS[1]), str(repo / TRAIN_CFGS[2])

    worst = 0.0
    for path in args.images:
        print(f"== {path}", flush=True)
        t0 = time.time()
        with contextlib.chdir(repo):
            im, *_ = tu.prepare_image(path, win_size=None, zero_crop_first=True, spacing=None, im_only=False,
                                      add_bf=False, device=args.device)
            ref = tu.evaluate_image(im, ckp_path=cfg["checkpoint"], feature_only=False, device=args.device,
                                    gen_cfg=gen_cfg, model_cfg=model_cfg)
        ref = {k: ([f.cpu() for f in v] if k == "feat" else v.cpu()) for k, v in ref.items()}
        im = im.cpu()
        t1 = time.time()
        prepared = wrapper.preprocess(path)
        res = wrapper.features(prepared)
        t2 = time.time()
        f, meta = res["features"], res["meta"]
        print(f"   original {t1 - t0:.1f}s (incl. model build + ckpt load), wrapper {t2 - t1:.1f}s; "
              f"input {tuple(im.shape)} bbox {meta['bbox']} of {meta['aligned_shape']}")

        d = {"input": diff(prepared["input"].cpu(), im)}
        d |= {f"feat_{i}": diff(f["feat_last" if i == 5 else f"feat_{i}"].cpu(), r[0]) for i, r in enumerate(ref["feat"])}
        tasks = [k for k in ref if k != "feat" and k not in UNTRAINED_OUTPUTS]
        d |= {f"task_{k}": diff(f[f"task_{k}"].cpu(), ref[k][0]) for k in tasks}
        assert not {f"task_{k}" for k in UNTRAINED_OUTPUTS} & set(f), "untrained SR outputs leaked"
        for k, v in d.items():
            print(f"   max|diff| {k:24s} {v:.3e}")
        worst = max(worst, *d.values())

        # corrected affine: voxel (0,0,0) of the cropped input == voxel bbox[0] of the aligned grid
        lo = np.asarray(meta["bbox"][0], dtype=np.float64)
        a, c = meta["aligned_affine"].numpy(), meta["affine"].numpy()
        assert np.allclose(c[:3, 3], a[:3, :3] @ lo + a[:3, 3]) and np.allclose(c[:3, :3], a[:3, :3])
        src = nib.load(path)
        if np.allclose(src.header.get_zooms()[:3], 1.0):  # no resampling: orientation must match nibabel's RAS
            assert np.allclose(a, nib.as_closest_canonical(src).affine), "aligned affine != nibabel canonical"
        fl = f["feat_last"]
        assert torch.isfinite(fl).all() and torch.allclose(fl.norm(dim=0), torch.ones(()), atol=1e-4)
        del ref, res, prepared, im
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"WORST max|diff| = {worst:.3e}")
    assert worst == 0.0, "wrapper differs from the original pipeline"
    print("PARITY OK (bitwise)")


if __name__ == "__main__":
    main()
