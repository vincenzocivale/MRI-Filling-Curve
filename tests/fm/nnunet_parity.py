"""nnU-Net / TotalSegmentator-MRI wrapper vs the original pipelines. Run inside fm-nnunet (GPU):
    PYTHONPATH=src $SFC_FM_ENVS/fm-nnunet/bin/python tests/fm/nnunet_parity.py [--configs ...] [--samples ...]

Per config x sample:
1. TotalSegmentator untouched: `nnUNet_predict_image` (TS/totalsegmentator/nnunet.py:382) with the task's settings;
   a spy around its `nnUNetv2_predict` copies the file it hands to nnU-Net and nnU-Net's exported segmentation.
   -> the wrapper's TotalSegmentator input file must be bitwise equal (data, affine).
2. nnU-Net's own data iterator (`_internal_get_data_iterator_from_lists_of_filenames`, predict_from_raw_data.py:283)
   on that file vs `wrapper.preprocess(sample)["data"]`: bitwise.
3. logits of the hooked run vs a plain `nnUNetPredictor.predict_logits_from_preprocessed_data`: bitwise
   (cuDNN set deterministic for this comparison).
4. stitching path: per-call network outputs recorded and replayed through the wrapper's stitcher (stride 1)
   must reproduce nnU-Net's logits bitwise -> the feature maps are aggregated by exactly nnU-Net's code.
5. labels: `convert_predicted_logits_to_segmentation_with_correct_shape` on the wrapper logits vs the segmentation
   TotalSegmentator's nnU-Net run exported (read back with the plans' reader); TS ran with cuDNN benchmark on,
   so a few boundary voxels may flip -> reported as a fraction, asserted < 1e-4.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

MODELS = Path(os.environ.get("SFC_FM_MODELS", "/leonardo_work/IscrC_SFMRI/fcorrent/models"))
TSMR = MODELS / "TotalSegmentatorMRI"
os.environ["TOTALSEG_WEIGHTS_PATH"] = str(TSMR)  # TS's setup_nnunet points nnUNet_results here
for var in ("nnUNet_results", "nnUNet_raw", "nnUNet_preprocessed"):
    os.environ[var] = str(TSMR)
os.environ.setdefault("TOTALSEG_HOME_DIR", "/leonardo_scratch/large/userexternal/fcorrent/fm_check/.totalseg")

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.nnunet import build

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
# config id -> (TotalSegmentator task_name deciding step size, nnU-Net task id)
TS_TASK = {870: "total_mr", 871: "total_mr", 872: "total_mr", 873: "total_mr", 857: "thigh_shoulder_muscles_mr"}


def ts_reference(cfg: dict, task: int, sample: Path, out: Path) -> None:
    import totalsegmentator.nnunet as tsn

    orig = tsn.nnUNetv2_predict

    def spy(dir_in, dir_out, *a, **k):
        shutil.copy(Path(dir_in) / "s01_0000.nii.gz", out / "ts_input.nii.gz")
        orig(dir_in, dir_out, *a, **k)
        shutil.copy(Path(dir_out) / "s01.nii.gz", out / "ts_seg.nii.gz")

    tsn.nnUNetv2_predict = spy
    folder = Path(cfg["checkpoint"]).name
    trainer, _, configuration = folder.split("__")
    try:
        tsn.nnUNet_predict_image(sample, None, task, model=configuration, folds=[0], trainer=trainer, tta=False,
                                 multilabel_image=True, resample=cfg["args"]["totalseg_resample"],
                                 task_name=TS_TASK[task], quiet=True, device="cuda", skip_saving=True,
                                 nr_threads_resampling=1, nr_threads_saving=1)
    except Exception as e:  # TS's label post-processing for these unregistered part models; spy files are written
        if not (out / "ts_seg.nii.gz").exists():
            raise
        print(f"    (TS post-processing after nnU-Net raised {type(e).__name__}: {e}; nnU-Net outputs captured)")
    finally:
        tsn.nnUNetv2_predict = orig


def check(name: str, cfg: dict, task: int, sample: Path) -> None:
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print(f"== {name} / {sample.name}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ts_reference(cfg, task, sample, tmp)
        w = build(cfg, "cuda")
        p = w.predictor

        (tmp / "mine").mkdir()
        mine, _ = w.totalseg_input(str(sample), tmp / "mine")
        a, b = nib.load(tmp / "ts_input.nii.gz"), nib.load(mine)
        assert a.get_data_dtype() == b.get_data_dtype() and np.array_equal(a.affine, b.affine)
        d = np.abs(a.get_fdata() - b.get_fdata()).max()
        print(f"  [1] TS input file: shape {a.shape} dtype {a.get_data_dtype()} max|diff| {d}")
        assert d == 0

        it = p._internal_get_data_iterator_from_lists_of_filenames([[str(tmp / "ts_input.nii.gz")]], None, None, 1)
        ref_data = next(iter(it))["data"]
        prep = w.preprocess(str(sample))
        d = (ref_data - prep["data"]).abs().max().item()
        print(f"  [2] preprocessed {tuple(prep['data'].shape)} vs nnU-Net iterator max|diff| {d}")
        assert torch.equal(ref_data, prep["data"])

        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        out = w.features(prep)
        plain = nnUNetPredictor(tile_step_size=p.tile_step_size, use_gaussian=True, use_mirroring=p.use_mirroring,
                                perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
        plain.initialize_from_trained_model_folder(cfg["checkpoint"], use_folds=(0,))
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        ref_logits = plain.predict_logits_from_preprocessed_data(prep["data"])
        logits = out["features"]["logits"]
        d = (ref_logits.float() - logits.float()).abs().max().item()
        print(f"  [3] logits {tuple(logits.shape)} {logits.dtype} hooked vs plain max|diff| {d}")
        assert torch.equal(ref_logits, logits)

        calls: list[torch.Tensor] = []
        hook = p.network.register_forward_hook(lambda _m, _i, o: calls.append(o.detach().cpu()))
        try:
            again = p.predict_logits_from_preprocessed_data(prep["data"])
        finally:
            hook.remove()
        replay = w._stitch(calls, [1] * len(p.configuration_manager.patch_size), prep["data"]).cpu()
        d = (replay.float() - again.float()).abs().max().item()
        print(f"  [4] replay stitching ({len(calls)} calls) vs nnU-Net logits max|diff| {d}")
        assert torch.equal(replay, again)

        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, prep["properties"])
        ts_seg, _ = p.plans_manager.image_reader_writer_class().read_seg(str(tmp / "ts_seg.nii.gz"))
        frac = float((ts_seg[0] != seg).mean())
        print(f"  [5] labels vs TotalSegmentator's nnU-Net export: {seg.shape}, differing fraction {frac:.2e}, "
              f"labels {len(np.unique(seg))}")
        assert frac < 1e-4

        f = out["features"]
        print("  features:", {k: (tuple(v.shape), str(v.dtype)) for k, v in f.items()},
              "canonical", out["canonical"], "tile calls", len(out["meta"]["tile_slicers"]))
        assert all(torch.isfinite(v.float()).all() for v in f.values())
        del w, plain
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", type=int, default=[872, 873, 870, 871, 857])
    ap.add_argument("--samples", nargs="+", default=["ixi002_T1.nii.gz", "pdgm0004_T1.nii.gz"])
    args = ap.parse_args()
    for i in args.configs:
        cfg = load_config(ROOT / f"configs/fm/nnunet_totalseg_mr_{i}.yaml")
        for s in args.samples:
            check(f"nnunet_totalseg_mr_{i}", cfg, i, SAMPLES / s)
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
