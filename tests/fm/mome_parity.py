"""MoME / MoME+ wrapper vs the original pipeline, run INSIDE fm-mome (one fork per process):

    PYTHONPATH=src <fm-mome>/bin/python tests/fm/mome_parity.py prepro   --model mome|mome_plus --out DIR  # CPU
    PYTHONPATH=src <fm-mome>/bin/python tests/fm/mome_parity.py features --model mome|mome_plus --out DIR  # GPU

prepro: the ORIGINAL C/one-step.py (verbatim copy of Codes_prepro run from a temp dir: the script chdirs into
its own folder and writes there, so it cannot run inside the clone; it also waits on input(), fed "\n") on a
nnU-Net-style folder, then the fork's run_case on its output, vs wrapper.preprocess (prepro=true). Asserts the
preprocessed arrays and the ANTs affine are bitwise equal; stores the original outputs in DIR/prepro for stage 2.
features: on those outputs (prepro=false), wrapper logits vs (a) a fresh untouched nnUNetPredictor
(initialize_from_trained_model_folder + run_case + predict_logits_from_preprocessed_data) and (b) the fork's
nnUNetv2_predict entry point (predict_entry_point, --save_probabilities) vs the fork's own export of the
wrapper logits; plus strict loads (the fork's load_state_dict) and the DNA assertions done at build.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.mome import FOLD, FORKS, PLUS_MODALITIES, build

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
PDGM = {"t1": "T1", "t1ce": "T1c", "t2": "T2", "flair": "FLAIR"}
# (case, {modality: sample}, skull) ; mome takes one image, mome_plus subsets of the pdgm modalities
CASES = {
    "mome": [("ixi002_T1", {"image": "ixi002_T1"}, True), ("pdgm0004_T1", {"image": "pdgm0004_T1"}, False)],
    "mome_plus": [("pdgm0004", {m: f"pdgm0004_{s}" for m, s in PDGM.items()}, False)],
}
SUBSETS = [("t1",), ("t1", "t2", "flair"), PLUS_MODALITIES]  # MultiMod scenarios checked for mome_plus


def maxdiff(a, b) -> float:
    a, b = torch.as_tensor(np.asarray(a, dtype=np.float64)), torch.as_tensor(np.asarray(b, dtype=np.float64))
    same_nonfinite = bool(((~torch.isfinite(a)) == (~torch.isfinite(b))).all() and (a.isinf() == b.isinf()).all())
    d = (a - b)[torch.isfinite(a) & torch.isfinite(b)].abs()
    return float(d.max()) if same_nonfinite else float("inf")  # finite part; inf if the non-finite masks differ


def wrapper(model: str, device: str, **args):
    cfg = load_config(ROOT / "configs" / "fm" / f"{model}.yaml")
    cfg["args"] = {**cfg["args"], **args}
    return build(cfg, device)


def plus_layout(files: dict[str, Path], root: Path) -> tuple[list[str], list[int]]:
    """imagesTs/BraTS<mod>/<case>_<mod>_0000.nii.gz (README, MoME+ section 3) -> run_case input + MultiMod."""
    for m, f in files.items():
        dst = root / f"BraTS{m}" / f"case_{m}_0000.nii.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(Path(f).resolve())
    first = next(m for m in PLUS_MODALITIES if m in files)
    return [str(root / f"BraTS{first}" / f"case_{first}_0000.nii.gz")], [int(m in files) for m in PLUS_MODALITIES]


def run_case(w, files: dict[str, Path], tmp: Path):
    p = w.predictor
    pre = p.configuration_manager.preprocessor_class(verbose=False)
    if w.plus:
        image_files, mm = plus_layout(files, tmp)
        return pre.run_case(image_files, None, p.plans_manager, p.configuration_manager, p.dataset_json,
                            MultiMod=mm)
    return pre.run_case([str(files["image"])], None, p.plans_manager, p.configuration_manager, p.dataset_json)


def stage_prepro(model: str, out: Path) -> None:
    import SimpleITK as sitk
    w = wrapper(model, "cuda" if model == "mome_plus" else "cpu")  # the CPU stage never runs the network
    env = os.environ | {"PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
                        "ANTS_RANDOM_SEED": str(w.args["ants_seed"]), "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1"}
    for case, mods, skull in CASES[model]:
        w.args["skull"] = skull
        with tempfile.TemporaryDirectory(prefix="momeparity-") as t:
            tmp = Path(t)
            codes = tmp / "codes"
            codes.mkdir()
            for f in w.codes.glob("*.py"):
                shutil.copy(f, codes)
            (codes / "Template").symlink_to(w.codes / "Template")
            ds = tmp / "ds" / "imagesTs"
            ds.mkdir(parents=True)
            for m, s in mods.items():
                shutil.copy(SAMPLES / f"{s}.nii.gz", ds / f"{m}_0000.nii.gz")
            subprocess.run([sys.executable, "one-step.py", "-dataset_path", str(ds), *(["-ss"] if skull else [])],
                           cwd=codes, env=env, input="\n", text=True, check=True)
            res = ds.parent / ("imagesTs_RegMNI152+skull_stripping+crop" if skull else "imagesTs_RegMNI152+crop")
            files = {m: res / f"{m}_to_MNI152_0000.nii.gz" for m in mods}
            ref, _, _ = run_case(w, files, tmp / "nn")
            dst = out / "prepro" / case
            dst.mkdir(parents=True, exist_ok=True)
            for m, f in files.items():
                shutil.copy(f, dst / f"{m}.nii.gz")
            image = ({m: str(SAMPLES / f"{s}.nii.gz") for m, s in mods.items()} if w.plus
                     else str(SAMPLES / f"{mods['image']}.nii.gz"))
            prepared = w.preprocess(image)
            print(f"[parity] {model} {case} preprocessed: shape {prepared['data'].shape}, "
                  f"max|diff| {maxdiff(prepared['data'], ref)}", flush=True)
            assert prepared["data"].dtype == ref.dtype and np.array_equal(prepared["data"], ref)
            for m in mods:
                tx = sitk.ReadTransform(str(ds.parent / "Affine2MNI152Matrix" / f"{m}_to_MNI152"
                                            / "warp_0GenericAffine.mat"))
                ours = prepared["prepro"][m]["ants_transform"]["parameters"]
                print(f"[parity] {model} {case} {m} ANTs affine max|diff| {maxdiff(tx.GetParameters(), ours)}")
                assert list(tx.GetParameters()) == ours


def cli_probabilities(w, files: dict[str, Path], tmp: Path) -> np.ndarray:
    """The fork's nnUNetv2_predict (predict_entry_point) on a nnU-Net results folder, --save_probabilities."""
    ckpt = Path(w.cfg["checkpoint"])
    results = tmp / "results"
    folder = results / "Dataset900_MoME" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    folder.mkdir(parents=True)
    (folder / "plans.json").symlink_to(ckpt.parent / "nnUNetPlans.json")
    (folder / "dataset.json").symlink_to(ckpt.parent / "dataset.json")
    (folder / f"fold_{FOLD}").symlink_to(ckpt.parent)
    if w.plus:
        image_files, mm = plus_layout(files, tmp / "in")
        inp, extra = str(Path(image_files[0]).parent), ["--MultiMod", *map(str, mm)]
    else:
        (tmp / "in").mkdir()
        (tmp / "in" / "case_0000.nii.gz").symlink_to(Path(files["image"]).resolve())
        inp, extra = str(tmp / "in"), []
    fork = os.pathsep.join(str(w.repo / d) for d in FORKS[w.key])  # the CLI process imports the same fork
    env = os.environ | {"nnUNet_results": str(results), "nnUNet_raw": str(tmp), "nnUNet_preprocessed": str(tmp),
                        "PYTHONPATH": f"{fork}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    code = "import sys; from nnunetv2.inference.predict_from_raw_data import predict_entry_point as m; sys.exit(m())"
    subprocess.run([sys.executable, "-c", code, "-i", inp, "-o", str(tmp / "out"), "-d", "Dataset900_MoME",
                    "-c", "3d_fullres", "-f", FOLD, "-chk", ckpt.name, "--save_probabilities", "-npp", "1",
                    "-nps", "1", "-device", "cuda", *extra], env=env, check=True)
    (npz,) = (tmp / "out").glob("*.npz")
    return np.load(npz)["probabilities"]


def stage_features(model: str, out: Path) -> None:
    w = wrapper(model, "cuda", prepro=False)  # puts the fork on sys.path
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    jobs = []
    for case, mods, _ in CASES[model]:
        files = {m: out / "prepro" / case / f"{m}.nii.gz" for m in mods}
        if w.plus:
            jobs += [(f"{case}_{'+'.join(s)}", {m: files[m] for m in s}) for s in SUBSETS]
        else:
            jobs.append((case, files))
    tmp_root = Path(tempfile.mkdtemp(prefix="momeparity-"))
    kw = {"csv_path": str(tmp_root / "d.csv")} if w.plus else {}
    ref = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_gpu=True,
                          device=torch.device("cuda"), allow_tqdm=False, **kw)  # untouched, no hooks
    ref.initialize_from_trained_model_folder(str(w.work / "model"), (FOLD,), Path(w.cfg["checkpoint"]).name)
    for i, (name, files) in enumerate(jobs):
        res = w({m: str(f) for m, f in files.items()} if w.plus else str(files["image"]))
        f = res["features"]
        print(f"[parity] {model} {name}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in f.items()), flush=True)
        # logits come straight from the predictor, whose fp16 accumulator (logit x 1000-scaled Gaussian) can
        # overflow to inf on its own (F/inference/predict_from_raw_data.py:690-724): reported, not asserted
        assert all(torch.isfinite(v.float()).all() for k, v in f.items() if k != "logits")
        print(f"[parity] {model} {name} non-finite logits (fork fp16 accumulator): "
              f"{int((~torch.isfinite(f['logits'].float())).sum())}", flush=True)
        tmp = tmp_root / name
        if w.plus:
            ref.MultiMod = [int(m in files) for m in PLUS_MODALITIES]
        data, _, props = run_case(w, files, tmp / "nn")
        logits = ref.predict_logits_from_preprocessed_data(torch.from_numpy(data))
        print(f"[parity] {model} {name} logits vs fresh predictor: max|diff| {maxdiff(f['logits'], logits)}",
              flush=True)
        assert torch.equal(f["logits"], logits)
        if i == 0:  # the full nnUNetv2_predict entry point once per model (it reloads every checkpoint)
            _, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
                f["logits"], ref.plans_manager, ref.configuration_manager, ref.label_manager, props, True)
            cli = cli_probabilities(w, files, tmp / "cli")
            print(f"[parity] {model} {name} probabilities vs nnUNetv2_predict: max|diff| {maxdiff(probs, cli)}",
                  flush=True)
            assert np.array_equal(probs, cli, equal_nan=True)
        torch.cuda.empty_cache()
    shutil.rmtree(tmp_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepro", "features"])
    ap.add_argument("--model", choices=["mome", "mome_plus"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (stage_prepro if a.stage == "prepro" else stage_features)(a.model, a.out)
    print(f"[parity] {a.model} {a.stage}: OK")
