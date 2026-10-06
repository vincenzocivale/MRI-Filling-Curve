"""MoME (MICCAI'24) and MoME+ (TMI'25), ZhangxinruBIT/MoME: mixture of modality-expert nnU-Nets, run with
the repo's own nnU-Net v2.2 forks. Refs: F = MoME_foundation/nnunetv2, P = MoME_plus/nnunetv2, C = Codes_prepro.

Imports: both forks are packages named `nnunetv2` and each patches dynamic_network_architectures (DNA), so a
process holds exactly one fork (api.py spawns one worker per config). `mome` -> MoME_foundation (uses
F/dynamic_network_architectures); `mome_plus` -> MoME_plus + MoME_plus/nnunetv2, so the top-level
`dynamic_network_architectures` imported by P/utilities/plans_handling/plans_handler.py:3 is
P/dynamic_network_architectures (num_experts=4 -> the released 324-channel `decoder.transpconvs.0`), not the
5-expert copy that P/utilities/get_network_from_plans.py:3 puts on sys.path afterwards. Asserted at build.

preprocess(image) (`args.prepro: true`, default): C/one-step.py run on one case with the repo's functions:
  1. C/AffineRegPrepro.py `generate_new_sample` (:349): antsRegistration rigid+affine (:42-67) to MNI152 1 mm,
     full-head template if the input has a skull (`args.skull`, = the `-ss` flag), else the skull-stripped one
     (:304-313). ANTs is made deterministic: ANTS_RANDOM_SEED=`args.ants_seed`, 1 ITK thread.
  2. `args.skull` only: C/skull_stripping.py `stripping` (:9-22): multiply by the template brain mask.
  3. C/crop.py `stripping` (:9-23): voxels [20:180, 20:216, 0:160] -> 160x196x160.
  then the fork's DefaultPreprocessor.run_case (F/preprocessing/preprocessors/default_preprocessor.py:40-90):
  SimpleITK read (array axes z,y,x), no crop (the fork disables it, F/preprocessing/cropping/cropping.py:33),
  whole-volume z-score (plans use_mask_for_norm=false), resampling to 1 mm (no-op on C output).
  `args.prepro: false`: the inputs are already C outputs (MNI152 1 mm, 160x196x160) and go to run_case directly.
  MoME takes one image of any modality. MoME+ takes {t1, t1ce, t2, flair} (any non-empty subset): MultiMod is
  the availability mask; P/preprocessing/preprocessors/default_preprocessor.py:175-209 fills missing channels
  by file duplication, so the files are laid out as `BraTS<mod>/case_<mod>_0000.nii.gz` (the names it rewrites).

features(prepared): nnUNetPredictor.predict_logits_from_preprocessed_data, untouched (sliding window 128^3,
step 0.5, Gaussian, 8x mirroring, fp16 autocast on CUDA; F/inference/predict_from_raw_data.py:501-730,
P/...:513-775). MoME: 5 experts (checkpoint_best1..5 = T1, T1ce, T2, FLAIR, DWI, F/training/nnUNetTrainer/
nnUNetTrainer.py:214-242; the jointly fine-tuned ones, not Pretrained_Experts) + gating aggregator on
cat(image, expert first-stage skips) (F/inference/predict_from_raw_data.py:602-610). MoME+: dispatch network
-> 4 experts -> aggregator with the modality one-hot at its bottleneck (P/inference/predict_from_raw_data.py:
617-667; needs CUDA: P/dynamic_network_architectures/architectures/unet.py:91). Captured with forward hooks:
  bottleneck  [T, M, 320, 4, 4, 4]  aggregator encoder last stage per tile (T) and mirror pass (M), raw: computed
              on the input flipped along meta.mirror_axes[m] (not flipped back). Canonical: the paper's t-SNE
              uses "the latent features obtained at the bottleneck of the gating network" (arXiv 2405.10246).
  skip<s>     [T, M, C_s, ...]      other aggregator encoder stages (`args.skip_stages`), same layout.
  decoder     [32, *shape]          aggregator decoder last stage, mirror-averaged and Gaussian-stitched with the
              predictor's own accumulation (re-applied to the captured per-tile logits and asserted bitwise
              equal to the predictor's logits, every call), but in float32: the predictor's fp16 accumulator
              overflows (inf) on these activations times its 1000-scaled Gaussian. `args.dense: false` drops it.
  logits      [C, *shape]           the predictor's fused output (softmax gives the segmentation).
  bottleneck_mean [320]             derived: mean over tiles, mirror passes and space.
"""
from __future__ import annotations

import ast
import csv
import importlib
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import Wrapper, add_to_path

FORKS = {"mome": ("MoME_foundation",), "mome_plus": ("MoME_plus", "MoME_plus/nnunetv2")}
DNA = {"mome": "MoME_foundation/nnunetv2/dynamic_network_architectures",
       "mome_plus": "MoME_plus/nnunetv2/dynamic_network_architectures"}
PLUS_MODALITIES = ("t1", "t1ce", "t2", "flair")  # order of MultiMod and of the experts (P/.../nnUNetTrainer.py:215-236)
CROP = ((20, 180), (20, 216), (0, 160))           # C/crop.py:16, nibabel voxel axes of the template grid
FOLD = "MoME"


def mirror_passes(axes: tuple[int, ...] | None) -> list[tuple[int, ...]]:
    """Spatial axes flipped in each forward pass, in the predictor's call order (F/.../predict_from_raw_data.py:
    602-644, P/...:667-687)."""
    if axes is None:
        return [()]
    combos = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    return [(), *[c for c in combos if all(a in axes for a in c)]]


def slices_to_list(sl) -> list[list[int]]:
    return [[s.start, s.stop] for s in sl]


class MoME(Wrapper):
    name = "mome"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        self.key = cfg["model"]
        self.plus = self.key == "mome_plus"
        self.name = self.key
        self.modalities = PLUS_MODALITIES if self.plus else ("image",)
        if self.plus and device != "cuda":
            raise RuntimeError("mome_plus needs CUDA: the aggregator calls ModEmbd.cuda() "
                               "(P/dynamic_network_architectures/architectures/unet.py:91).")
        self.repo = Path(cfg["repo"])
        self.codes = self.repo / "Codes_prepro"
        self._tmp = tempfile.TemporaryDirectory(prefix="mome-")
        self.work = Path(self._tmp.name)
        # the repo's scripts rewrite paths by substring (images->labels, BraTSt1->BraTSt1ce, _t1_->_t1ce_)
        if any(s in str(self.work) for s in ("images", "BraTS", "_t1", "_t2", "_flair")):
            raise RuntimeError(f"temporary directory {self.work} contains a substring the MoME scripts rewrite.")
        add_to_path(*[self.repo / d for d in FORKS[self.key]], self.codes)
        self.predictor = self._build_predictor(Path(cfg["checkpoint"]))
        self.skip_stages = [int(s) for s in self.args.get("skip_stages", [4, 5])]
        self.dense = bool(self.args.get("dense", True))

    # ------------------------------------------------------------------ model
    def _build_predictor(self, checkpoint: Path):
        import nnunetv2
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        fork = self.repo / FORKS[self.key][0]
        if not Path(nnunetv2.__file__).resolve().is_relative_to(fork.resolve()):
            raise RuntimeError(f"nnunetv2 resolved to {nnunetv2.__file__}, expected inside {fork}.")
        # initialize_from_trained_model_folder reads <dir>/{plans,dataset}.json and <dir>/fold_<f>/<checkpoint>
        # (F/inference/predict_from_raw_data.py:67-110); the HF release ships nnUNetPlans.json next to the files.
        farm = self.work / "model"
        farm.mkdir()
        (farm / "plans.json").symlink_to(checkpoint.parent / "nnUNetPlans.json")
        (farm / "dataset.json").symlink_to(checkpoint.parent / "dataset.json")
        (farm / f"fold_{FOLD}").symlink_to(checkpoint.parent)
        kwargs: dict[str, Any] = {
            "tile_step_size": float(self.args.get("tile_step_size", 0.5)), "use_gaussian": True,
            "use_mirroring": bool(self.args.get("use_mirroring", True)), "perform_everything_on_gpu": True,
            "device": torch.device(self.device), "verbose": False, "verbose_preprocessing": False,
            "allow_tqdm": False}
        if self.plus:
            kwargs["csv_path"] = str(self.work / "dispatch.csv")  # the predictor logs the dispatch matrix here
        predictor = nnUNetPredictor(**kwargs)
        predictor.initialize_from_trained_model_folder(str(farm), (FOLD,), checkpoint.name)
        dna = (self.repo / DNA[self.key]).resolve()
        for net in (predictor.network, predictor.network1):
            src = Path(inspect.getfile(type(net))).resolve()
            if not src.is_relative_to(dna):
                raise RuntimeError(f"{type(net).__name__} comes from {src}, expected the fork's DNA in {dna}.")
        if self.plus:
            import dynamic_network_architectures
            if not Path(dynamic_network_architectures.__file__).resolve().is_relative_to(dna):
                raise RuntimeError(f"dynamic_network_architectures resolved to "
                                   f"{dynamic_network_architectures.__file__}, expected {dna}.")
        return predictor

    # ------------------------------------------------------------------ preprocessing
    def _codes_prepro(self, src: str, work: Path) -> tuple[Path, dict]:
        """C/one-step.py:49-62 for one case, with the repo's own functions (module globals set as their
        __main__ blocks do: AffineRegPrepro.py:384, skull_stripping.py:47-53)."""
        import nibabel as nib
        import SimpleITK as sitk
        skull = bool(self.args.get("skull", True))
        seed = int(self.args.get("ants_seed", 1))
        images = work / "imagesTs"
        images.mkdir(parents=True)
        case = images / "case_0000.nii.gz"
        shutil.copyfile(src, case)
        (work / "Template").symlink_to(self.codes / "Template")  # the scripts use relative 'Template/...' paths
        env_bin = str(Path(sys.executable).parent)  # antsRegistration of this env (the worker is not activated)
        if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        os.environ["ANTS_RANDOM_SEED"] = str(seed)
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
        reg = importlib.import_module("AffineRegPrepro")
        cwd = Path.cwd()
        os.chdir(work)
        try:
            reg.Path = str(images)
            reg.generate_new_sample(str(case), skull)
            out = work / "imagesTs_RegMNI152" / "case_to_MNI152_0000.nii.gz"
            if not out.exists():  # AffineRegPrepro.py:222 runs antsRegistration via os.system, unchecked
                raise RuntimeError(f"antsRegistration produced no output for {src}.")
            if skull:
                ss = importlib.import_module("skull_stripping")
                ss.Path, ss.mask_ref = str(out.parent), "Template/standard_mni152_BrainMask.nii.gz"
                ss.dt = nib.load(ss.mask_ref)
                (work / "imagesTs_RegMNI152_skull_stripping").mkdir()
                ss.stripping(str(out))
                out = work / "imagesTs_RegMNI152_skull_stripping" / out.name
            importlib.import_module("crop").stripping(str(out), Lab=True)
        finally:
            os.chdir(cwd)
        tx = sitk.ReadTransform(str(work / "Affine2MNI152Matrix" / "case_to_MNI152" / "warp_0GenericAffine.mat"))
        template = "standard_mni152.nii.gz" if skull else "standard_mni152_skull_strip.nii.gz"
        shift = np.eye(4)
        shift[:3, 3] = [c[0] for c in CROP]
        meta = {"template": template, "skull_mask": skull, "ants_seed": seed,
                "ants_transform": {"type": tx.GetName(), "parameters": list(tx.GetParameters()),
                                   "fixed_parameters": list(tx.GetFixedParameters()),
                                   "convention": "ITK, LPS mm; maps template points to input points"},
                "crop_voxels": [list(c) for c in CROP],
                # crop.py:21 keeps the uncropped template affine in the file header; this is the true one
                "affine_ras": nib.load(self.codes / "Template" / template).affine @ shift}
        return out, meta

    def preprocess(self, image) -> dict:
        if self.plus:
            if not isinstance(image, dict) or not image:
                raise ValueError(f"mome_plus takes a non-empty {{modality: path}} dict, modalities {PLUS_MODALITIES}.")
            images = {m: image[m] for m in PLUS_MODALITIES if m in image}
        else:
            images = {"image": image if isinstance(image, str) else image["image"]}
        case = Path(tempfile.mkdtemp(dir=self.work, prefix="case"))
        try:
            files, prepro = {}, {}
            for m, path in images.items():
                if self.args.get("prepro", True):
                    files[m], prepro[m] = self._codes_prepro(path, case / m)
                else:
                    files[m], prepro[m] = Path(path), {"input_space": "Codes_prepro output (args.prepro=false)"}
            pm, cm, dj = self.predictor.plans_manager, self.predictor.configuration_manager, self.predictor.dataset_json
            preprocessor = cm.preprocessor_class(verbose=False)
            if self.plus:
                for m, f in files.items():
                    dst = case / "nnunet" / f"BraTS{m}" / f"case_{m}_0000.nii.gz"
                    dst.parent.mkdir(parents=True)
                    dst.symlink_to(Path(f).resolve())
                multimod = [int(m in files) for m in PLUS_MODALITIES]
                first = next(m for m in PLUS_MODALITIES if m in files)
                data, _, props = preprocessor.run_case(
                    [str(case / "nnunet" / f"BraTS{first}" / f"case_{first}_0000.nii.gz")], None, pm, cm, dj,
                    MultiMod=multimod)
            else:
                multimod = None
                data, _, props = preprocessor.run_case([str(files["image"])], None, pm, cm, dj)
        finally:
            shutil.rmtree(case)
        return {"data": data, "properties": props, "multimod": multimod, "prepro": prepro}

    # ------------------------------------------------------------------ features
    def _stitch(self, tiles: list[torch.Tensor], slicers: list, padded: tuple[int, ...], revert,
                dtype: torch.dtype = torch.half) -> torch.Tensor:
        """The accumulation of nnUNetPredictor.predict_sliding_window_return_logits (F/inference/
        predict_from_raw_data.py:686-726): Gaussian-weighted sum / weight in half precision, padding reverted."""
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        p = self.predictor
        dev = p.device if p.perform_everything_on_gpu else torch.device("cpu")
        acc = torch.zeros((tiles[0].shape[0], *padded), dtype=dtype, device=dev)
        n = torch.zeros(padded, dtype=torch.half, device=dev)
        g = compute_gaussian(tuple(p.configuration_manager.patch_size), sigma_scale=1. / 8,
                             value_scaling_factor=1000, device=dev) if p.use_gaussian else 1
        for t, sl in zip(tiles, slicers):
            acc[sl] += t.to(dev) * g if p.use_gaussian else t.to(dev)
            n[sl[1:]] += g
        acc /= n
        return acc[(slice(None), *revert[1:])]

    def features(self, prepared: dict) -> dict:
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        p = self.predictor
        data = torch.from_numpy(prepared["data"])
        if self.plus:
            p.MultiMod = prepared["multimod"]
            Path(p.csv_path).unlink(missing_ok=True)
        padded, revert = pad_nd_image(data, p.configuration_manager.patch_size, "constant", {"value": 0}, True, None)
        slicers = p._internal_get_sliding_window_slicers(padded.shape[1:])
        passes = mirror_passes(p.allowed_mirroring_axes if p.use_mirroring else None)
        n_stages = len(p.configuration_manager.conv_kernel_sizes)
        stages = sorted({*self.skip_stages, n_stages - 1})

        skips: list[list[torch.Tensor]] = []
        tile_logits: list[torch.Tensor] = []
        tile_dec: list[torch.Tensor] = []
        dec_sum: list[torch.Tensor] = []

        def on_encoder(_m, _i, out):
            skips.append([out[s][0].detach().to("cpu", copy=True) for s in stages])  # batch of 1

        def on_decoder(_m, _i, out):
            axes = tuple(a + 2 for a in passes[len(dec_sum) % len(passes)])
            x = torch.flip(out, axes) if axes else out.clone()
            dec_sum.append(x)

        original = p._internal_maybe_mirror_and_predict

        def tile(x, *a):
            dec_sum.clear()
            r = original(x, *a)
            tile_logits.append(r[0])
            if self.dense:  # same order of operations as the logits: first pass, += the others, /= count
                s = dec_sum[0]
                for d in dec_sum[1:]:
                    s += d
                if len(passes) > 1:
                    s /= len(passes)
                tile_dec.append(s[0])
            return r

        hooks = [p.network.encoder.register_forward_hook(on_encoder)]
        if self.dense:
            hooks.append(p.network.decoder.stages[-1].register_forward_hook(on_decoder))
        p._internal_maybe_mirror_and_predict = tile
        try:
            logits = p.predict_logits_from_preprocessed_data(data)
        finally:
            del p._internal_maybe_mirror_and_predict
            for h in hooks:
                h.remove()
        if len(skips) != len(slicers) * len(passes):
            raise RuntimeError(f"captured {len(skips)} aggregator passes, expected {len(slicers)}x{len(passes)}.")
        stitched = self._stitch(tile_logits, slicers, tuple(padded.shape[1:]), revert).cpu()
        if not torch.equal(stitched, logits):
            raise RuntimeError("re-stitched tile logits differ from the predictor's logits.")

        feats: dict[str, torch.Tensor] = {}
        for i, s in enumerate(stages):
            x = torch.stack([c[i] for c in skips])
            feats["bottleneck" if s == n_stages - 1 else f"skip{s}"] = x.view(len(slicers), len(passes), *x.shape[1:])
        if self.dense:
            # float32 accumulator: decoder activations x the 1000-scaled Gaussian overflow fp16 (the logits don't)
            feats["decoder"] = self._stitch(tile_dec, slicers, tuple(padded.shape[1:]), revert, torch.float32).cpu()
        feats["logits"] = logits
        feats["bottleneck_mean"] = feats["bottleneck"].float().mean(dim=(0, 1, 3, 4, 5))
        strides = np.cumprod(np.array(p.configuration_manager.pool_op_kernel_sizes), axis=0)
        meta = {
            "array_axes": "nnU-Net array of the preprocessed image in SimpleITK order (z, y, x) = reversed "
                          "nibabel voxel axes; logits/decoder cover it voxel for voxel",
            "shape": list(data.shape[1:]), "padded_shape": list(padded.shape[1:]),
            "revert_padding": slices_to_list(revert[1:]),
            "tiles": [slices_to_list(sl[1:]) for sl in slicers],
            "mirror_axes": [list(a) for a in passes],
            "patch_size": list(p.configuration_manager.patch_size), "tile_step_size": p.tile_step_size,
            "gaussian_sigma_scale": 1. / 8, "skip_strides": {s: strides[s].tolist() for s in stages},
            "nnunet_properties": prepared["properties"], "prepro": prepared["prepro"]}
        if self.plus:
            with open(p.csv_path) as f:
                row = list(csv.reader(f))[-1]
            meta["multimod"] = dict(zip(PLUS_MODALITIES, prepared["multimod"]))
            meta["dispatch_matrix"] = [ast.literal_eval(c) for c in row]  # P/inference/predict_from_raw_data.py:650
        return {"features": feats, "canonical": "bottleneck", "meta": meta, "derived": ["bottleneck_mean"]}


def build(cfg: dict, device: str) -> MoME:
    return MoME(cfg, device)
