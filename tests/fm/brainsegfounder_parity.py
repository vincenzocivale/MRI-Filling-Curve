"""BrainSegFounder parity: the repo's own pipelines vs the `brainsegfounder` wrapper, one sample per checkpoint.

Run inside the env:  PYTHONPATH=src <fm-brainsegfounder>/bin/python tests/fm/brainsegfounder_parity.py [names...]
Original side (lab-smile/BrainSegFounder @ 0bbf432, untouched), per config:
- ukb / brats_ssl: the repo's `get_T1T2_dataloaders` on a real split json; one batch of its validation loader
  (random 96^3 crops, seeded) must equal the same boxes of the wrapper's preprocessed volume, and the repo's
  `SSLHead` (HF-README loading: full state_dict, `module.` stripped) `swinViT(crop)` must equal the wrapper's
  per-window predictor on that crop.
- brats_ft: `finetuning/utils/data_utils.get_loader(test_mode=True)` (with an all-zero label, the loader needs one)
  vs the wrapper input; test.py's model (SwinUNETR + strict load) inside `sliding_window_inference(roi 128,
  overlap 0.6)` with hooks on swinViT / decoder1 vs the wrapper features.
- atlas: `ATLASDataset` (bidsio) on a one-subject BIDS tree with the transforms of pretrain.py:105-108; ATLAS-ft
  via `torch.load` of the pickle (stub ATLASPredictor + DDP without process group), ATLAS-pt via the ATLAS
  `SSLHead` (pretrain.py:120-142; its `conv.*` decoder does not match the checkpoint, so only swinViT keys).
- MONAI: monai 1.2.0 `SwinTransformer` vs the source at MONAI a23c7f54 (the repo's pin) on the same input.
Expected: bitwise (fp32 CPU).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from torch import nn

from sfc_gdn2.fm.api import expand
from sfc_gdn2.fm.brainsegfounder import BRATS_MODALITIES, build, load_file, ssl_args

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
T1 = str(SAMPLES / "ixi002_T1.nii.gz")
BRATS = {"flair": str(SAMPLES / "pdgm0004_FLAIR.nii.gz"), "t1ce": str(SAMPLES / "pdgm0004_T1c.nii.gz"),
         "t1": str(SAMPLES / "pdgm0004_T1.nii.gz"), "t2": str(SAMPLES / "pdgm0004_T2.nii.gz")}
MONAI_PIN = Path("/leonardo_work/IscrC_SFMRI/fcorrent/repos/MONAI-a23c7f54/monai/networks/nets/swin_unetr.py")
WORST = [0.0]


def check(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a - b).abs().max().item()
    WORST[0] = max(WORST[0], d)
    print(f"  {name:28s} {tuple(a.shape)} max|diff|={d:.3g}", flush=True)
    assert torch.equal(a, b), name


def crop_box(crop, vol) -> tuple[slice, ...]:
    """Box of a RandSpatialCropSamplesd output inside the uncropped volume, from the two MetaTensor affines
    (MONAI 1.2's ToTensord resets applied_operations, the affine survives)."""
    start = torch.linalg.solve(vol.affine.double(), crop.affine.double()[:, 3])[:3]
    lo = [round(v) for v in start.tolist()]
    assert torch.allclose(start, torch.tensor(lo, dtype=torch.float64), atol=1e-4), start
    return tuple(slice(a, a + n) for a, n in zip(lo, crop.shape[1:]))


def ssl_case(cfg: dict, image, wrapper) -> None:
    from monai.utils import set_determinism
    a = cfg["args"]
    pipe = a["pipeline"]
    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp) / "split.json"
        rng = {"a_min": a["a_min"], "a_max": a["a_max"], "sw_batch_size": 2}
        if pipe == "ukb":
            du = load_file("orig_pretrain_du", Path(cfg["repo"]) / "pretrain/utils/data_utils.py")
            split.write_text(json.dumps({"training": [{"image": [image, image]}], "validation": [{"image": [image, image]}]}))
            args = ssl_args(split_json=str(split), modality="T1", in_channels=1, **rng)
        else:
            du = load_file("orig_brats_du", Path(cfg["repo"]) / "downstream/BraTS/ssl/utils/data_utils.py")
            split.write_text(json.dumps({"training": [{"fold": 0, "image": [image[m] for m in BRATS_MODALITIES]}]}))
            args = ssl_args(split_json=str(split), target_data_path="/", target_data_fold=0, T1T2_target_Brats=True,
                            modality="T1T2", in_channels=4, **rng)
        _, val_loader = du.get_T1T2_dataloaders(args, num_workers=0)
        set_determinism(seed=0)
        crops = val_loader.dataset[0]  # list of sw_batch_size dicts, each a random 96^3 crop
    prepared = wrapper.preprocess(image)
    vol = prepared["image"]
    head = load_file("orig_ssl_head", Path(cfg["repo"]) / "pretrain/models/ssl_head.py")
    ref = head.SSLHead(Namespace(spatial_dims=3, in_channels=a["in_channels"], feature_size=48, bottleneck_depth=768,
                                 num_swin_blocks_per_stage=a["depths"], num_heads_per_stage=[3, 6, 12, 24],
                                 dropout_path_rate=0.0, use_checkpoint=False)).eval()
    sd = torch.load(cfg["checkpoint"], map_location="cpu")["state_dict"]
    ref.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
    for i, c in enumerate(crops):
        box = crop_box(c["image"], vol)
        check(f"input crop{i} {[(s.start, s.stop) for s in box]}", c["image"].as_tensor(), vol[(slice(None), *box)])
        x = c["image"].as_tensor()[None]
        with torch.no_grad():
            want = ref.swinViT(x.contiguous())
            got = wrapper._predict(x)
        for s in range(5):
            check(f"crop{i} stage{s}", got[f"stage{s}"], want[s])
    monai_pin_case(ref.swinViT, crops[0]["image"].as_tensor()[None], a)
    out = wrapper(image)
    print("  wrapper features:", {k: tuple(v.shape) for k, v in out["features"].items()})


def monai_pin_case(swin: nn.Module, x: torch.Tensor, a: dict) -> None:
    spec = importlib.util.spec_from_file_location("swin_a23c7f54", MONAI_PIN)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    pinned = old.SwinTransformer(in_chans=a["in_channels"], embed_dim=48, window_size=(7,) * 3, patch_size=(2,) * 3,
                                 depths=a["depths"], num_heads=(3, 6, 12, 24), mlp_ratio=4.0, qkv_bias=True,
                                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                                 use_checkpoint=False, spatial_dims=3).eval()
    pinned.load_state_dict(swin.state_dict())
    with torch.no_grad():
        for s, (p, q) in enumerate(zip(pinned(x), swin(x))):
            check(f"monai a23c7f54 vs 1.2 stage{s}", p, q)


def brats_ft_case(cfg: dict, image: dict, wrapper) -> None:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SwinUNETR
    du = load_file("orig_ft_du", Path(cfg["repo"]) / "downstream/BraTS/finetuning/utils/data_utils.py")
    with tempfile.TemporaryDirectory() as tmp:
        ref_img = nib.load(image["flair"])
        nib.save(nib.Nifti1Image(np.zeros(ref_img.shape, np.uint8), ref_img.affine), f"{tmp}/seg.nii.gz")
        Path(f"{tmp}/folds.json").write_text(json.dumps({"training": [
            {"fold": 0, "image": [image[m] for m in BRATS_MODALITIES], "label": f"{tmp}/seg.nii.gz"}]}))
        args = Namespace(data_dir="/", json_list=f"{tmp}/folds.json", fold=0, test_mode=True, distributed=False,
                         workers=0, roi_x=128, roi_y=128, roi_z=128, batch_size=1)
        batch = next(iter(du.get_loader(args)))
    x_ref = batch["image"]
    prepared = wrapper.preprocess(image)
    check("input", x_ref[0], prepared["image"])
    model = SwinUNETR(img_size=128, in_channels=4, out_channels=3, feature_size=48, drop_rate=0.0, attn_drop_rate=0.0,
                      dropout_path_rate=0.0, use_checkpoint=False)  # test.py:72-81
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu")["state_dict"])  # test.py:82-83
    model.eval()

    def predictor(x):
        grabbed = {}
        h1 = model.swinViT.register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", o))
        h2 = model.decoder1.register_forward_hook(lambda m, i, o: grabbed.__setitem__("d", o))
        model(x)
        h1.remove()
        h2.remove()
        return {**{f"stage{s}": t for s, t in enumerate(grabbed["h"])}, "decoder": grabbed["d"]}

    with torch.no_grad():  # test.py:87-93 inferer, with the encoder/decoder outputs instead of logits
        want = sliding_window_inference(x_ref.as_tensor(), roi_size=[128] * 3, sw_batch_size=1, predictor=predictor,
                                        overlap=0.6)
    got = wrapper(image)["features"]
    for k, v in want.items():
        check(k, got[k], v[0])


def atlas_case(cfg: dict, image: str, wrapper) -> None:
    import monai
    add = Path(cfg["repo"]) / "downstream/ATLAS"
    sys.path.insert(0, str(add))
    from dataset.ATLASDataset import ATLASDataset, data_entities, target_entities
    with tempfile.TemporaryDirectory() as tmp:  # minimal BIDS tree: derivatives/ATLAS/sub-01/ses-1/anat
        anat = Path(tmp) / "derivatives/ATLAS/sub-01/ses-1/anat"
        anat.mkdir(parents=True)
        (Path(tmp) / "dataset_description.json").write_text(json.dumps({"Name": "ATLAS", "BIDSVersion": "1.6.0"}))
        (Path(tmp) / "derivatives/ATLAS/dataset_description.json").write_text(json.dumps(
            {"Name": "ATLAS", "BIDSVersion": "1.6.0", "DatasetType": "derivative", "GeneratedBy": [{"Name": "ATLAS"}]}))
        img = nib.load(image)
        (anat / "sub-01_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").symlink_to(Path(image).resolve())
        nib.save(nib.Nifti1Image(np.zeros(img.shape, np.uint8), img.affine),
                 anat / "sub-01_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz")
        ds = ATLASDataset(data_entities, target_entities, data_derivatives_names=["ATLAS"],
                          target_derivatives_names=["ATLAS"], root_dir=tmp,
                          transform=monai.transforms.Compose([monai.transforms.ToTensor(),
                                                             monai.transforms.Resize(spatial_size=[96, 96, 96])]))
        x_ref, _ = ds[0]
    x_ref = x_ref.as_tensor() if hasattr(x_ref, "as_tensor") else x_ref
    a = cfg["args"]
    x_in = torch.cat([x_ref, x_ref]) if a["in_channels"] == 2 else x_ref
    prepared = wrapper.preprocess(image)
    check("input", prepared["image"], x_in)
    if a.get("pickled"):
        mm = types.ModuleType("__mp_main__")
        mm.ATLASPredictor = type("ATLASPredictor", (nn.Module,), {})
        sys.modules["__mp_main__"] = mm
        ddp = torch.nn.parallel.DistributedDataParallel
        setstate, ddp.__setstate__ = ddp.__setstate__, lambda self, st: self.__dict__.update(st)
        try:
            model = torch.load(cfg["checkpoint"], map_location="cpu").module.base_model.eval()
        finally:
            ddp.__setstate__ = setstate
        swin = model.swinViT
    else:
        head = load_file("orig_atlas_ssl_head", add / "models/ssl_head.py")
        ref = head.SSLHead(spatial_dimensions=3, in_channels=2, feature_size=48, dropout_rate=0.0,
                           stochastic_depth_rate=0.0, depths=a["depths"], heads=[3, 6, 12, 24], use_checkpoint=False)
        sd = torch.load(cfg["checkpoint"], map_location="cpu")["state_dict"]
        res = ref.load_state_dict({k: v for k, v in sd.items() if k.startswith("swinViT.")}, strict=False)
        assert not [k for k in res.missing_keys if k.startswith("swinViT.")], res.missing_keys
        swin = ref.eval().swinViT
    with torch.no_grad():
        want = swin(x_in[None].contiguous())
    got = wrapper(image)["features"]
    for s in range(5):
        check(f"stage{s}", got[f"stage{s}"], want[s][0])


def main(names: list[str]) -> None:
    for name in names:
        cfg = expand(yaml.safe_load((ROOT / f"configs/fm/{name}.yaml").read_text()))
        print(f"== {name}", flush=True)
        wrapper = build(cfg, "cpu")
        pipe = cfg["args"]["pipeline"]
        if pipe == "ukb":
            ssl_case(cfg, T1, wrapper)
        elif pipe == "brats_ssl":
            ssl_case(cfg, BRATS, wrapper)
        elif pipe == "brats_ft":
            brats_ft_case(cfg, BRATS, wrapper)
        else:
            atlas_case(cfg, T1, wrapper)
    print(f"OK: bitwise equal (max abs diff {WORST[0]})")


if __name__ == "__main__":
    os.environ.setdefault("SFC_FM_ENVS", "/leonardo_work/IscrC_SFMRI/fcorrent/envs")
    os.environ.setdefault("SFC_FM_REPOS", "/leonardo_work/IscrC_SFMRI/fcorrent/repos")
    os.environ.setdefault("SFC_FM_MODELS", "/leonardo_work/IscrC_SFMRI/fcorrent/models")
    main(sys.argv[1:] or ["bsf_ukb_pretrain", "bsf_atlas_pretrain", "bsf_atlas_finetune", "bsf_brats_pretrain",
                          "bsf_brats_finetune"])
