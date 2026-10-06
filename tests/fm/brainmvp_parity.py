"""BrainMVP wrapper vs the original repo pipeline. Run inside fm-brainmvp:

    PYTHONPATH=src $SFC_FM_ENVS/fm-brainmvp/bin/python tests/fm/brainmvp_parity.py

1. Preprocessing: the repo's transform objects, untouched, on a 2-file list [img, img] (the repo always
   loads >=2 modality files, stacked by MONAI into channels; identical copies leave the per-channel
   percentiles and the foreground box unchanged) -> channel 0 must equal the wrapper's 1-file output.
   downstream: `custom_transform(mode="val")` with a label (the image itself). pretrain: the Compose built
   inside `utils.data_utils.get_loader` (captured by stubbing the dataset), up to the second
   CenterCropForegroundd.
2. Network: an independently built repo RecModel (main.py:217-222 + main.py:266-270 loading) -> state
   dicts and encoder outputs equal; for the UniFormer also the Downstream `UniUnet(in_channels=1)` encoder
   loaded as Downstream/train_script.py:67-79 (without the 4-channel stem deletion).
3. Inference: the wrapper's sliding window on a 96^3 input (one window) == repo encoder on that window.
All comparisons bitwise on CPU (fp32; autocast is CUDA-only).
"""
from __future__ import annotations

import os
import sys
import time
import types
import warnings
from pathlib import Path

import torch

from sfc_gdn2.fm.brainmvp import PRETRAIN_ARGS, build, encoder_maps

warnings.filterwarnings("ignore")
REPO = os.path.expandvars("${SFC_FM_REPOS}/BrainMVP")
CKPT = os.path.expandvars("${SFC_FM_MODELS}/BrainMVP/BrainMVP_{}.pt")
SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
IMAGES = [SAMPLES / "pdgm0004_T1.nii.gz", SAMPLES / "ixi002_T1.nii.gz"]
sys.path[:0] = [REPO, REPO + "/Downstream"]


def maxdiff(a, b) -> float:
    assert a.shape == b.shape, (a.shape, b.shape)
    return float((a.float() - b.float()).abs().max())


def check(name: str, a, b) -> None:
    d = maxdiff(a, b)
    print(f"  {name}: shape {tuple(a.shape)} max|diff| = {d:.3g}", flush=True)
    assert d == 0, name


def repo_pretrain_transform():
    """The Compose that utils/data_utils.py:get_loader builds (lines 100-126), captured untouched."""
    import utils.data_utils as du
    captured = {}

    class Capture(torch.utils.data.Dataset):
        def __init__(self, data, transform, args):
            captured["t"] = transform

        def __len__(self):
            return 1

    du.load_mmri_sampled_list = lambda *a, **k: []
    du.MaskedInputDataset = Capture
    args = types.SimpleNamespace(num_workers=0, dataset=["none"], base_dir="", rank=0, debug=False, roi_x=96,
                                 roi_y=96, roi_z=96, sw_batch_size=1, cache_dataset=False, smartcache_dataset=False,
                                 distributed=False, batch_size=1)
    du.get_loader(args)
    steps = captured["t"].transforms
    names = [type(t).__name__ for t in steps]
    # [Sample_fix_seqd, Load, EnsureChannelFirst, Orientation, Spacing, CCF, Percentiles, CCF, RecordAffine, RandCrop, Pad, ToTensor]
    assert names[1] == "LoadImaged" and names[7] == "CenterCropForegroundd", names
    from monai.transforms import Compose
    return Compose(steps[1:8])


def main() -> None:
    from dataset.transforms import custom_transform
    t0 = time.time()
    wrappers = {}
    for arch in ("uniformer", "unet"):
        for mode in ("downstream", "pretrain"):
            cfg = {"model": "brainmvp", "repo": REPO, "checkpoint": CKPT.format(arch) if mode == "downstream" else None,
                   "args": {"arch": arch, "mode": mode}}
            wrappers[arch, mode] = build(cfg, "cpu")
    print(f"wrappers built ({time.time() - t0:.0f}s)", flush=True)

    print("1. preprocessing")
    repo_t = {"downstream": custom_transform(patch_shape=96, mode="val"), "pretrain": repo_pretrain_transform()}
    prepared = {}
    for img in IMAGES:
        for mode, t in repo_t.items():
            data = {"image": [str(img), str(img)]} | ({"label": str(img)} if mode == "downstream" else {})
            ref = t(data)["image"]
            ours = wrappers["uniformer", mode].preprocess(str(img))["image"]
            print(f" {img.name} [{mode}] range [{float(ours.min()):.3f}, {float(ours.max()):.3f}]")
            check("image", ours[0], ref[0])
            check("affine", ours.affine, ref.affine)
            prepared[img.name, mode] = ours

    print("2./3. network + single-window inference")
    from model.Uni_unet import UniUnet
    for arch in ("uniformer", "unet"):
        if arch == "uniformer":
            from models.Uniformer import RecModel
        else:
            from models.Unet import RecModel
        ref = RecModel(types.SimpleNamespace(device="cpu", **PRETRAIN_ARGS), dim=512)
        sd = torch.load(CKPT.format(arch), map_location="cpu")["state_dict"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(sd, "module.")
        print(f" {arch}: strict load {ref.load_state_dict(sd, strict=True)}")
        ref.eval()
        w = wrappers[arch, "downstream"]
        ours_sd = w.model.state_dict()
        assert ours_sd.keys() == ref.state_dict().keys()
        assert all(torch.equal(ours_sd[k], v) for k, v in ref.state_dict().items()), "state dicts differ"
        print("  state_dict: identical")
        if arch == "uniformer":
            uni = UniUnet(input_shape=96, in_channels=1, out_channels=3, multi_scale=True)
            ds = {k.replace("module.", "").replace("uniformer.", ""): v for k, v in sd.items()}  # train_script.py:72
            res = uni.load_state_dict(ds, strict=False)
            assert not [k for k in res.missing_keys if k.startswith("encoder.")], res.missing_keys
            uni.eval()
        for img in IMAGES:
            vol = prepared[img.name, "downstream"]
            c = [max(0, (s - 96) // 2) for s in vol.shape[1:]]
            crop = vol[:, c[0]:c[0] + 96, c[1]:c[1] + 96, c[2]:c[2] + 96]
            x = crop.as_tensor()[None].float()
            with torch.no_grad():
                if arch == "uniformer":
                    _, *outs = ref.encoder(x)
                    want = dict(zip(("x1", "x2", "x3", "x4"), (o.permute(0, 1, 3, 4, 2) for o in outs)))
                    _, *douts = uni.encoder(x)
                    for k, o in zip(want, douts):
                        check(f"{img.name} Downstream UniUnet.encoder {k}", o.permute(0, 1, 3, 4, 2), want[k])
                else:
                    want = dict(zip(("c1d", "c2d", "c3d", "c4d"), ref.encoder(x)))
                for k, v in encoder_maps(w.model, arch, x).items():
                    check(f"{img.name} encoder {k}", v, want[k])
                got = w.features({"image": crop})["features"]
            for k in want:
                check(f"{img.name} sliding-window(96^3) {k}", got[k], want[k])
    print(f"ALL PARITY CHECKS PASSED ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
