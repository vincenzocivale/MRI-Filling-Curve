"""MedicalNet parity: the repo's own test-phase pipeline vs the `medicalnet` wrapper, every configs/fm/medicalnet_*.

Run inside the env:  source configs/fm/leonardo.env
                     PYTHONPATH=src $SFC_FM_ENVS/fm-medicalnet/bin/python tests/fm/medicalnet_parity.py [image ...]
Original side (Tencent/MedicalNet, untouched): `BrainS18Dataset` phase "test" through a `DataLoader(batch_size=1)`
(test.py:88-89), the net from `model.generate_model` (phase "test", no_cuda) wrapped in `nn.DataParallel` exactly
as model.py:72-85 does, the pretrain copy of model.py:112-115 on that wrapped net, and `model(volume)` (test.py:57)
with a hook on layer4. On CPU `nn.DataParallel` is a passthrough, so the run needs no GPU. The background noise
of brains18.py:146 is drawn from numpy's global RNG: both sides seed it with the same value. Expected: bitwise.
"""
from __future__ import annotations

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from common import SAMPLES, check
from torch import nn
from torch.utils.data import DataLoader

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.medicalnet import build

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs/fm").glob("medicalnet_*.yaml"))


def original(cfg: dict, image: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Needs the repo on sys.path (the wrapper's build puts it there)."""
    from datasets.brains18 import BrainS18Dataset
    from model import generate_model
    a = cfg["args"]
    d, h, w = a["input_size"]
    sets = Namespace(model="resnet", model_depth=int(a["depth"]), resnet_shortcut="A" if a["depth"] in (18, 34) else "B",
                     input_D=d, input_H=h, input_W=w, n_seg_classes=2, no_cuda=True, phase="test",
                     pretrain_path=cfg["checkpoint"], new_layer_names=["conv_seg"], gpu_id=[0])
    net, _ = generate_model(sets)
    net = nn.DataParallel(net, device_ids=None)          # model.py:82 (CPU: passthrough)
    net_dict = net.state_dict()                          # model.py:83
    pretrain = torch.load(sets.pretrain_path, weights_only=True, map_location="cpu")
    pretrain_dict = {k: v for k, v in pretrain["state_dict"].items() if k in net_dict}  # model.py:112
    assert len(pretrain_dict) == len(pretrain["state_dict"]), "not every pretrained key matched"
    net_dict.update(pretrain_dict)
    net.load_state_dict(net_dict)
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "test.txt"
        lst.write_text(image + "\n")
        loader = DataLoader(BrainS18Dataset(tmp, str(lst), sets), batch_size=1, shuffle=False, num_workers=0)
        np.random.seed(int(a["seed"]))
        volume = next(iter(loader))
    net.eval()
    grabbed = {}
    net.module.layer4.register_forward_hook(lambda m, i, o: grabbed.__setitem__("x", o))
    with torch.no_grad():
        net(volume)
    return volume[0], grabbed["x"][0]


def main(images: list[str]) -> None:
    for path in CONFIGS:
        cfg = load_config(path)
        wrapper = build(cfg, "cpu")
        for image in images:
            x_ref, f_ref = original(cfg, image)
            check(f"{path.stem} {Path(image).name} input", wrapper.preprocess(image)["image"], x_ref)
            check(f"{path.stem} {Path(image).name} layer4", wrapper(image)["features"]["layer4"], f_ref)
    print("PARITY OK (bitwise)")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(SAMPLES / "ixi002_T1.nii.gz")])
