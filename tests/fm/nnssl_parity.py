"""Parity of the OpenMind / nnFoundation wrapper (src/sfc_gdn2/fm/nnssl.py) with the original code, run inside fm-nnssl:

    source configs/fm/leonardo.env
    PYTHONPATH=src $SFC_FM_ENVS/fm-nnssl/bin/python tests/fm/nnssl_parity.py cpu configs/fm/{openmind,nnfoundation}_*.yaml
    PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 $SFC_FM_ENVS/fm-nnssl/bin/python tests/fm/nnssl_parity.py gpu ...

cpu: (1) preprocessing: wrapper.preprocess == nnssl `preprocess_case` / `no_resample_preprocess_case` called directly
     with masks=None (bitwise), for both samples; (2) weights: every encoder tensor of the wrapper's network
     torch.equal to a network built and loaded independently with the official
     `PretrainedTrainer[_Primusx].build_network_architecture` + `load_pretrained_weights`, and to the raw checkpoint
     tensors under the plan's keys (pos-embed: to nnU-Net's `interpolate_patch_embed_3d` when the patch differs).
gpu: (3) forward: every sliding-window tile and the SSL3D crop of the wrapper == the independent official network
     on the same input (ResEnc-L `network.encoder`, Primus `down_projection -> rearrange -> eva` as primus.py:165-179),
     same precision (fp16 autocast), deterministic algorithms -> bitwise.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.nnssl import _torch_load_on_cpu, build

SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")
IMAGES = [SAMPLES / "ixi002_T1.nii.gz", SAMPLES / "pdgm0004_T1.nii.gz"]


def official_network(w, device, pt_patch=None):
    """Independent copy of the official build + load (same calls as like_nnssl -> nnUNetv2_train_pretrained).
    pt_patch=None: the pre-training patch of the checkpoint's embedded plan, as like_nnssl.py:139 passes it."""
    from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import PretrainedTrainer
    from nnunetv2.training.nnUNetTrainer.pretraining.thrp_primusx_finetuning import PretrainedTrainer_Primusx

    ap = w.adaptation_plan
    trainer = PretrainedTrainer_Primusx if w.is_primus else PretrainedTrainer
    arch = ap["architecture_plans"]
    net = trainer.build_network_architecture(w.arch, deepcopy(arch["arch_kwargs"]),
                                             arch["arch_kwargs_requiring_import"], input_patch_size=w.patch,
                                             num_input_channels=1, num_output_channels=1,
                                             enable_deep_supervision=False)
    if pt_patch is None:
        pt_patch = next(iter(ap["pretrain_plan"]["configurations"].values()))["patch_size"]
    with _torch_load_on_cpu():
        net, _ = trainer.load_pretrained_weights(net, w.cfg["checkpoint"], ap["pretrain_num_input_channels"], 1,
                                                 pt_patch, w.patch, ap["key_to_encoder"], ap["key_to_stem"],
                                                 tuple(ap["keys_to_in_proj"]), ap["key_to_lpe"])
    return net.to(device).eval()


def encoder_state(net, is_primus: bool) -> dict[str, torch.Tensor]:
    sd = net.state_dict()
    roots = ("down_projection.", "eva.") if is_primus else ("encoder.",)
    return {k: v for k, v in sd.items() if k.startswith(roots)}


def check_cpu(w) -> None:
    from nnssl.preprocessing.preprocessors.default_preprocessor import preprocess_case
    from nnssl.preprocessing.preprocessors.no_resampling_preprocessor import no_resample_preprocess_case
    from nnunetv2.utilities.load_weights_utils import interpolate_patch_embed_3d

    fn = no_resample_preprocess_case if w.config_plan.spacing_style == "noresample" else preprocess_case
    for img in IMAGES:
        data, props = w.plan.image_reader_writer_class()().read_images([str(img)])
        ref, _ = fn(data, None, props, w.plan, w.config_plan, False)
        got = w.preprocess(str(img))["data"]
        assert got.dtype == ref.dtype and got.shape == ref.shape and np.array_equal(got, ref), img.name
        print(f"  preprocess {img.name}: shape {ref.shape} bitwise equal (max|diff| 0)")

    ap = w.adaptation_plan
    shipped = json.loads(Path(w.cfg["checkpoint"]).with_name("adaptation_plan.json").read_text())
    emb_patch = next(iter(ap["pretrain_plan"]["configurations"].values()))["patch_size"]
    emb, shp = deepcopy(ap), deepcopy(shipped)
    for d in (emb, shp):
        d.pop("citations", None)
        d.pop("trainer_name", None)
        next(iter(d["pretrain_plan"]["configurations"].values())).pop("patch_size")
    assert json.loads(json.dumps(emb)) == json.loads(json.dumps(shp)), "plans differ beyond the patch size"
    if list(emb_patch) != list(w.pt_patch):
        err = None
        try:
            official_network(w, "cpu")
        except Exception as e:  # noqa: BLE001 - documenting the original failure
            err = e
        assert err is not None, "embedded-plan load was expected to fail"
        print(f"  embedded plan patch {list(emb_patch)} != shipped {w.pt_patch}: untouched official load raises "
              f"{type(err).__name__}; using the shipped adaptation_plan.json")
    ref_sd = encoder_state(official_network(w, "cpu", w.pt_patch), w.is_primus)
    got_sd = encoder_state(w.network.cpu(), w.is_primus)
    assert ref_sd.keys() == got_sd.keys()
    bad = [k for k in ref_sd if not torch.equal(ref_sd[k], got_sd[k])]
    assert not bad, bad[:5]
    with _torch_load_on_cpu():
        raw = torch.load(w.cfg["checkpoint"], weights_only=True)["network_weights"]
    # ResEnc state_dicts list each conv/norm twice (`conv.*` and `all_modules.<i>.*`, one tensor). ResEncL-S3D
    # stores two DIFFERENT tensors there: Spark's conversion clones the module per name (nnssl architectures/
    # spark_utils.py:334-335) and only `all_modules` runs in forward, so `all_modules` is the reference.
    alias: dict[int, list[str]] = {}
    for k, v in got_sd.items():
        alias.setdefault(v.data_ptr(), []).append(k)
    n_raw = n_stale = 0
    for k, v in got_sd.items():
        k_ref = next((a for a in alias[v.data_ptr()] if ".all_modules." in a), k)
        if w.is_primus:
            src = (ap["key_to_stem"] + k[len("down_projection"):] if k.startswith("down_projection.")
                   else ap["key_to_encoder"] + k[len("eva"):])
        else:
            src = k_ref
        r = raw[src]
        n_stale += not w.is_primus and k != k_ref and not torch.equal(raw[k], r)
        if k == "eva.pos_embed" and r.shape != v.shape:
            assert r.shape[1] == np.prod([p // 8 for p in w.pt_patch]), "pos-embed vs pre-training patch"
            r = interpolate_patch_embed_3d(r, dict(zip("xyz", [p // 8 for p in w.pt_patch])),
                                           dict(zip("xyz", [p // 8 for p in w.patch])))
        assert torch.equal(r, v), k
        n_raw += 1
    print(f"  weights: {len(got_sd)} encoder tensors torch.equal to official load_pretrained_weights; "
          f"{n_raw} equal to raw checkpoint['{ap['key_to_encoder']}'/'{ap['key_to_stem']}'] "
          f"({n_stale} stale non-`all_modules` duplicates in the file ignored; pos-embed "
          f"{tuple(raw[ap['key_to_lpe']].shape) if ap['key_to_lpe'] else '-'} -> "
          f"{tuple(got_sd['eva.pos_embed'].shape) if w.is_primus else '-'})")


def check_gpu(w) -> None:
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from datasets.preprocess_3D_data.crop_to_mask import crop_center_with_padding_np, get_mask_center

    net = official_network(w, "cuda", w.pt_patch)

    def ref_encode(x):
        if w.is_primus:  # primus.py:165-179
            t = net.down_projection(x)
            _, _, a, b, c = t.shape
            tok, _ = net.eva(rearrange(t, "b c w h d -> b (w h d) c"))
            return rearrange(tok, "b (w h d) c -> b c w h d", w=a, h=b, d=c)[0]
        return net.encoder(x)[-1][0]

    last = "tokens" if w.is_primus else "stage5"
    for img in IMAGES[:1]:
        prep = w.preprocess(str(img))
        out = w.features(prep)
        padded, _ = pad_nd_image(torch.from_numpy(prep["data"]), w.patch, "constant", {"value": 0}, True, None)
        slicers = w.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        crop = crop_center_with_padding_np(prep["data"][0], get_mask_center(prep["nonzero"].astype(np.uint8)),
                                           tuple(w.patch))
        with torch.inference_mode(), torch.autocast("cuda", enabled=True):
            refs = [ref_encode(padded[s][None].cuda()).float().cpu() for s in slicers]
            ref_crop = ref_encode(torch.from_numpy(crop)[None, None].cuda()).float().cpu()
        tiles = out["features"][f"tiles_{last}"]
        d_t = max((a - b).abs().max().item() for a, b in zip(tiles, refs))
        d_c = (out["features"]["crop_last"] - ref_crop).abs().max().item()
        nonfinite = [k for k, v in out["features"].items() if not torch.isfinite(v).all()]
        print(f"  forward {img.name}: {len(slicers)} tiles {tuple(tiles.shape)} max|diff| {d_t:.3g}; crop "
              f"{tuple(ref_crop.shape)} max|diff| {d_c:.3g}; canonical={out['canonical']}; non-finite: {nonfinite or None}")
        assert d_t == 0 and d_c == 0 and not nonfinite


def main(part: str, configs: list[str]) -> None:
    if part == "gpu":
        torch.use_deterministic_algorithms(True)
    for c in configs:
        cfg = load_config(c)
        print(f"== {Path(c).name}", flush=True)
        w = build(cfg, "cuda" if part == "gpu" else "cpu")
        if part == "gpu":
            torch.backends.cudnn.benchmark = False  # nnUNetPredictor.__init__ turns it on
            check_gpu(w)
        else:
            check_cpu(w)
        del w
        torch.cuda.empty_cache()
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
