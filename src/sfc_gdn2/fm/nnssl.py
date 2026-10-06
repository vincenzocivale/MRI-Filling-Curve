"""OpenMind and nnFoundation checkpoints (MIC-DKFZ/nnssl, branch nnFoundation).

Both are nnssl self-supervised pre-trainings, so they share one on-disk format
(`nnssl/training/nnsslTrainer/AbstractTrainer.py::save_checkpoint`, `utilities/add_adaptation_plan.py`):

    checkpoint_final.pth = {"network_weights": state_dict of the whole pre-training network,
                            "nnssl_adaptation_plan": serialized `AdaptationPlan`, "trainer_name", ...}

`network_weights` holds encoder *and* decoder. We keep only the encoder:
- CNN (ResEnc-L, `ResidualEncoderUNet`): keys `encoder.stem.*`, `encoder.stages.*`
  (plan: key_to_stem="encoder.stem", key_to_encoder="encoder.stages");
- ViT (Primus / EvaMAE): keys `down_projection.*` and `eva.*`
  (plan: key_to_stem="down_projection", key_to_encoder="eva"); `up_projection`, `decoder`,
  `mask_token` are the MAE decoder and are ignored.

The architecture is rebuilt from the checkpoint's own `nnssl_adaptation_plan` when present (so any
OpenMind / nnFoundation release loads without further configuration); `model_args` only
decides the architecture of the random-init baseline and of plan-less checkpoints.

Needs `dynamic-network-architectures>=0.4.4,<0.5` (Apache-2.0) and its deps (timm, einops); the
`nnssl` package itself is NOT needed (it requires python>=3.12) and its code is not vendored.
Weights: OpenMind `AnonRes/ResEncL-OpenMind-MAE` etc. and `MIC-DKFZ/nnFoundationCNN|ViT` on
Hugging Face (nnssl is CC-BY-SA-4.0; the checkpoints carry their own model cards).

Features: deepest encoder stage (`stage: -1`; stride 32 for ResEnc-L -> 6^3 at 192^3, 5^3 at 160^3)
or, for Primus, the token grid (stride 8). Our isotropic cube is resampled to `input_size`.
"""
from __future__ import annotations

import json
import pydoc
from pathlib import Path

import torch
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, require, strip_prefix

# nnssl/architectures/architecture_registry.py::get_res_enc_l (the "ResEncL" preset)
_RESENC_L = {
    "n_stages": 6, "features_per_stage": [32, 64, 128, 256, 320, 320],
    "kernel_sizes": [[3, 3, 3]] * 6, "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6], "n_conv_per_stage_decoder": [1] * 5, "conv_bias": True,
    "norm_op_kwargs": {"eps": 1e-5, "affine": True}, "nonlin_kwargs": {"inplace": True},
}
# dynamic_network_architectures.architectures.primus._PRIMUS_CONFIGS (+ nnFoundationViT = "X")
_PRIMUS = {"S": (396, 12, 6), "B": (792, 12, 12), "M": (864, 16, 12), "L": (1056, 24, 16),
           "X": (1056, 40, 16)}  # (embed_dim, depth, heads); X: nnFoundationViT trainer
_RESENC_NAMES = ("ResEncL", "NoSkipResEncL", "ResidualEncoderUNet")


def _resenc_encoder(kw: dict, in_ch: int) -> nn.Module:
    unet = require("dynamic_network_architectures.architectures.unet", "'dynamic-network-architectures>=0.4.4,<0.5'",
                   "nnssl ResEnc checkpoints")
    kw = dict(kw)
    kw.setdefault("conv_op", nn.Conv3d)
    kw.setdefault("norm_op", nn.InstanceNorm3d)
    kw.setdefault("nonlin", nn.LeakyReLU)
    for k in ("conv_op", "norm_op", "nonlin", "dropout_op"):  # plans store these as dotted names
        if isinstance(kw.get(k), str):
            kw[k] = pydoc.locate(kw[k])
    return unet.ResidualEncoderUNet(input_channels=in_ch, num_classes=1, deep_supervision=False, **kw).encoder


class PrimusEncoder(nn.Module):
    """`down_projection` + `eva` of Primus / nnssl EvaMAE with identical state_dict keys, no decoder."""

    def __init__(self, in_ch: int, embed_dim: int, depth: int, heads: int, patch_embed_size, input_shape,
                 init_values=0.1, scale_attn_inner=True):
        super().__init__()
        require("timm", "'timm<1.0.23'", "Primus / EVA")
        from dynamic_network_architectures.building_blocks.eva import Eva
        from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed
        self.down_projection = PatchEmbed(tuple(patch_embed_size), in_ch, embed_dim)
        self.eva = Eva(embed_dim=embed_dim, depth=depth, num_heads=heads, init_values=init_values,
                       scale_attn_inner=scale_attn_inner,
                       ref_feat_shape=tuple(i // p for i, p in zip(input_shape, patch_embed_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_projection(x)                       # [B,C,w,h,d]
        b, c, *g = x.shape
        tokens, _ = self.eva(x.flatten(2).transpose(1, 2))  # no patch drop at inference -> keep_indices None
        return tokens.transpose(1, 2).reshape(b, c, *g)


def _make(spec: dict) -> nn.Module:
    if spec["kind"] == "resenc":
        return _resenc_encoder(spec["kwargs"], spec["in_ch"])
    return PrimusEncoder(spec["in_ch"], **spec["kwargs"])


def _spec_from_arch(arch: str, in_ch: int, input_size: int, arch_kwargs: dict | None) -> dict:
    if arch in ("ResEncL", "NoSkipResEncL"):  # NoSkip only changes the decoder
        return {"kind": "resenc", "in_ch": in_ch, "kwargs": _RESENC_L}
    if arch == "ResidualEncoderUNet":
        if not arch_kwargs:
            raise ValueError("arch ResidualEncoderUNet needs model_args.arch_kwargs (nnssl DynamicArchitecturePlans).")
        return {"kind": "resenc", "in_ch": in_ch, "kwargs": arch_kwargs}
    if arch.startswith("Primus"):
        scale = arch.removeprefix("Primus")
        if arch == "PrimusX" and arch_kwargs:  # EvaMAE-style dict from the nnFoundationViT trainer / plan
            a = arch_kwargs
            return {"kind": "primus", "in_ch": in_ch, "kwargs": {
                "embed_dim": a["embed_dim"], "depth": a["encoder_eva_depth"], "heads": a["encoder_eva_numheads"],
                "patch_embed_size": a["patch_embed_size"], "input_shape": a["input_shape"],
                "init_values": a.get("init_values"), "scale_attn_inner": a.get("scale_attn_inner", False)}}
        if scale not in _PRIMUS:
            raise ValueError(f"Unknown Primus scale {arch!r}; expected one of {sorted(_PRIMUS)}.")
        d, depth, heads = _PRIMUS[scale]
        return {"kind": "primus", "in_ch": in_ch, "kwargs": {
            "embed_dim": d, "depth": depth, "heads": heads, "patch_embed_size": (8, 8, 8),
            "input_shape": (input_size,) * 3}}
    raise ValueError(f"Unsupported nnssl architecture {arch!r}; supported: ResEncL, NoSkipResEncL, "
                     f"ResidualEncoderUNet, Primus[S|B|M|L|X].")


def spec_from_plan(plan: dict) -> tuple[dict, int | None]:
    """Serialized `nnssl_adaptation_plan` -> (encoder spec, pre-training patch side or None)."""
    ap = plan["architecture_plans"]
    cfgs = plan.get("pretrain_plan", {}).get("configurations", {})
    patch = next((c.get("patch_size") for c in cfgs.values() if c.get("patch_size")), None) \
        or plan.get("recommended_downstream_patchsize")
    side = int(patch[0]) if patch and len(set(patch)) == 1 else None
    return _spec_from_arch(ap["arch_class_name"], plan.get("pretrain_num_input_channels", 1), side or 160,
                           ap.get("arch_kwargs")), side


class NnsslEncoder(FoundationEncoder):
    """normalize: `zscore_all` (nnU-Net ZScoreNormalization over the whole volume, as in the nnssl plans
    with use_mask_for_norm=False; our zero padding is part of the volume) or `zscore_fg`."""

    def __init__(self, name: str, arch: str, input_size: int | None, default_size: int, normalize: str,
                 stage: int, arch_kwargs: dict | None, use_plan: bool):
        super().__init__()
        self.name, self.stage, self.norm_mode, self.use_plan = name, stage, normalize, use_plan
        self.input_size, self._size_fixed = input_size or default_size, input_size is not None
        self.arch, self.arch_kwargs = arch, arch_kwargs
        self.spec = _spec_from_arch(arch, 1, self.input_size, arch_kwargs)
        self.net = _make(self.spec)
        self.eval().requires_grad_(False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "zscore_fg":
            return super().normalize(x)
        m, s = x.mean((1, 2, 3, 4), keepdim=True), x.std((1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
        return (x - m) / s

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(x)
        return {"map": out[self.stage] if isinstance(out, list) else out}

    def load_checkpoint(self, path: str | Path) -> dict:
        ckpt = read_checkpoint(path)
        sd = pick(ckpt, ("network_weights", "state_dict"))
        for _ in range(2):  # DDP `module.`, torch.compile `_orig_mod.`
            sd = strip_prefix(sd, "module.", "_orig_mod.")
        plan = ckpt.get("nnssl_adaptation_plan") if isinstance(ckpt, dict) else None
        if plan and self.use_plan:
            spec, side = spec_from_plan(plan)
            if spec["in_ch"] != 1:
                raise NotImplementedError(f"Checkpoint was pre-trained on {spec['in_ch']} channels.")
            if json.dumps(spec, sort_keys=True, default=str) != json.dumps(self.spec, sort_keys=True, default=str):
                device = next(self.net.parameters()).device
                self.spec, self.net = spec, _make(spec).to(device).eval().requires_grad_(False)
            if side and not self._size_fixed:
                self.input_size = side
        root = ("encoder.",) if self.spec["kind"] == "resenc" else ("down_projection.", "eva.")
        enc = {k.removeprefix("encoder.") if root == ("encoder.",) else k: v
               for k, v in sd.items() if k.startswith(root)}
        if not enc:
            raise RuntimeError(f"{path}: no encoder keys found (got e.g. {list(sd)[:4]}).")
        dropped = [k for k in sd if not k.startswith(root)]
        info = load_checked(self.net, enc, what=f"{self.name} checkpoint {path}")
        return {**info, "ignored": dropped, "arch": self.spec["kind"], "input_size": self.input_size,
                "trainer": ckpt.get("trainer_name") if isinstance(ckpt, dict) else None}


# nnFoundationViT_trainer: BaseEvaMAETrainer_BS96_192ps_2500ep_40_16_8_16_1056_lr2e3
_NNFOUNDATION_VIT = {"embed_dim": 1056, "encoder_eva_depth": 40, "encoder_eva_numheads": 16,
                     "patch_embed_size": (8, 8, 8), "input_shape": (192, 192, 192), "init_values": 0.1,
                     "scale_attn_inner": True}


def build(name: str, a: dict) -> NnsslEncoder:
    """model_args: arch (ResEncL | NoSkipResEncL | ResidualEncoderUNet | Primus{S,B,M,L,X}),
    variant (nnfoundation only: cnn | vit), input_size, normalize (zscore_all | zscore_fg), stage (-1),
    arch_kwargs, use_plan (default true: rebuild the architecture from the checkpoint's plan)."""
    if name == "nnfoundation":  # nnFoundationCNN = ResEnc-L, nnFoundationViT = Primus-X; both 192^3
        arch, size = a.get("arch", "PrimusX" if a.get("variant", "cnn") == "vit" else "ResEncL"), 192
        kw = a.get("arch_kwargs", _NNFOUNDATION_VIT if arch == "PrimusX" else None)
    else:                       # openmind: ResEnc-L (or Primus-M), pre-trained on 160^3 patches
        arch, size, kw = a.get("arch", "ResEncL"), 160, a.get("arch_kwargs")
    if arch == "PrimusX" and kw:
        kw = {**kw, "input_shape": (a.get("input_size") or size,) * 3}
    return NnsslEncoder(name, arch, a.get("input_size"), size, a.get("normalize", "zscore_all"),
                        int(a.get("stage", -1)), kw, bool(a.get("use_plan", True)))
