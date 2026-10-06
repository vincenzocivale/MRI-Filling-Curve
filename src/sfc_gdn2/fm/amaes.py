"""AMAES (asbjrnmunk/amaes) and the FOMO baseline (fomo25/baseline-codebase) MAE-pretrained encoders.

Both repos share one layout. Their encoders are small and the repos are not installable packages
(poetry projects that pin yucca==1.1.7, python 3.11, torch<2.3), so the ENCODERS are re-implemented
here in pure torch with state_dict keys identical to the originals, instead of vendoring or importing
repo code:

- U-Net (`unet_xl`: 64 base filters, `unet_b`: 32; `src/models/networks/unet.py::UNetEncoder`): five
  stages `in_conv`, `encoder_conv1..4` (MaxPool 2 between them), each two Conv3d-InstanceNorm-LeakyReLU.
- MedNeXt (`mednext_m3|l3`, the released ones; `src/models/networks/mednext.py::MedNeXtEncoder`, blocks from yucca,
  MIT, itself from MIC-DKFZ/MedNeXt): `stem`, `enc_block_i`, `down_i`, `bottleneck`.

Checkpoint format. The released files (`unet_xl_lw_dec_fullaug.pth`, `mednext_l3_lw_dec_fullaug.pth`, ...,
Zenodo 13604788) are consumed by `src/train.py:248-259` as `torch.load(path)` -> a bare state_dict of the
Lightning module, i.e. keys `model.encoder.*` (+ `model.rec_head.*`, the light-weight reconstruction
decoder), optionally `_orig_mod.` (torch.compile) and, for a raw Lightning `.ckpt`, nested under
`state_dict` (FOMO baseline `src/utils/utils.py::load_pretrained_weights`, same layout). The finetune
code transfers by key name, so the `model.` prefix is what matters; we accept it, `model._orig_mod.`
or no prefix, keep `encoder.*` and ignore every head.

Intensity convention: AMAES pretrains on volumes scaled to [0, 1] (`self_supervised.py` asserts it),
which is what our cube already is (`normalize: unit`). The FOMO baseline z-normalises per volume
(`normalize: zscore_fg`: foreground z-score, background 0). Pre-training patch sizes: AMAES 128^3
(`train.py --patch_size` default), FOMO 96^3 (README).

`fomo26` = the AMAES-pretrained FOMO checkpoints (FOMO60K/FOMO300K, README "New 2026") in the FOMO
baseline convention. The repo does not document a separate FOMO26 architecture, so it is built from the
same encoders; pick `arch` to match the checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .base import FoundationEncoder, load_checked, pick, read_checkpoint, strip_prefix


# ------------------------------------------------------------------------------------------ U-Net
class _ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.norm = nn.InstanceNorm3d(cout, eps=1e-5, affine=True, momentum=0.1)
        self.activation = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class _Block(nn.Module):
    """`MultiLayerConvDropoutNormNonlin(num_layers=2)`; dropout p=0 so no module is created."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1, self.conv2 = _ConvNormAct(cin, cout), _ConvNormAct(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, filters: int = 64):
        super().__init__()
        f = filters
        self.in_conv = _Block(in_ch, f)
        self.encoder_conv1, self.encoder_conv2 = _Block(f, 2 * f), _Block(2 * f, 4 * f)
        self.encoder_conv3, self.encoder_conv4 = _Block(4 * f, 8 * f), _Block(8 * f, 16 * f)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        xs = [self.in_conv(x)]
        for blk in (self.encoder_conv1, self.encoder_conv2, self.encoder_conv3, self.encoder_conv4):
            xs.append(blk(self.pool(xs[-1])))
        return xs


# ----------------------------------------------------------------------------------------- MedNeXt
class _MedNeXtBlock(nn.Module):
    """yucca MedNeXtBlock (group norm, no GRN): depthwise k^3 -> GroupNorm -> 1x1 expand -> GELU -> 1x1."""

    def __init__(self, cin: int, cout: int, exp_r: int, k: int, do_res: bool = True):
        super().__init__()
        self.do_res = do_res
        self.conv1 = nn.Conv3d(cin, cin, k, 1, k // 2, groups=cin)
        self.norm = nn.GroupNorm(num_groups=cin, num_channels=cin)
        self.conv2 = nn.Conv3d(cin, exp_r * cin, 1)
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(exp_r * cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(self.act(self.conv2(self.norm(self.conv1(x)))))
        return x + y if self.do_res else y


class _MedNeXtDown(_MedNeXtBlock):
    def __init__(self, cin: int, cout: int, exp_r: int, k: int, do_res: bool = True):
        super().__init__(cin, cout, exp_r, k, do_res=False)
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv3d(cin, cout, kernel_size=1, stride=2)
        self.conv1 = nn.Conv3d(cin, cin, k, 2, k // 2, groups=cin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y + self.res_conv(x) if self.resample_do_res else y


class MedNeXtEncoder(nn.Module):
    def __init__(self, in_ch: int, filters: int, exp_r: list[int], blocks: list[int], k: int):
        super().__init__()
        f = filters
        self.stem = nn.Conv3d(in_ch, f, kernel_size=1)
        mult = [1, 2, 4, 8, 16]
        for i in range(4):
            setattr(self, f"enc_block_{i}", nn.Sequential(
                *[_MedNeXtBlock(f * mult[i], f * mult[i], exp_r[i], k) for _ in range(blocks[i])]))
            setattr(self, f"down_{i}", _MedNeXtDown(f * mult[i], f * mult[i + 1], exp_r[i + 1], k))
        self.bottleneck = nn.Sequential(*[_MedNeXtBlock(f * 16, f * 16, exp_r[4], k) for _ in range(blocks[4])])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, xs = self.stem(x), []
        for i in range(4):
            xs.append(getattr(self, f"enc_block_{i}")(x))
            x = getattr(self, f"down_{i}")(xs[-1])
        return [*xs, self.bottleneck(x)]


# `networks/unet.py` / `networks/mednext.py` presets (the `_lw_dec` / `_std_dec` variants share the encoder)
ARCHS = {
    "unet_xl": lambda c: UNetEncoder(c, 64),
    "unet_b": lambda c: UNetEncoder(c, 32),
    "mednext_m3": lambda c: MedNeXtEncoder(c, 32, [2, 3, 4, 4, 4], [3, 4, 4, 4, 4], 3),
    "mednext_l3": lambda c: MedNeXtEncoder(c, 32, [3, 4, 8, 8, 8], [3, 4, 8, 8, 8], 3),
}


class AmaesEncoder(FoundationEncoder):
    def __init__(self, name: str, arch: str, input_size: int, normalize: str, stage: int, in_channels: int = 1):
        super().__init__()
        base = arch.removesuffix("_lw_dec").removesuffix("_std_dec")
        if base not in ARCHS:
            raise ValueError(f"Unknown AMAES architecture {arch!r}; available: {sorted(ARCHS)} "
                             f"(+ _lw_dec/_std_dec suffixes).")
        if normalize not in ("unit", "zscore_fg"):
            raise ValueError("normalize must be 'unit' or 'zscore_fg'.")
        self.name, self.arch, self.input_size, self.norm_mode, self.stage = name, base, input_size, normalize, stage
        self.in_channels = in_channels
        self.encoder = ARCHS[base](in_channels)
        self.eval().requires_grad_(False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.norm_mode == "unit" else super().normalize(x)

    def features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"map": self.encoder(x)[self.stage]}

    def load_checkpoint(self, path: str | Path) -> dict:
        sd = pick(read_checkpoint(path), ("state_dict",))       # raw Lightning .ckpt or bare state_dict
        sd = strip_prefix({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, "module.")  # compile / DDP
        prefix = "model.encoder." if any(k.startswith("model.encoder.") for k in sd) else "encoder."
        enc = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not enc:
            raise RuntimeError(f"{path}: no `model.encoder.*` / `encoder.*` keys (got e.g. {list(sd)[:4]}).")
        # the encoder module here is `self.encoder`, so its keys are already relative to it
        info = load_checked(self.encoder, enc, what=f"{self.name} ({self.arch}) checkpoint {path}")
        return {**info, "ignored": [k for k in sd if not k.startswith(prefix)], "arch": self.arch}


def build(name: str, a: dict) -> AmaesEncoder:
    """model_args: arch (unet_xl | unet_b | mednext_{m3,l3}, `_lw_dec` suffix allowed), input_size,
    normalize (unit | zscore_fg), stage (-1 = deepest; 0..4), in_channels (1)."""
    amaes = name == "amaes"
    return AmaesEncoder(name, a.get("arch", "unet_xl" if amaes else "unet_b"),
                        int(a.get("input_size", 128 if amaes else 96)),
                        a.get("normalize", "unit" if amaes else "zscore_fg"), int(a.get("stage", -1)),
                        int(a.get("in_channels", 1)))
