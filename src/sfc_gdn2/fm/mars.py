"""MARS (zqiuak/MARS, Nature Biomed. Eng. 2026; arXiv 2508.07165 as PRISM): MONAI Swin-ViT (SwinUNETR v2 encoder).

Released checkpoint (README Google Drive link, the only one): Swin-B, feature_size 48, depths 2/2/2/2, heads
3/6/12/24, `use_v2=True` (the residual conv block after every stage, keys `layers{1..4}c`); a flat state dict of
134 keys, all `swinViT.*` (18.6M trainable parameters). The paper picks Swin-B on the 336k pool as its main model;
Swin-L / Swin-H were scaling runs only and are not released.

Preprocessing, from the repo's own downstream code (`main/dataset/finetune/ds.py`, the shipped ACDC task):
LoadImaged(NibabelReader, channel first) -> Orientationd(RAS) -> NormalizeIntensityd(nonzero, channel_wise) ->
CropForegroundd(source image, k_divisible=roi) -> SpatialPadd(roi). The ACDC-specific Transposed and its
32x192x192 input size are dropped (a cardiac short-axis layout, not a property of the model).
FLAG (our choice): Spacingd to 1.0 mm isotropic (bilinear) right after Orientationd. The pretraining cache was
resampled to 1.0 mm (paper Methods 4.1; `dataset/datautils.py` read_dicom_series / read_itk_img, kargs.Spacing);
the shipped downstream task reads native spacing and the paper does not state the downstream spacing, so the
encoder is fed the spacing it was pretrained on.

Network / features: `SwinUNETR(use_v2=True).swinViT`, loaded strict. `swinViT(x, normalize=True)` hidden states
`stage0..stage4` (48/96/192/384/768 ch, strides 2/4/8/16/32); canonical `stage4` (what the SwinUNETR decoder's
`encoder10` reads). Sliding window roi 96, overlap 0.75 and sw_batch_size 2 as the repo's config (`sw_overlap`,
`sw_batch_size`); constant blending. `stage4_gap` and the foreground means are ours (`derived`).
Segmentation: the repo's downstream network, MONAI SwinUNETR (`model/model_ft.py`), swinViT from the checkpoint,
the conv encoders/decoders from scratch; patch = roi 96^3 (`run/options.py` roi_x/y/z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import Image, Wrapper, add_to_path, fg_means, plain

STRIDES = (2, 4, 8, 16, 32)


class MARS(Wrapper):
    """args: roi (96), overlap (0.75), sw_batch_size (2), spacing (1.0 mm; null = native, as the ACDC example),
    feature_size (48), depths ([2, 2, 2, 2]), num_heads ([3, 6, 12, 24])."""

    name = "mars"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        a = self.args
        if cfg.get("repo"):
            add_to_path(Path(cfg["repo"]) / "main")
        self.roi = int(a.get("roi", 96))
        self.overlap = float(a.get("overlap", 0.75))
        self.sw_batch_size = int(a.get("sw_batch_size", 2))
        sp = a.get("spacing", 1.0)
        self.spacing = None if sp is None else (float(sp),) * 3
        self.transform = self._build_transform()
        self.swin = self._build_model()
        self.swin.eval().to(device)

    # ---------------------------------------------------------------- preprocessing
    def _build_transform(self):
        from monai import transforms as T
        steps = [T.LoadImaged(keys=["image"], reader="NibabelReader", ensure_channel_first=True),
                 T.Orientationd(keys=["image"], axcodes="RAS")]
        if self.spacing is not None:   # FLAG: the pretraining spacing (module docstring)
            steps.append(T.Spacingd(keys=["image"], pixdim=self.spacing, mode="bilinear"))
        steps += [T.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                  T.CropForegroundd(keys=["image"], source_key="image", k_divisible=self.roi),
                  T.SpatialPadd(keys=["image"], spatial_size=(self.roi,) * 3)]
        return T.Compose(steps)

    def preprocess(self, image: Image) -> dict:
        d = self.transform({"image": image})
        return {"image": d["image"], "path": image,
                "crop_start": d.get("foreground_start_coord"), "crop_end": d.get("foreground_end_coord")}

    # ---------------------------------------------------------------- network
    def _swinunetr(self, n_out: int):
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(img_size=(self.roi,) * 3, in_channels=1, out_channels=n_out,
                         feature_size=int(self.args.get("feature_size", 48)),
                         depths=tuple(self.args.get("depths", [2, 2, 2, 2])),
                         num_heads=tuple(self.args.get("num_heads", [3, 6, 12, 24])),
                         use_v2=True, use_checkpoint=False)

    def _build_model(self) -> nn.Module:
        ckpt = torch.load(self.cfg["checkpoint"], map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        rest = [k for k in sd if not k.startswith("swinViT.")]
        if rest:
            raise RuntimeError(f"mars: unexpected checkpoint keys {rest[:4]}")
        swin = self._swinunetr(1).swinViT
        swin.load_state_dict({k.removeprefix("swinViT."): v for k, v in sd.items()}, strict=True)
        return swin

    # ---------------------------------------------------------------- segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        """The pipeline's network input (RAS, 1 mm, z-scored, foreground crop) and input-voxel <- source-voxel map."""
        x = prepared["image"]
        m = np.linalg.inv(x.affine.double().numpy()) @ np.asarray(x.meta["original_affine"], dtype=np.float64)
        return x.as_tensor().float(), m

    def seg_net(self, n_out: int, pretrained: bool = True):
        """The repo's downstream network (SwinUNETR v2): swinViT from the checkpoint, the rest from scratch;
        patch = the roi (96^3)."""
        net = self._swinunetr(n_out)
        if pretrained:
            net.swinViT.load_state_dict(self.swin.state_dict(), strict=True)
        # val / test with the repo's own sliding window (finetune/socks.py infer_model: MONAI sliding_window_inference,
        # overlap = config sw_overlap 0.75, MONAI's default constant blending); fm/segrun.py `sliding` reads this
        net.monai_slide = {"overlap": float(self.args.get("seg_overlap", 0.75)), "mode": "constant"}
        return net, ["swinViT"], (self.roi,) * 3

    # ---------------------------------------------------------------- inference
    def _predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {f"stage{i}": h for i, h in enumerate(self.swin(x.contiguous(), normalize=True))}

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        from monai.inferers import sliding_window_inference
        x = prepared["image"]
        inputs = torch.stack([plain(x)]).to(self.device)
        out = sliding_window_inference(inputs, roi_size=[self.roi] * 3, sw_batch_size=self.sw_batch_size,
                                       predictor=self._predict, overlap=self.overlap)
        feats = {k: v[0].float() for k, v in out.items()}
        feats["stage4_gap"] = feats["stage4"].mean((1, 2, 3))
        fg = fg_means({k: feats[k] for k in ("stage2", "stage3", "stage4")}, inputs[0, 0] != 0)  # z-score keeps 0 = bg
        feats |= fg
        meta = {"input_path": prepared["path"], "spacing": self.spacing,
                "inference": {"mode": "sliding_window_inference", "roi": (self.roi,) * 3, "overlap": self.overlap,
                              "sw_batch_size": self.sw_batch_size, "blend": "constant"},
                "input_shape": tuple(inputs.shape[2:]), "strides": {f"stage{i}": s for i, s in enumerate(STRIDES)},
                "input_affine": x.affine.to(torch.float64),
                "source_affine": torch.as_tensor(np.asarray(x.meta["original_affine"]), dtype=torch.float64),
                "source_shape": tuple(int(s) for s in x.meta["spatial_shape"])}
        if prepared.get("crop_start") is not None:
            meta |= {"crop_start": tuple(int(c) for c in prepared["crop_start"]),
                     "crop_end": tuple(int(c) for c in prepared["crop_end"])}
        return {"features": feats, "canonical": "stage4", "meta": meta, "derived": ["stage4_gap", *fg]}


build = MARS
