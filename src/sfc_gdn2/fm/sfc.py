"""Our encoder (LeJEPA, causal GDN-2) as a benchmark model: the same `<out>/<id>.pt` as the external FMs, so
bench-extract / bench-probe / bench_seg treat it like any of them. Runs in our own env (sfc-gdn2), `repo` = this repo.

- preprocessing: the pretraining one (data/volume.py): canonical orientation (axis permutation / flips only), native
  voxels, values / 99th percentile of the voxels > 0, patches of round(patch_mm / spacing) voxels per axis.
- features: probe.encoder_extractor, i.e. the probes' own read: the whole patch grid along each curve of
  `args.curves` (default: the run's view_curves, averaged), forward + reversed. `map` = bi tokens on the patch grid
  [2d, gx, gy, gz]; `mean` = bi averaged over the foreground patches (the pyramid top, what volume probes read:
  canonical).
- meta: `source_to_input` (source voxel index -> canonical voxel index) and `strides` (`map`: the scan's patch k,
  voxels per axis), read by bench_geom / bench_seg.

`checkpoint`: an `encoder_step*.pt` (its run's config.yaml next to it), a run directory (its last checkpoint, or
`args.step`), or an output_root holding exactly one run.
"""
from __future__ import annotations

from pathlib import Path

import nibabel.orientations as nio
import numpy as np
import torch

from .base import Image, Wrapper


def checkpoint_path(path: str | Path, step: int | None = None) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not (p / "config.yaml").exists():
        runs = sorted(p.glob("lejepa-*"))
        if len(runs) != 1:
            raise FileNotFoundError(f"sfc: {p} holds {len(runs)} runs; point `checkpoint` at one")
        p = runs[0]
    if step is not None:
        return p / f"encoder_step{int(step):06d}.pt"
    ckpts = sorted(p.glob("encoder_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"sfc: no encoder_step*.pt in {p}")
    return ckpts[-1]


class SFC(Wrapper):
    """args: curves (list, default the run's view_curves), step (int, default the last checkpoint)."""

    name = "sfc_gdn2"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        from ..io import load_yaml

        self.ckpt = checkpoint_path(cfg["checkpoint"], self.args.get("step"))
        self.run_cfg = load_yaml(self.ckpt.parent / "config.yaml")
        self.curves = list(self.args.get("curves") or self.run_cfg["objective"]["view_curves"])
        self._extract = None  # built on first use: preprocessing workers never load GDN-2

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("sfc: single-modality, pass a NIfTI path")
        from ..data.volume import VolumeStore, open_image, patch_dims, patch_grid

        img = open_image(image)
        arr, spacing, (a, c) = VolumeStore.decode(img)
        k = patch_dims(spacing, self.run_cfg["data"]["patch_mm"])
        ornt = nio.io_orientation(img.affine)
        src = np.linalg.inv(nio.inv_ornt_aff(ornt, img.shape[:3]))  # source voxel -> canonical voxel
        meta = {"source_affine": torch.from_numpy(img.affine.copy()), "source_shape": tuple(img.shape[:3]),
                "source_to_input": torch.from_numpy(src), "input_shape": tuple(arr.shape),
                "spacing": tuple(spacing), "strides": {"map": tuple(int(v) for v in k)}}
        return {"volume": torch.from_numpy(np.ascontiguousarray(arr)), "value": (a, c), "k": torch.tensor(k),
                "grid": patch_grid(arr.shape, k), "spacing": torch.tensor(spacing, dtype=torch.float32), "meta": meta}

    def extractor(self):
        if self._extract is None:
            from ..model import Encoder
            from ..probe import encoder_extractor

            enc = Encoder.from_config(self.run_cfg)
            enc.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
            enc = enc.to(self.device).eval().requires_grad_(False)
            obj = self.run_cfg["objective"]
            self._extract = encoder_extractor(enc, self.curves, "both", int(self.run_cfg["seed"]),
                                              obj.get("fg_threshold", 0.05))
        return self._extract

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        from ..data.volume import patchify

        dev = torch.device(self.device)
        a, c = prepared["value"]
        k, grid = prepared["k"], prepared["grid"]
        x = patchify(prepared["volume"].to(dev).float() * a + c, k.tolist())
        out = self.extractor()({"patches": [x], "grid": grid[None].to(dev), "k": k[None].to(dev),
                                "spacing": prepared["spacing"][None].to(dev)})
        bi = out["bi"][0, : len(x)]
        feats = {"mean": out["mean"][0], "map": bi.T.reshape(bi.shape[1], *grid.tolist())}
        return {"features": feats, "canonical": "mean", "derived": [],
                "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}}

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__}


def build(cfg: dict, device: str) -> SFC:
    return SFC(cfg, device)
