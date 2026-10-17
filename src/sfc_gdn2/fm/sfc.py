"""Our encoder (LeJEPA, causal GDN-2) as a benchmark model: the same `<out>/<id>.pt` as the external FMs, so
bench-extract / bench-probe / bench-segdec treat it like any of them. Runs in our own env (sfc-gdn2), `repo` = this repo.

- preprocessing: the pretraining one (data/volume.py): canonical orientation (axis permutation / flips only), native
  voxels, values / 99th percentile of the voxels > 0, patches of round(patch_mm / spacing) voxels per axis.
- features: model.encoder_extractor (the read of every evaluation): the whole patch grid along each curve of
  `args.curves` (default: the run's view_curves, averaged), forward + reversed. `map` = bi tokens on the patch grid
  [2d, gx, gy, gz]; `mean` = bi averaged over the foreground patches (the pyramid top, what volume probes read:
  canonical).
- meta: `source_to_input` (source voxel index -> canonical voxel index), returned by `seg_input`.
- segmentation (bench-segdec, fm/segrun.py): the frozen encoder + Primus' decoder (`PatchDecode`, the decoder of the
  OpenMind / nnFoundation token models: transposed convs, LayerNorm, GELU, He init 1e-2) from each token to a 16^3
  grid over its patch (1 mm at patch_mm 16), resized trilinearly to the patch's native voxels (k differs per scan).
  Input: the native canonical grid (`seg_input`), the spacing as `seg_context`; tiles of SEG_PATCH voxels.

`checkpoint`: an `encoder_step*.pt` (its run's config.yaml next to it), a run directory (its last checkpoint, or
`args.step`), or an output_root holding exactly one run.
"""
from __future__ import annotations

from pathlib import Path

import nibabel.orientations as nio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import Image, Wrapper

SEG_PATCH = (160, 160, 160)   # voxels, as the Primus / nnFoundation ViT runs
DECODE = 16                   # PatchDecode cells per patch edge (a power of 2)


class SegNet(nn.Module):
    """`encoder` (frozen: segrun) read along `curves` as in extraction -> bi tokens -> `decode` per token."""

    def __init__(self, encoder, curves: list[str], seed: int, patch_mm: float, n_out: int, embed_only: bool = False,
                 context: bool = False, tta: bool = False, neighbors: bool = False, kernel: bool = False,
                 refine: bool = False, image: bool = False):
        super().__init__()
        from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd, PatchDecode
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He

        from ..model import encoder_extractor

        self.encoder, self.patch_mm, self.context = encoder, patch_mm, context
        self.unique = encoder.bidir != "late"  # one stream: bi = h ++ h, the decoder takes h once
        width = encoder.dim * (1 + len(encoder.taps)) * (1 if self.unique else 2) * (2 if context else 1)
        # more information, not a bigger decoder (user, 16/10): Primus' PatchDecode scales its channels with its input
        # (2.1M parameters at 1024, ~400M at 20480), so a wider input is first projected to d per token
        self.proj = nn.Linear(width, encoder.dim) if width > encoder.dim else nn.Identity()
        # neighbors: each token also sees its 26 neighbours on the patch grid (a ConvNeXt-style residual block, ~1M
        # parameters, starting as the identity); the decoder itself stays per token
        c, self.mix = min(width, encoder.dim), None
        if neighbors:
            self.mix = nn.Sequential(nn.Conv3d(c, c, 3, padding=1, groups=c), LayerNormNd(c), nn.Conv3d(c, c, 1), nn.GELU())
            nn.init.zeros_(self.mix[2].weight), nn.init.zeros_(self.mix[2].bias)
        self.decode = PatchDecode((DECODE,) * 3, min(width, encoder.dim), n_out, norm=LayerNormNd, activation=nn.GELU)
        self.decode.apply(InitWeights_He(1e-2))
        # kernel: the kernel embedding run backwards: each token gives the coefficients of a function in the basis
        # phi_x phi_y phi_z of a KernelEmbed, integrated over every native voxel (no 16^3 grid, nothing resampled).
        # refine: first 16 -> 8 mm along the Hilbert curve (a cell's 8 children are consecutive on it) by a 2-block
        # GDN-2 Encoder at 8 mm, decoded in its own KernelEmbed's basis; image: + that KernelEmbed of the voxels.
        self.kernel, self.refine, self.image, self.seed = kernel or refine, refine, image, seed
        if refine:
            from ..model import Encoder
            self.fine = Encoder(128, 2, 64, 2, bidir="alternate", patch_mm=patch_mm / 2, freq_decay=1.0)
            self.up, c = nn.Linear(c, 128), 128
        if self.kernel:
            self.coef = nn.Sequential(nn.Linear(c, 64), nn.GELU(), nn.Linear(64, n_out * encoder.patch_embed.rank ** 3))
            del self.decode
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0, embed_only=embed_only, reverse=tta)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        """x [B, 1, *T] native voxels, spacing [B, 3] mm -> logits [B, n_out, *T]."""
        from ..data.volume import patch_dims, patch_grid, patchify

        T = tuple(x.shape[2:])
        ks = [patch_dims(s, self.patch_mm) for s in spacing.tolist()]
        grid = torch.stack([patch_grid(T, k) for k in ks]).to(x.device)
        with torch.no_grad():
            bi = self.extract({"patches": [patchify(x[b, 0].float(), k) for b, k in enumerate(ks)], "grid": grid,
                               "k": torch.tensor(ks, device=x.device), "spacing": spacing.float()})["bi"]
        out = []
        for b, k in enumerate(ks):
            g = grid[b].tolist()
            tok = bi[b, : int(grid[b].prod())]
            if self.unique:
                tok = tok[:, : tok.shape[1] // 2]
            if self.context:  # the tile's mean token beside each token: the decoder can centre it (token - mean)
                tok = torch.cat([tok, tok.mean(0, keepdim=True).expand_as(tok)], -1)
            z = self.proj(tok).T.reshape(1, -1, *g)
            z = z if self.mix is None else z + self.mix(z)
            if self.kernel:
                emb, z = self.encoder.patch_embed, z[0].flatten(1).T                      # [N, c]
                if self.refine:
                    z, k, g = self.to_fine(z, x[b, 0], spacing[b], k, g)
                    emb = self.fine.patch_embed
                out.append(self.kernel_decode(z, emb, spacing[b], k, g)[:, : T[0], : T[1], : T[2]])
                continue
            m = self.decode(z)                                                            # [1, n_out, 16 g]
            m = F.interpolate(m, size=[a * c for a, c in zip(k, g)], mode="trilinear", align_corners=False)
            out.append(m[0, :, : T[0], : T[1], : T[2]])
        return torch.stack(out)

    def to_fine(self, z, vol, sp, k, g):
        """Tokens [N, c] of the 16 mm grid g -> [Nf, 128] of the 8 mm grid (parent token + the voxels' embedding),
        mixed along its Hilbert curve; returns them with the fine patch dims and grid."""
        from ..curves import grid_coords, keys
        from ..data.volume import patch_dims, patch_grid, patchify

        kf = patch_dims(sp.tolist(), self.patch_mm / 2)
        gf = patch_grid(vol.shape, kf).to(z.device)
        xyz = grid_coords(gf[None], int(gf.prod()))[0]                                    # [Nf, 3] raster
        par = ((xyz + 0.5) * torch.tensor(kf, device=z.device) / torch.tensor(k, device=z.device)).long()
        par = torch.minimum(par, torch.tensor(g, device=z.device) - 1)
        t = self.up(z)[(par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]]
        if self.image:
            e = self.fine.patch_embed
            t = t + e(patchify(vol.float(), kf), e.weights(sp.float(), kf))
        order = keys("hilbert", xyz, gf, self.seed).argsort()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.fine.flat(t[order], torch.tensor([0, len(t)], device=z.device, dtype=torch.int32)).float()
        return h[order.argsort()], kf, gf.tolist()

    def kernel_decode(self, z, emb, sp, k, g):
        """Tokens [N, c] on grid g -> logits [n_out, *(g k)] at the native voxels: per token coefficients in the
        basis of `emb` (KernelEmbed), each voxel the mean of that function over its extent (same 1D integrals)."""
        # each basis function at RMS 1 over the patch's voxels (any spacing), the sum of r^3 terms over r^1.5: the
        # trained encoder's phi reach 1.3, unscaled every Adam step moved the logits by ~10 (no learning, 17/10)
        px, py, pz = (p / p.pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6) for p in emb.axis_integrals(sp, k))
        r = emb.rank
        m = torch.einsum("noabc,xa,yb,zc->noxyz", self.coef(z).view(len(z), -1, r, r, r), px, py, pz) / r ** 1.5
        o = m.shape[1]
        return m.reshape(*g, o, *k).permute(3, 0, 4, 1, 5, 2, 6).reshape(o, *(a * c for a, c in zip(g, k)))


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
    """args: curves (list, default the run's view_curves), step (int, default the last checkpoint), crop (read only
    the foreground's bounding box, as the training's global views), encoder: none (the patch embedding alone)."""

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
        from ..data.volume import decode, open_image, patch_dims, patch_grid

        img = open_image(image)
        arr, spacing, (a, c) = decode(img)
        k = patch_dims(spacing, self.run_cfg["data"]["patch_mm"])
        ornt = nio.io_orientation(img.affine)
        src = np.linalg.inv(nio.inv_ornt_aff(ornt, img.shape[:3]))  # source voxel -> canonical voxel
        meta = {"source_affine": torch.from_numpy(img.affine.copy()), "source_shape": tuple(img.shape[:3]),
                "source_to_input": torch.from_numpy(src), "input_shape": tuple(arr.shape),
                "spacing": tuple(spacing)}
        return {"volume": torch.from_numpy(np.ascontiguousarray(arr)), "value": (a, c), "k": torch.tensor(k),
                "grid": patch_grid(arr.shape, k), "spacing": torch.tensor(spacing, dtype=torch.float32), "meta": meta}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        a, c = prepared["value"]
        return (prepared["volume"].float() * a + c)[None], prepared["meta"]["source_to_input"].numpy()

    def seg_context(self, prepared: dict) -> torch.Tensor:
        return prepared["spacing"]

    def _encoder(self, pretrained: bool = True):
        from ..model import Encoder

        enc = Encoder.from_config(self.run_cfg)
        if pretrained:
            enc.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
        return enc

    def seg_net(self, n_out: int, pretrained: bool = True):
        enc = self._encoder(pretrained)
        taps = self.args.get("taps") or ()  # also these blocks' outputs (layer-normed) to the decoder; an int k: the k
        if isinstance(taps, int):           # quarter points below the last block (2: n/2, 3n/4; 3: also n/4; 4: also
            taps = [len(enc.blocks) * i // 4 - 1 for i in range(4 - taps, 4)]  # the patch embedding, block -1)
        enc.taps = tuple(taps)
        net = SegNet(enc, self.curves, int(self.run_cfg["seed"]), self.run_cfg["data"]["patch_mm"], n_out,
                     embed_only=self.args.get("encoder") == "none", context=bool(self.args.get("context")),
                     tta=bool(self.args.get("tta")), neighbors=bool(self.args.get("neighbors")),
                     kernel=bool(self.args.get("kernel")), refine=bool(self.args.get("refine")),
                     image=bool(self.args.get("image")))
        return net, ["encoder"], SEG_PATCH

    def extractor(self):
        if self._extract is None:
            from ..model import encoder_extractor

            enc = self._encoder().to(self.device).eval().requires_grad_(False)
            if self.args.get("encoder") != "none":  # also blocks n/2 and 3n/4 (probe menu: mean_b<block>)
                n = len(enc.blocks)
                enc.taps = tuple(self.args.get("probe_taps") or (n // 2 - 1, 3 * n // 4 - 1))
            self._taps, self._dim = enc.taps, enc.dim
            obj = self.run_cfg["objective"]
            self._extract = encoder_extractor(enc, self.curves, "both", int(self.run_cfg["seed"]),
                                              obj.get("fg_threshold", 0.05), crop=bool(self.args.get("crop")),
                                              embed_only=self.args.get("encoder") == "none",
                                              reverse=bool(self.args.get("tta")))  # also the *_tta readings
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
        d, n = self._dim, len(self._taps) + 1

        def block(v, j):  # [..., 2dn] = fwd (last block, taps) ++ bwd (same) -> block j's fwd ++ bwd [..., 2d]
            return torch.cat([v[..., j * d: (j + 1) * d], v[..., (n + j) * d: (n + j + 1) * d]], -1)

        bi, g = block(out["bi"][0, : len(x)], 0), grid.tolist()
        feats = {"map": bi.T.reshape(bi.shape[1], *g), "fg": out["fg"][0, : len(x)].view(*g)}  # last block's tokens
        for stat in ("mean", "std", "mean_tta", "std_tta"):  # per block: last (no suffix), taps (_b<block>)
            if stat in out:
                feats |= {stat + ("" if j == 0 else f"_b{self._taps[j - 1] + 1}"): block(out[stat][0], j) for j in range(n)}
        return {"features": feats, "canonical": "mean", "derived": [],
                "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}}

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__}


build = SFC
