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
from torch.utils.checkpoint import checkpoint

from .base import Image, Wrapper

SEG_PATCH = (160, 160, 160)   # voxels, as the Primus / nnFoundation ViT runs
DECODE = 16                   # PatchDecode cells per patch edge (a power of 2)


class HeadErase(nn.Module):
    """A GDN-2 mixer's b_proj with one erase gate per head (KDA's), broadcast over the head's key channels. GDN-2's
    channel-wise erase makes the state transition I - k (b*k)^T non-normal: any direction off k is amplified, and on one
    64000-token sequence (decoder F's 4 mm level, a 1 mm stroke scan, 18/10) the state grew past fp32, the exact
    recurrence too. With b = beta * 1 and |k| = 1 (the kernel's l2norm) it is I - beta k k^T, eigenvalues in
    [1 - beta, 1]: the state cannot grow, whatever the weights or the length. The write gate stays per channel."""

    def __init__(self, mix: nn.Module):
        super().__init__()
        self.proj, self.dk = nn.Linear(mix.b_proj.in_features, mix.num_heads, bias=False), mix.head_k_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h).repeat_interleave(self.dk, -1)


class SegNet(nn.Module):
    """`encoder` (frozen: segrun) read along `curves` as in extraction -> bi tokens -> `decode` per token."""

    def __init__(self, encoder, curves: list[str], seed: int, patch_mm: float, n_out: int, embed_only: bool = False,
                 context: bool = False, tta: bool = False, neighbors: bool = False, kernel: bool = False,
                 refine: int = 0, image: bool = False, fine_curves: list | None = None, rough: bool = False,
                 zoom: bool = False, fine_seg: int = 0, erase: str = "channel",
                 fine_mm: list | None = None):
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
        # refine n: n halvings 16 -> 8 (-> 4) mm along the Hilbert curve (a cell's 8 children are consecutive on it),
        # each by a 2-block GDN-2 Encoder per curve of `fine_curves` (in turn), decoded in the last one's KernelEmbed basis;
        # image: + that KernelEmbed of the voxels at each scale (rough: without the 1/f gain, sharper edges).
        self.kernel, self.refine, self.image, self.seed = kernel or bool(refine), int(refine), image, seed
        self.fine_seg = int(fine_seg)   # 0: one sequence per tile; n: segments of n tokens along the curve
        # the refine levels' patch (mm): halvings of the encoder's patch, or fixed sizes (fine_mm, e.g. [8, 4] under
        # 12 mm tokens: a parent/child ratio of 1.5 per axis, children found by position as for anisotropic voxels)
        self.fine_mm = [float(m) for m in fine_mm] if fine_mm else [patch_mm / 2 ** i for i in range(1, int(refine) + 1)]
        self.fine_curves = list(fine_curves or ["hilbert"])
        if self.refine:
            from ..model import Encoder
            self.fine = nn.ModuleList(nn.ModuleList(Encoder(128, 2, 64, 2, bidir="alternate", patch_mm=self.fine_mm[i],
                                                            freq_decay=0.0 if rough else 1.0) for _ in self.fine_curves)
                                      for i in range(self.refine))
            if erase == "head":   # HeadErase: refine blocks whose state cannot diverge
                for m in self.fine.modules():
                    if hasattr(m, "b_proj") and hasattr(m, "head_k_dim"):
                        m.b_proj = HeadErase(m)
            self.up = nn.ModuleList([nn.Linear(c, 128)] + [nn.Linear(128, 128) for _ in range(self.refine - 1)])
            # zoom: the image at each scale through the encoder's own (frozen) patch kernel, zoomed onto the cell (its
            # voxels at 2^i x their spacing fill one 16 mm patch: exact, nothing resampled), projected to 128
            self.zoom = nn.ModuleList(nn.Linear(encoder.dim, 128) for _ in range(self.refine)) if zoom else None
            c = 128
        if self.kernel:
            self.coef = nn.Sequential(nn.Linear(c, 64), nn.GELU(), nn.Linear(64, n_out * encoder.patch_embed.rank ** 3))
            del self.decode
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0, embed_only=embed_only, reverse=tta)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        """x [B, 1, *T] native voxels, spacing [B, 3] mm, or [B, 6] = spacing + the scan's shape (segrun `seg_extent`)
        -> logits [B, n_out, *T]. Only the scan's own voxels are read: segrun right-pads a tile up to SEG_PATCH voxels,
        ~160 mm for resampled models but up to 1.2 m of zeros on our native grids (TotalSeg 64^3 at 7.5 mm: 5e5 tokens,
        GPU OOM; SOOP 25 slices of 6 mm: 85 % of the tile), never seen in pretraining. The padding (label 255,
        ignored) gets zero logits."""
        from ..data.volume import patch_dims, patch_grid, patchify

        T = tuple(x.shape[2:])
        ext = spacing[:, 3:].long().tolist() if spacing.shape[1] > 3 else [T] * len(x)
        spacing = spacing[:, :3]
        V = [tuple(min(a, t) for a, t in zip(e, T)) for e in ext]                           # the scan's part of each tile
        ks = [patch_dims(s, self.patch_mm) for s in spacing.tolist()]
        grid = [patch_grid(v, k).to(x.device) for v, k in zip(V, ks)]
        xs = [x[b, 0, : v[0], : v[1], : v[2]].float() for b, v in enumerate(V)]
        with torch.no_grad():   # one tile at a time
            bi = [self.extract({"patches": [patchify(xs[b], k)], "grid": grid[b][None],
                                "k": torch.tensor([k], device=x.device), "spacing": spacing[b:b + 1].float()})["bi"][0]
                  for b, k in enumerate(ks)]
        out = []
        for b, k in enumerate(ks):
            g, v = grid[b].tolist(), V[b]
            tok = bi[b][: int(grid[b].prod())]
            if self.unique:
                tok = tok[:, : tok.shape[1] // 2]
            if self.context:  # the tile's mean token beside each token: the decoder can centre it (token - mean)
                tok = torch.cat([tok, tok.mean(0, keepdim=True).expand_as(tok)], -1)
            z = self.proj(tok).T.reshape(1, -1, *g)
            z = z if self.mix is None else z + self.mix(z)
            if self.kernel:
                emb, z = self.encoder.patch_embed, z[0].flatten(1).T                      # [N, c]
                for i in range(self.refine):
                    z, k, g = self.to_fine(z, xs[b], spacing[b], k, g, i)
                    emb = self.fine[i][0].patch_embed
                out.append(self.kernel_decode(z, emb, spacing[b], k, g)[:, : v[0], : v[1], : v[2]])
                continue
            # in slabs along the first axis: with coarse voxels a tile holds 10^4-10^5 tokens and 16^3 cells each
            s = max(1, (1 << 25) // (DECODE ** 3 * g[1] * g[2] * max(self.decode.decode[-1].out_channels, 1)))

            def slab(zz, kk=k, gg=g):   # recomputed in backward
                return F.interpolate(self.decode(zz), size=[zz.shape[2] * kk[0], gg[1] * kk[1], gg[2] * kk[2]],
                                     mode="trilinear", align_corners=False)
            m = torch.cat([checkpoint(slab, z[:, :, i:i + s], use_reentrant=False) for i in range(0, g[0], s)], 2)
            out.append(m[0, :, : v[0], : v[1], : v[2]])
        return torch.stack([F.pad(o, [p for v, t in zip(reversed(o.shape[1:]), reversed(T)) for p in (0, t - v)])
                            for o in out])

    def to_fine(self, z, vol, sp, k, g, i):
        """Tokens [N, c] on grid g -> [Nf, 128] on the grid of half the patch (parent token + the voxels' embedding),
        mixed along each curve in turn; returns them with the fine patch dims and grid."""
        from ..curves import grid_coords, keys
        from ..data.volume import patch_dims, patch_grid, patchify

        kf = patch_dims(sp.tolist(), self.fine_mm[i])
        gf = patch_grid(vol.shape, kf).to(z.device)
        xyz = grid_coords(gf[None], int(gf.prod()))[0]                                    # [Nf, 3] raster
        par = ((xyz + 0.5) * torch.tensor(kf, device=z.device) / torch.tensor(k, device=z.device)).long()
        par = torch.minimum(par, torch.tensor(g, device=z.device) - 1)
        t = self.up[i](z)[(par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]]
        if self.image and self.zoom is not None:
            e = self.encoder.patch_embed
            with torch.no_grad():
                im = e(patchify(vol.float(), kf), e.weights(sp.float() * (self.patch_mm / self.fine_mm[i]), kf))
            t = t + self.zoom[i](im)
        elif self.image:
            e = self.fine[i][0].patch_embed
            t = t + e(patchify(vol.float(), kf), e.weights(sp.float(), kf))
        cu = torch.tensor([*range(0, len(t), self.fine_seg or len(t)), len(t)], device=z.device, dtype=torch.int32)
        for curve, enc in zip(self.fine_curves, self.fine[i]):
            order = keys(curve, xyz, gf, self.seed).argsort()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = enc.flat(t[order], cu).float()[order.argsort()]
        return t, kf, gf.tolist()

    def kernel_decode(self, z, emb, sp, k, g):
        """Tokens [N, c] on grid g -> logits [n_out, *(g k)] at the native voxels: per token coefficients in the
        basis of `emb` (KernelEmbed), each voxel the mean of that function over its extent (same 1D integrals)."""
        # each basis function at RMS 1 over the patch's voxels (any spacing), the sum of r^3 terms over r^1.5: the
        # trained encoder's phi reach 1.3, unscaled every Adam step moved the logits by ~10 (no learning, 17/10)
        px, py, pz = (p / p.pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6) for p in emb.axis_integrals(sp, k))
        r = emb.rank

        def dec(zz):   # in chunks recomputed in backward: the coefficients are tokens x classes x r^3
            return torch.einsum("noabc,xa,yb,zc->noxyz", self.coef(zz).view(len(zz), -1, r, r, r), px, py, pz) / r ** 1.5
        n = max(1, (1 << 25) // (self.coef[-1].out_features))
        m = torch.cat([checkpoint(dec, z[i:i + n], use_reentrant=False) for i in range(0, len(z), n)])
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

    seg_extent = True   # segrun appends the scan's shape to the context (SegNet reads no padding)

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
                     kernel=bool(self.args.get("kernel")), refine=int(self.args.get("refine") or 0),
                     image=bool(self.args.get("image")), fine_curves=self.args.get("fine_curves"), rough=bool(self.args.get("rough")),
                     zoom=bool(self.args.get("zoom")), fine_seg=int(self.args.get("fine_seg") or 0),
                     erase=self.args.get("erase") or "channel", fine_mm=self.args.get("fine_mm"))
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
        stats = ("mean", "std", "mean_tta", "std_tta", *(k for k in out if "@" in k))   # k@<curve>: one curve's
        for stat in stats:  # per block: last (no suffix), taps (_b<block>)
            if stat in out:
                feats |= {stat + ("" if j == 0 else f"_b{self._taps[j - 1] + 1}"): block(out[stat][0], j) for j in range(n)}
        return {"features": feats, "canonical": "mean", "derived": [],
                "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}}

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__}


build = SFC
