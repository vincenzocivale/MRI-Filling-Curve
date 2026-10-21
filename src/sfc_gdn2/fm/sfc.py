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

import math
from pathlib import Path

import nibabel.orientations as nio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .base import Image, Wrapper
from .token_cache import TokenCache, fine_ok

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
                 fine_mm: list | None = None, image_kernel: str = "own", fine_encoder: bool = False,
                 refine_frac: float = 1.0, token_cache: str | None = None, fine_pca: int = 0,
                 adaptive: bool = False, refine_tau: float = 0.9):
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
            # image_kernel "encoder" (18/10): the image at each fine level through the encoder's own frozen kernel applied
            # to the fine cell (its patch size set to the cell's: with basis bspline the exact refinement of the
            # encoder's family, with fourier the same physical frequencies over a smaller window), projected to 128,
            # instead of a kernel of the decoder's own
            assert image_kernel in ("own", "encoder"), image_kernel
            self.image_kernel = image_kernel
            self.enc_img = nn.ModuleList(nn.Linear(encoder.dim, 128) for _ in range(self.refine)) \
                if image and image_kernel == "encoder" else None
            # fine_encoder (18/10): each fine level also gets the frozen encoder itself run at that level's patch (a
            # model trained at several resolutions reads every scale: the same tokens), through the level-0 projection
            self.fine_tok = nn.ModuleList(nn.Linear(c, 128) for _ in range(self.refine)) if fine_encoder else None
            # refine_frac < 1 (18/10, adaptive): the encoder re-reads only the children of that share of the parent cells
            # that differ most from their neighbours (an edge, no label needed); the others keep the parent token
            self.refine_frac, self.enc_curves = float(refine_frac), list(curves)
            # token_cache (20/10, fm/token_cache.py): the encoder's tokens of the whole scan, read once at 16 / 8 / 4 mm
            # on nested grids; tiles start on the 16 mm grid (segrun aligns them), the decoder takes the cached tokens
            # (16 mm and the finer levels: last block ++ taps through proj) instead of running the encoder
            c = 128
        if self.kernel:
            self.coef = nn.Sequential(nn.Linear(c, 64), nn.GELU(), nn.Linear(64, n_out * encoder.patch_embed.rank ** 3))
            del self.decode
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0, embed_only=embed_only, reverse=tta)
        # every decoder can read the cached whole-scan tokens (20/10)
        self.cache, self._cur, self.width = (TokenCache(token_cache) if token_cache else None), None, width
        # adaptive (21/10, notes/adaptive_decoding.md): a per-cell head on the whole scan's 16 mm tokens, the finer
        # levels only on the cells it is unsure of (+ GT edges and a random share in training), loss on points
        self.adaptive, self.tau = bool(adaptive), float(refine_tau)
        if self.adaptive:
            assert self.cache is not None and self.kernel and self.refine, "adaptive decoding: token cache and decoder F"
            self.coarse = nn.Sequential(nn.Linear(encoder.dim, 128), nn.GELU(), nn.Linear(128, n_out))
        if self.cache is not None and getattr(self, "fine_tok", None) is not None:
            self.cache.levels = (16.0, 8.0, 4.0)
            if fine_pca:   # the fine levels' tokens on token_cache.project's PCA basis: fine_tok reads them directly
                self.cache.pca = int(fine_pca)
                self.fine_tok = nn.ModuleList(nn.Linear(int(fine_pca), 128) for _ in range(self.refine))

    def forward(self, x: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        """x [B, 1, *T] native voxels, spacing [B, 3] mm, or [B, 6] = spacing + the scan's shape (segrun `seg_extent`)
        -> logits [B, n_out, *T]. Only the scan's own voxels are read: segrun right-pads a tile up to SEG_PATCH voxels,
        ~160 mm for resampled models but up to 1.2 m of zeros on our native grids (TotalSeg 64^3 at 7.5 mm: 5e5 tokens,
        GPU OOM; SOOP 25 slices of 6 mm: 85 % of the tile), never seen in pretraining. The padding (label 255,
        ignored) gets zero logits."""
        from ..data.volume import patch_dims, patch_grid, patchify

        T = tuple(x.shape[2:])
        cached = getattr(self, "cache", None) is not None and spacing.shape[1] >= 10   # spacing, shape, volume, corner
        if cached:
            vids, corners = spacing[:, 6].long().tolist(), spacing[:, 7:10].long().tolist()
        ext = spacing[:, 3:6].long().tolist() if spacing.shape[1] > 3 else [T] * len(x)
        spacing = spacing[:, :3]
        V = [tuple(min(a, t) for a, t in zip(e, T)) for e in ext]                           # the scan's part of each tile
        if cached:   # tiles anywhere on the scan (segrun pads the volume so the last one ends on the 16 mm grid)
            V = [tuple(max(1, min(a - c, t)) for a, c, t in zip(e, co, T)) for e, co in zip(ext, corners)]
        xs = [x[b, 0, : v[0], : v[1], : v[2]].float() for b, v in enumerate(V)]
        if cached:
            ids = [self.cache.ids[v] for v in vids]
            k4s = [self.cache.meta(i)["k4"] for i in ids]
            ks = [[4 * a for a in k4] for k4 in k4s]
            grid = [patch_grid(v, k).to(x.device) for v, k in zip(V, ks)]
            w16 = self.width // (2 if self.context else 1)   # this decoder's blocks: the cache holds last ++ taps
            bi = [self.cache.tokens(i, 16.0, c, grid[b].tolist(), k)[:, :w16].to(x.device).float()
                  for b, (i, c, k) in enumerate(zip(ids, corners, ks))]
        else:
            ks = [patch_dims(s, self.patch_mm) for s in spacing.tolist()]
            grid = [patch_grid(v, k).to(x.device) for v, k in zip(V, ks)]
            with torch.no_grad():   # one tile at a time
                bi = [self.extract({"patches": [patchify(xs[b], k)], "grid": grid[b][None],
                                    "k": torch.tensor([k], device=x.device), "spacing": spacing[b:b + 1].float()})["bi"][0]
                      for b, k in enumerate(ks)]
        out, st = [], []
        for b, k in enumerate(ks):
            g, v = grid[b].tolist(), V[b]
            tok = bi[b][: int(grid[b].prod())]
            if self.unique and not cached:   # the cache holds h once
                tok = tok[:, : tok.shape[1] // 2]
            if self.context:  # the tile's mean token beside each token: the decoder can centre it (token - mean)
                tok = torch.cat([tok, tok.mean(0, keepdim=True).expand_as(tok)], -1)
            z = self.proj(tok).T.reshape(1, -1, *g)
            z = z if self.mix is None else z + self.mix(z)
            if self.kernel:   # the tiles' levels together below (fine_mix packs their sequences)
                st.append([z[0].flatten(1).T, k, g, self.encoder.patch_embed, (ids[b], corners[b], k4s[b]) if cached else None])
                continue
            # in slabs along the first axis: with coarse voxels a tile holds 10^4-10^5 tokens and 16^3 cells each
            s = max(1, (1 << 25) // (DECODE ** 3 * g[1] * g[2] * max(self.decode.decode[-1].out_channels, 1)))

            def slab(zz, kk=k, gg=g):   # recomputed in backward
                return F.interpolate(self.decode(zz), size=[zz.shape[2] * kk[0], gg[1] * kk[1], gg[2] * kk[2]],
                                     mode="trilinear", align_corners=False)
            m = torch.cat([checkpoint(slab, z[:, :, i:i + s], use_reentrant=False) for i in range(0, g[0], s)], 2)
            out.append(m[0, :, : v[0], : v[1], : v[2]])
        if self.kernel:
            for i in range(self.refine):   # the finer levels only while their cells hold >= 8 voxels
                act, inp = [], []
                for b, (z, k, g, _, cur) in enumerate(st):
                    self._cur = cur
                    if i and (st[b][3] is not self.fine[i - 1][0].patch_embed or not fine_ok(self.fine_dims(spacing[b], i))):
                        continue
                    act.append(b)
                    inp.append(self.to_fine(z, xs[b], spacing[b], k, g, i))
                if not act:
                    break
                ts = self.fine_mix([a[0] for a in inp], [a[1] for a in inp], [a[2] for a in inp], i)
                for b, t, (_, _, gf, kf) in zip(act, ts, inp):
                    st[b] = [t, kf, gf.tolist(), self.fine[i][0].patch_embed, st[b][4]]
            out = [self.kernel_decode(z, emb, spacing[b], k, g)[:, : V[b][0], : V[b][1], : V[b][2]]
                   for b, (z, k, g, emb, _) in enumerate(st)]
        return torch.stack([F.pad(o, [p for v, t in zip(reversed(o.shape[1:]), reversed(T)) for p in (0, t - v)])
                            for o in out])

    def fine_dims(self, sp, i) -> list:
        """Voxels per axis of a level-i fine cell: the cache's nested grids (k4 x 2^(1-i)), else the level's patch."""
        from ..data.volume import patch_dims
        return [a * 2 ** (1 - i) for a in self._cur[2]] if self._cur else patch_dims(sp.tolist(), self.fine_mm[i])

    def to_fine(self, z, vol, sp, k, g, i, keep=None, zmap=None):
        """Tokens [N, c] on grid g -> the fine level's inputs [Nf, 128] on the grid of half the patch (parent token +
        the voxels' embedding), their coordinates, grid and patch dims; fine_mix mixes them along each curve."""
        from ..curves import grid_coords, keys
        from ..data.volume import patch_grid, patchify

        kf = self.fine_dims(sp, i)
        gf = patch_grid(vol.shape, kf).to(z.device)
        xyz = grid_coords(gf[None], int(gf.prod()))[0]                                    # [Nf, 3] raster
        if keep is not None:   # adaptive decoding: only these children (z holds the kept parents, zmap: dense -> row)
            xyz = xyz[keep]
        par = ((xyz + 0.5) * torch.tensor(kf, device=z.device) / torch.tensor(k, device=z.device)).long()
        par = torch.minimum(par, torch.tensor(g, device=z.device) - 1)
        pidx = (par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]
        t = self.up[i](z)[pidx if zmap is None else zmap[pidx]]
        sub = (lambda a: a) if keep is None else (lambda a: a[keep])
        if self._cur is not None and self.fine_tok is not None:   # fine_encoder from the cache: the whole-scan reading
            ci, corner, _ = self._cur                                 # of this level, last block ++ taps as at 16 mm
            tf = self.cache.tokens(ci, 8.0 if i == 0 else 4.0, corner, gf.tolist(), kf).to(z.device).float()
            tf = sub(tf)
            t = t + self.fine_tok[i](tf if self.cache.pca else self.proj(tf))
        elif self.fine_tok is not None:   # the frozen encoder at this level's patch, one sequence (curves summed in turn)
            if self.refine_frac >= 1:   # every child (the extractor's reading without its h ++ h copies and statistics:
                sel = torch.arange(len(xyz), device=z.device)   # TotalSeg's 4 mm level, ~2e5 tokens, did not fit)
            else:   # adaptive: only the children of the most edge-like parents
                zg = z.T.reshape(1, -1, *g)
                edge = (zg - F.avg_pool3d(F.pad(zg, (1,) * 6, mode="replicate"), 3, 1)).norm(dim=1).flatten()
                top = edge >= edge.quantile(1 - self.refine_frac)
                sel = top[(par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]].nonzero()[:, 0]
            if len(sel):
                pe, mm0 = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm
                pe.patch_mm = self.fine_mm[i]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x = pe(patchify(vol.float(), kf)[sel], pe.weights(sp.float(), kf))
                    cu1 = torch.tensor([0, len(sel)], device=z.device, dtype=torch.int32)
                    bf = sum(self.encoder.flat(x[o], cu1).float()[o.argsort()]
                             for o in (keys(c, xyz[sel], gf, self.seed).argsort() for c in self.enc_curves)) / len(self.enc_curves)
                pe.patch_mm = mm0
                t = t.index_add(0, sel, self.fine_tok[i](self.proj(bf)))
        if self.image and self.enc_img is not None:
            pe, mm0 = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm
            pe.patch_mm = self.fine_mm[i]
            with torch.no_grad():
                im = pe(sub(patchify(vol.float(), kf)), pe.weights(sp.float(), kf))
            pe.patch_mm = mm0
            t = t + self.enc_img[i](im)
        elif self.image and self.zoom is not None:
            e = self.encoder.patch_embed
            with torch.no_grad():
                im = e(sub(patchify(vol.float(), kf)), e.weights(sp.float() * (self.patch_mm / self.fine_mm[i]), kf))
            t = t + self.zoom[i](im)
        elif self.image:
            e = self.fine[i][0].patch_embed
            t = t + e(sub(patchify(vol.float(), kf)), e.weights(sp.float(), kf))
        return t, xyz, gf, kf

    def fine_mix(self, ts: list, xyzs: list, gfs: list, i: int) -> list:
        """Level i's refine blocks along each curve of fine_curves in turn, the tiles' sequences packed in one call
        (cu_seqlens): a batch of sliding-window tiles costs one GDN-2 call per curve, not one per tile."""
        from ..curves import keys

        for curve, enc in zip(self.fine_curves, self.fine[i]):
            orders = [keys(curve, xyz, gf, self.seed).argsort() for xyz, gf in zip(xyzs, gfs)]
            cu, o = [0], 0
            for t in ts:
                cu += [o + j for j in range(self.fine_seg or len(t), len(t), self.fine_seg or len(t))] + [o + len(t)]
                o += len(t)
            cu = torch.tensor(cu, device=ts[0].device, dtype=torch.int32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = enc.flat(torch.cat([t[od] for t, od in zip(ts, orders)]), cu).float()
            ts = [hh[od.argsort()] for hh, od in zip(h.split([len(t) for t in ts]), orders)]
        return ts

    def kernel_decode(self, z, emb, sp, k, g):
        """Tokens [N, c] on grid g -> logits [n_out, *(g k)] at the native voxels: per token coefficients in the
        basis of `emb` (KernelEmbed), each voxel the mean of that function over its extent (same 1D integrals)."""
        # each basis function at RMS 1 over the patch's voxels (any spacing), the sum of r^3 terms over r^1.5: the
        # trained encoder's phi reach 1.3, unscaled every Adam step moved the logits by ~10 (no learning, 17/10)
        r = emb.rank
        # rank per axis = the cell's voxels on that axis (at most r, 20/10): more functions than voxels are redundant,
        # and with ~50 classes the r^3 coefficients per token made decoder F unusable on TotalSeg (~57 s per iteration)
        R = [min(r, int(a)) for a in k]
        px, py, pz = (p[:, :q] / p[:, :q].pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6)
                      for p, q in zip(emb.axis_integrals(sp, k), R))
        lin = self.coef[-1]
        w = lin.weight.view(-1, r, r, r, lin.in_features)[:, : R[0], : R[1], : R[2]].reshape(-1, lin.in_features)
        b = lin.bias.view(-1, r, r, r)[:, : R[0], : R[1], : R[2]].reshape(-1)

        def dec(zz):   # in chunks recomputed in backward: the coefficients are tokens x classes x R0 R1 R2
            c = nn.functional.linear(self.coef[:-1](zz), w, b).view(len(zz), -1, *R)
            return torch.einsum("noabc,xa,yb,zc->noxyz", c, px, py, pz) / (R[0] * R[1] * R[2]) ** 0.5
        n = max(1, (1 << 25) // max(len(b), 1))
        m = torch.cat([checkpoint(dec, z[i:i + n], use_reentrant=False) for i in range(0, len(z), n)])
        if g is None:   # per cell: [N, n_out, *k]
            return m
        o = m.shape[1]
        return m.reshape(*g, o, *k).permute(3, 0, 4, 1, 5, 2, 6).reshape(o, *(a * c for a, c in zip(g, k)))


    # ------------------------------------------------------------------------------ adaptive decoding (21/10)
    @staticmethod
    def _cells(y: torch.Tensor, k) -> tuple:
        """Labels [*S] (255 = ignore) -> per cell of k voxels: labels [N, P] and valid [N, P] (padding invalid)."""
        from ..data.volume import patchify
        v = patchify(y.float() + 1, k)
        return (v - 1).clamp_min(0).long(), (v > 0) & (v < 256)

    @staticmethod
    def _dilate(sel: torch.Tensor, g) -> torch.Tensor:
        return F.max_pool3d(sel.float().view(1, 1, *g), 3, 1, 1).flatten().bool()

    def _coarse(self, ci: str, corner, k4, V, dev) -> tuple:
        """The 16 mm tokens of a region (cache) -> projected tokens [N16, d], per-cell logits, cell dims, grid."""
        k16 = [4 * a for a in k4]
        g16 = [-(-v // a) for v, a in zip(V, k16)]
        w16 = self.width // (2 if self.context else 1)
        z16 = self.proj(self.cache.tokens(ci, 16.0, corner, g16, k16)[:, :w16].to(dev).float())
        return z16, self.coarse(z16), k16, g16

    def _chain(self, z16, sel, vol, sp, k16, g16, cur) -> tuple:
        """The refine levels on the descendants of the selected 16 mm cells only -> tokens of the finest cells, their
        dims, grid, dense indices in it, and the kernel embedding to decode them with."""
        from ..curves import grid_coords
        self._cur, dev = cur, z16.device
        idx = sel.nonzero()[:, 0]
        rows = torch.full((len(z16),), -1, dtype=torch.long, device=dev)
        rows[idx] = torch.arange(len(idx), device=dev)
        z, k, g, dense, emb = z16[idx], k16, g16, idx, self.encoder.patch_embed
        for i in range(self.refine):
            if i and not fine_ok(self.fine_dims(sp, i)):
                break
            kf = self.fine_dims(sp, i)
            gf = [-(-v // a) for v, a in zip(vol.shape, kf)]
            xyz = grid_coords(torch.tensor(gf, device=dev)[None], math.prod(gf))[0]
            par = ((xyz + 0.5) * torch.tensor(kf, device=dev) / torch.tensor(k, device=dev)).long()
            par = torch.minimum(par, torch.tensor(g, device=dev) - 1)
            keep = (rows[(par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]] >= 0).nonzero()[:, 0]
            if not len(keep):   # no child inside the scan (fine_mix cannot take an empty sequence)
                break
            t, xyzk, gft, _ = self.to_fine(z, vol, sp, k, g, i, keep=keep, zmap=rows)
            t = self.fine_mix([t], [xyzk], [gft], i)[0]
            rows = torch.full((math.prod(gf),), -1, dtype=torch.long, device=dev)
            rows[keep] = torch.arange(len(keep), device=dev)
            z, k, g, dense, emb = t, kf, gf, keep, self.fine[i][0].patch_embed
        return z, k, g, dense, emb

    def adaptive_sums(self, x: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor, n_out: int, n_points: int = 16384):
        """One training tile ([C, *T] voxels, [*T] labels, segrun's ctx row) -> the loss sums (segrun.sums_loss) of
        the final prediction (fine levels in the refined cells, every voxel; the coarse head elsewhere, on uniform
        points weighted to the region's voxel count) and of the coarse head alone (uniform points everywhere)."""
        dev, T = x.device, tuple(x.shape[1:])
        corner, ext = ctx[7:10].long().tolist(), ctx[3:6].long().tolist()
        V = [max(1, min(e - c, t)) for e, c, t in zip(ext, corner, T)]
        vol, yy, sp = x[0, : V[0], : V[1], : V[2]].float(), y[: V[0], : V[1], : V[2]], ctx[:3].float()
        ci = self.cache.ids[int(ctx[6])]
        k4 = self.cache.meta(ci)["k4"]
        z16, cl, k16, g16 = self._coarse(ci, corner, k4, V, dev)
        lab16, val16 = self._cells(yy, k16)
        hi = torch.where(val16, lab16, -1).max(1).values
        lo = torch.where(val16, lab16, 1 << 20).min(1).values
        edge = (hi >= 0) & (hi != lo)                                          # cells holding two labels (GT)
        # the criterion used at test, capped at the most unsure 20 % of the cells (early on every cell is unsure)
        u = 1 - cl.detach().float().softmax(1).max(1).values
        unsure = (u > 1 - self.tau) & (u >= u.quantile(0.8))
        sel = self._dilate(edge | unsure | (torch.rand(len(cl), device=dev) < 0.05), g16)
        # uniform points over the region's voxels (cell, voxel within it)
        cell = torch.randint(len(cl), (n_points,), device=dev)
        vox = torch.randint(lab16.shape[1], (n_points,), device=dev)
        ok = val16[cell, vox]
        cell, lab = cell[ok], lab16[cell[ok], vox[ok]]
        total = val16.sum()
        coarse = sums(cl[cell], lab, total / max(len(lab), 1), n_out)          # the coarse head everywhere
        out = ~sel[cell]
        n_out_vox = val16[~sel].sum()
        final = sums(cl[cell[out]], lab[out], n_out_vox / max(int(out.sum()), 1), n_out)
        z = kf = None
        if sel.any():
            z, kf, _, dense, emb = self._chain(z16, sel, vol, sp, k16, g16, (ci, corner, k4))
        if z is not None and kf != k16:
            lg = self.kernel_decode(z, emb, sp, kf, None).flatten(2).transpose(1, 2)   # [Nf, P, n_out]
            labf, valf = self._cells(yy, kf)
            labf, valf = labf[dense], valf[dense]
            final = [a + b for a, b in zip(final, sums(lg[valf], labf[valf], 1.0, n_out))]
        self.last_refined = float(sel.float().mean())
        return final, coarse

    @torch.no_grad()
    def adaptive_predict(self, x: torch.Tensor, ctx: torch.Tensor, n_out: int) -> torch.Tensor:
        """A whole scan [C, *S] in one pass -> class probabilities [n_out, *S]: the coarse head on every 16 mm cell,
        the finer levels on the cells it is unsure of (max probability < tau) and their neighbours, in chunks."""
        from ..curves import grid_coords
        dev, S = x.device, tuple(x.shape[1:])
        vol, sp = x[0].float(), ctx[:3].float().to(dev)
        ci = self.cache.ids[int(ctx[6])]
        k4 = self.cache.meta(ci)["k4"]
        cur = (ci, [0, 0, 0], k4)
        z16, cl, k16, g16 = self._coarse(ci, [0, 0, 0], k4, S, dev)
        p16 = cl.float().softmax(1)
        sel = self._dilate(p16.max(1).values < self.tau, g16)
        self._cur = cur
        last = max(i for i in range(self.refine) if i == 0 or fine_ok(self.fine_dims(sp, i)))
        kf = self.fine_dims(sp, last)
        gf = [-(-v // a) for v, a in zip(S, kf)]
        xyz = grid_coords(torch.tensor(gf, device=dev)[None], math.prod(gf))[0]
        par = torch.minimum((xyz * torch.tensor(kf, device=dev)) // torch.tensor(k16, device=dev),
                            torch.tensor(g16, device=dev) - 1)
        fits = n_out * math.prod(gf) * math.prod(kf) * 2 <= torch.cuda.mem_get_info(dev)[0] / 4
        pr = p16[(par[:, 0] * g16[1] + par[:, 1]) * g16[2] + par[:, 2]].half().to(dev if fits else "cpu")
        pr = pr[:, :, None].expand(-1, -1, math.prod(kf)).contiguous()                  # [Nf, n_out, P]
        for chunk in sel.nonzero()[:, 0].split(4096):
            s = torch.zeros_like(sel)
            s[chunk] = True
            z, kfc, _, dense, emb = self._chain(z16, s, vol, sp, k16, g16, cur)
            if kfc != kf:   # the chain stopped early (no child in the scan): the coarse probabilities stay
                continue
            pr[dense.to(pr.device)] = self.kernel_decode(z, emb, sp, kfc, None).float().softmax(1).flatten(2).half().to(pr.device)
        self.last_refined = float(sel.float().mean())
        out = pr.view(*gf, n_out, *kf).permute(3, 0, 4, 1, 5, 2, 6).reshape(n_out, *(a * b for a, b in zip(gf, kf)))
        return out[:, : S[0], : S[1], : S[2]]


def sums(logits: torch.Tensor, lab: torch.Tensor, w, k: int) -> list:
    """Points' logits [M, k] and labels [M], each point standing for w voxels -> [CE sum, voxels, per class: soft
    intersection, predicted sum, target sum] (segrun.sums_loss: CE / voxels + 1 - mean soft Dice of the classes > 0)."""
    lp = logits.float().log_softmax(1)
    p = lp.exp()
    z = torch.zeros(k, device=logits.device)
    return [-lp.gather(1, lab[:, None]).sum() * w, torch.as_tensor(len(lab) * w, device=logits.device, dtype=torch.float),
            z.scatter_add(0, lab, p.gather(1, lab[:, None])[:, 0]) * w, p.sum(0) * w,
            z.scatter_add(0, lab, torch.ones_like(lab, dtype=torch.float)) * w]


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
        if self.args.get("patch_mm"):  # read at another patch size (18/10: a model trained at several resolutions)
            self.run_cfg["data"]["patch_mm"] = float(self.args["patch_mm"])
        self.curves = list(self.args.get("curves") or self.run_cfg["objective"]["view_curves"])
        self._extract = None  # built on first use: preprocessing workers never load GDN-2
        self._cov = None      # args.cov_basis: {basis: {name: (mu, P)}, dim} of probe.second_order

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("sfc: single-modality, pass a NIfTI path")
        from ..data.volume import decode, open_image, patch_dims, patch_grid

        img = open_image(image)
        arr, spacing, (a, c) = decode(img)
        mm = self.run_cfg["data"]["patch_mm"]
        k = patch_dims(spacing, mm)
        # args.max_tokens / args.fallback_mm (20/10): a volume whose grid would exceed max_tokens is read at fallback_mm
        # (8 mm whole-body scans: ~10^6 tokens, ~80 GB for the encoder alone)
        if self.args.get("max_tokens") and np.prod(patch_grid(arr.shape, k).tolist()) > int(self.args["max_tokens"]):
            mm = float(self.args["fallback_mm"])
            k = patch_dims(spacing, mm)
        ornt = nio.io_orientation(img.affine)
        src = np.linalg.inv(nio.inv_ornt_aff(ornt, img.shape[:3]))  # source voxel -> canonical voxel
        meta = {"source_affine": torch.from_numpy(img.affine.copy()), "source_shape": tuple(img.shape[:3]),
                "source_to_input": torch.from_numpy(src), "input_shape": tuple(arr.shape),
                "spacing": tuple(spacing), "patch_mm": mm}
        return {"volume": torch.from_numpy(np.ascontiguousarray(arr)), "value": (a, c), "k": torch.tensor(k),
                "grid": patch_grid(arr.shape, k), "spacing": torch.tensor(spacing, dtype=torch.float32), "meta": meta}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        a, c = prepared["value"]
        return (prepared["volume"].float() * a + c)[None], prepared["meta"]["source_to_input"].numpy()

    seg_extent = True   # segrun appends the scan's shape to the context (SegNet reads no padding)
    seg_microbatch = True   # many classes: segrun.micro_step (the batch's gradient, one scan at a time)

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
                     erase=self.args.get("erase") or "channel", fine_mm=self.args.get("fine_mm"),
                     image_kernel=self.args.get("image_kernel") or "own", fine_encoder=bool(self.args.get("fine_encoder")),
                     refine_frac=float(self.args.get("refine_frac") or 1.0), token_cache=self.args.get("token_cache"),
                     fine_pca=int(self.args.get("fine_pca") or 0), adaptive=bool(self.args.get("adaptive")),
                     refine_tau=float(self.args.get("refine_tau") or 0.9))
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
                                              reverse=bool(self.args.get("tta")),  # also the *_tta readings
                                              per_curve_stats=self.args.get("per_curve", True))
            self._extract_enc = enc
        return self._extract

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        from ..data.volume import patchify

        dev = torch.device(self.device)
        self.extractor()   # built (and the encoder) before its kernel takes this volume's patch size
        pe = self._extract_enc.patch_embed
        pe.patch_mm = prepared["meta"].get("patch_mm", pe.patch_mm)
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
        if self.args.get("cov_basis"):  # the second-order reading computed here, without storing the token map (20/10)
            if self._cov is None:
                self._cov = torch.load(self.args["cov_basis"], map_location=dev)
            t, dim = bi.float(), self._cov["dim"]
            keep = t.norm(dim=1) > 0
            iu = torch.triu_indices(dim, dim, device=dev)
            for name, sel in (("covsqrt", keep), ("covsqrt_fg", keep & out["fg"][0, : len(x)])):
                mu, pr = self._cov["basis"][name]
                z = (t[sel] - mu) @ pr
                z = z - z.mean(0)
                e, v = torch.linalg.eigh(z.T @ z / max(len(z) - 1, 1) + 1e-6 * torch.eye(dim, device=dev))
                feats[name] = ((v * e.clamp(min=0).sqrt()) @ v.T)[iu[0], iu[1]].half()
        return {"features": feats, "canonical": "mean", "derived": [],
                "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}}

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__}


build = SFC
