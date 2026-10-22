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
MIX_MAX = 1 << 19   # tokens per refine-block call (fine_mix)
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
                 adaptive: bool = False, refine_tau: float = 0.9, matryoshka: bool = False, quota: float = 0.5, head_w: float = 1.0,
                 rand_dec: float = 0.02, test_all: bool = False, kernel_skip: int = 0, kernel_skip_mm: float = 0.0,
                 level2: bool = False, decoder_c: bool = False, c_points: int = 32768):
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
        # kernel_skip K (21/10, user: small objects): the decoded cell's own voxels enter the output in the kernel's
        # basis, without the token's bottleneck: their coefficients through K learned separable maps per axis (a
        # filter in the basis, identity at init), back on the voxels, then per voxel sum_k g_ok(token) gelu(a_k(token)
        # f_k + b_k(token)). The token says what to look for, the voxels where it is (gates zero at init: = decoder F).
        # kernel_skip_mm m (22/10): the filter reads the cell plus m mm of its neighbours per side (the window in the
        # basis of its own extent), so a dot or an edge across the cell border is seen whole
        self.kskip, self.kskip_mm = int(kernel_skip), float(kernel_skip_mm)
        # level2 (22/10, user: "forse dobbiamo davvero fare 2 mm"): a decoder-only 2 mm level under the decoded 4 mm
        # cells: each child = its parent's token + its own voxels through the encoder's frozen kernel at 2 mm (the
        # B-spline refinement of the 4 mm basis), mixed along the curve by 2 GDN-2 blocks (the fine cells see each
        # other), then decoded at the native voxels; the encoder itself never reads 2 mm
        # decoder C (22/10, notes/adaptive_decoding.md "decoder C"): the 16 / 8 / 4 mm levels on every cell of a large tile
        # (the whole scan at test), each level's tokens spread as a continuous field (quadratic B-splines over the 27
        # nearest cells: smooth across cell borders, nothing resampled), and a point head: per voxel [the three fields,
        # the voxel's and its 6 neighbours' intensities] -> logits. Trained on stratified points (class x boundary,
        # importance-weighted: unbiased CE and soft-Dice sums) + the cell heads; no per-cell coefficient output.
        self.decoder_c, self.c_points = bool(decoder_c), int(c_points)
        if self.decoder_c:
            CF = 32
            self.cfield = nn.ModuleList([nn.Linear(encoder.dim, CF), nn.Linear(128, CF), nn.Linear(128, CF)])
            self.chead = nn.Sequential(nn.Linear(3 * CF + 7, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                       nn.Linear(128, n_out))
            if hasattr(self, "coef"):   # the per-cell coefficient output is not used (n_out x rank^3 per cell)
                del self.coef
        self.level2 = bool(level2)
        if self.level2:
            from ..model import Encoder
            self.up2, self.img2 = nn.Linear(128, 128), nn.Linear(encoder.dim, 128)
            self.fine2 = nn.ModuleList(Encoder(128, 2, 64, 2, bidir="alternate", patch_mm=2.0, freq_decay=0.0 if rough else 1.0)
                                       for _ in self.fine_curves)
            if erase == "head":
                for m in self.fine2.modules():
                    if hasattr(m, "b_proj") and hasattr(m, "head_k_dim"):
                        m.b_proj = HeadErase(m)
        if self.kskip:
            r = encoder.patch_embed.rank
            self.kA = nn.Parameter(torch.eye(r).expand(3, self.kskip, r, r).clone())
            self.kgate = nn.Linear(c, self.kskip * (2 + n_out))
            nn.init.zeros_(self.kgate.weight)
            nn.init.zeros_(self.kgate.bias)
            with torch.no_grad():
                self.kgate.bias[: self.kskip] = 1.0   # a = 1, b = 0, g = 0
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0, embed_only=embed_only, reverse=tta)
        # every decoder can read the cached whole-scan tokens (20/10)
        self.cache, self._cur, self.width = (TokenCache(token_cache) if token_cache else None), None, width
        # adaptive (21/10, notes/adaptive_decoding.md): a per-cell head on the whole scan's 16 mm tokens, the finer
        # levels only on the cells it is unsure of (+ GT edges and a random share in training), loss on points
        self.adaptive, self.tau = bool(adaptive), float(refine_tau)
        # matryoshka (21/10, notes/adaptive_decoding.md): a cell head at 16, 8 and 4 mm, each trained as a prediction of
        # its own (exact loss from the cells' label counts, summed: any depth is a valid stop); 16 -> 8 mm dense, the
        # 4 mm level only in the `quota` of 8 mm cells whose voxel intensities vary most (label-free, decided before
        # reading them), the native-voxel decoding only in its boundary / unsure cells
        self.matryoshka, self.quota, self.head_w = bool(matryoshka), float(quota), float(head_w)
        self.rand_dec, self.test_all = float(rand_dec), bool(test_all)   # training's random decoded share; decode all at test
        if self.matryoshka:
            assert self.cache is not None and self.kernel and self.refine == 2, "matryoshka decoding: token cache, F, refine 2"
            self.head16, self.head8, self.head4 = nn.Linear(encoder.dim, n_out), nn.Linear(128, n_out), nn.Linear(128, n_out)
        if self.adaptive:
            assert self.cache is not None and self.kernel and self.refine, "adaptive decoding: token cache and decoder F"
            self.cellhead = nn.Linear(128, n_out)   # one prediction per cell of the finest level
        if self.cache is not None and getattr(self, "fine_tok", None) is not None:
            self.cache.levels = (16.0, 8.0, 4.0)
            if fine_pca:   # the fine levels' tokens on token_cache.project's PCA basis: fine_tok reads them directly
                self.cache.pca = int(fine_pca)
                self.fine_tok = nn.ModuleList(nn.Linear(int(fine_pca), 128) for _ in range(self.refine))

    def forward(self, x: torch.Tensor, spacing: torch.Tensor, levels: bool = False, stop: int | None = None):
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
            self._z16 = [(a[0], a[1], a[2]) for a in st]   # the 16 mm tokens (matryoshka decoding's coarse head)
            self._lv = [[(a[0], a[1], a[2])] for a in st]   # every level's tokens, dims, grid (decoder C)
            for i in range(self.refine if stop is None else stop):   # the finer levels while their cells hold >= 8 voxels
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
                    self._lv[b].append((t, list(kf), gf.tolist()))
            if levels:   # adaptive decoding: each tile's finest tokens [z, k, g, emb, cur], its scan part V, spacing
                return st, V, spacing
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

    def fine_mix(self, ts: list, xyzs: list, gfs: list, i: int, encs=None) -> list:
        """Level i's refine blocks along each curve of fine_curves in turn, the tiles' sequences packed in one call
        (cu_seqlens): a batch of sliding-window tiles costs one GDN-2 call per curve, not one per tile."""
        from ..curves import keys

        for curve, enc in zip(self.fine_curves, self.fine[i] if encs is None else encs):
            orders = [keys(curve, xyz, gf, self.seed).argsort() for xyz, gf in zip(xyzs, gfs)]
            cu, o = [0], 0
            for t in ts:
                cu += [o + j for j in range(self.fine_seg or len(t), len(t), self.fine_seg or len(t))] + [o + len(t)]
                o += len(t)
            seq = torch.cat([t[od] for t, od in zip(ts, orders)])
            # calls of at most MIX_MAX tokens, cut at sequence boundaries: the short convolution's Triton grid failed
            # on a whole-body scan (~1e6 tokens in one call, 21/10)
            groups, a = [], 0
            for j in range(1, len(cu)):
                if cu[j] - cu[a] > MIX_MAX and j - 1 > a:
                    groups.append((a, j - 1))
                    a = j - 1
            groups.append((a, len(cu) - 1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = torch.cat([enc.flat(seq[cu[a]: cu[b]], torch.tensor([c - cu[a] for c in cu[a: b + 1]], device=seq.device,
                                                                        dtype=torch.int32)).float() for a, b in groups])
            ts = [hh[od.argsort()] for hh, od in zip(h.split([len(t) for t in ts]), orders)]
        return ts

    def kernel_decode(self, z, emb, sp, k, g, vox=None, kv=None):
        """Tokens [N, c] on grid g -> logits [n_out, *(g k)] at the native voxels: per token coefficients in the
        basis of `emb` (KernelEmbed), each voxel the mean of that function over its extent (same 1D integrals).
        vox [N, prod(kv)]: the cells' voxels (kernel_skip), kv: their window (default k; with a margin, k + 2m)."""
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

        use = self.kskip and vox is not None
        if use:   # the window's basis (its own extent), normalized as px
            kv = list(kv or k)
            Rv = [min(r, int(a)) for a in kv]
            qx, qy, qz = (p[:, :q] / p[:, :q].pow(2).mean(0, keepdim=True).sqrt().clamp_min(1e-6)
                          for p, q in zip(emb.axis_integrals(sp, kv), Rv))

        def dec(zz, vv):   # in chunks recomputed in backward: the coefficients are tokens x classes x R0 R1 R2
            c = nn.functional.linear(self.coef[:-1](zz), w, b).view(len(zz), -1, *R)
            out = torch.einsum("noabc,xa,yb,zc->noxyz", c, px, py, pz) / (R[0] * R[1] * R[2]) ** 0.5
            if vv is None:
                return out
            K = self.kskip
            ci = torch.einsum("nxyz,xa,yb,zc->nabc", vv.float().view(len(vv), *kv), qx, qy, qz) / math.prod(kv)
            A = [self.kA[j, :, : R[j], : Rv[j]] for j in range(3)]
            fc = torch.einsum("nabc,kda,keb,kfc->nkdef", ci, *A)
            f = torch.einsum("nkdef,xd,ye,zf->nkxyz", fc, px, py, pz)
            gt = self.kgate(zz)
            a, bb, gg = gt[:, :K], gt[:, K: 2 * K], gt[:, 2 * K:].view(len(zz), -1, K)
            h = nn.functional.gelu(a[:, :, None, None, None] * f + bb[:, :, None, None, None])
            return out + torch.einsum("nok,nkxyz->noxyz", gg, h)
        n = max(1, (1 << 25) // max(len(b) + (self.kskip * math.prod(kv) if use else 0), 1))
        m = torch.cat([checkpoint(dec, z[i:i + n], vox[i:i + n] if use else None, use_reentrant=False)
                       for i in range(0, len(z), n)])
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
        ok = (v > 0) & (v < 256)
        return torch.where(ok, v - 1, 0).long(), ok   # ignored voxels (255) get class 0 with weight 0, not index 255

    @staticmethod
    def _dilate(sel: torch.Tensor, g) -> torch.Tensor:
        return F.max_pool3d(sel.float().view(1, 1, *g), 3, 1, 1).flatten().bool()

    @staticmethod
    def _pred_edge(cl: torch.Tensor, g) -> torch.Tensor:
        """Cells whose predicted class differs from one of their 26 neighbours' (the predicted boundary): decoded at
        every voxel whatever the head's confidence (21/10: a head sure everywhere decoded nothing, blocks of 4 mm)."""
        a = cl.argmax(1).float().view(1, 1, *g)
        return (F.max_pool3d(a, 3, 1, 1) != -F.max_pool3d(-a, 3, 1, 1)).flatten()

    def _cell_sums(self, cl: torch.Tensor, hist: torch.Tensor) -> list:
        """Cells predicted by one logit vector each ([M, n_out]) and their voxels' label counts [M, n_out] -> the exact
        voxel sums of sums() (every voxel of a cell shares the cell's probabilities)."""
        lp = cl.float().log_softmax(1)
        p, nv = lp.exp(), hist.sum(1, keepdim=True)
        return [-(hist * lp).sum(), hist.sum(), (p * hist).sum(0), (p * nv).sum(0), hist.sum(0)]

    def adaptive_sums(self, x: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor, n_out: int, unsure: float = 0.05,
                      rand: float = 0.02, edges: float = 0.1) -> tuple:
        """A batch of tiles [B, C, *T] (labels [B, *T], segrun's ctx [B, 10]) -> the loss sums of the final prediction
        and of the cell head, summed over the batch. The refine levels run densely, as decoder F; only the expensive
        part, the kernel decoding of every native voxel (x n_out classes), is adaptive: at the finest level a cell head
        gives one prediction per cell, and the voxels are decoded only in cells holding two labels (GT edges, a random
        subset of at most `edges` of the cells), the most unsure `unsure` share and a random `rand` share. The other cells' voxels share their cell's prediction:
        their sums come exactly from the cell's label counts. Both losses are exact over every voxel."""
        st, V, sp = self(x, ctx, levels=True)
        fin = aux = None
        for b, (z, k, g, emb, _) in enumerate(st):
            yy = y[b, : V[b][0], : V[b][1], : V[b][2]]
            lab, val = self._cells(yy, k)
            hist = torch.zeros(len(lab), n_out, device=z.device).scatter_add_(1, lab, val.float())
            cl = self.cellhead(z)
            a = self._cell_sums(cl, hist)
            u = 1 - cl.detach().float().softmax(1).max(1).values
            # the decoding's backward is ~3/4 of the step (TotalSeg: 50 organs, many edge cells): a random subset of
            # the edge cells (at most `edges` of all cells) per step, the rest of them through the cell head as at test
            edge = (hist > 0).sum(1) > 1
            r = torch.rand(len(u), device=z.device)
            pe = self._pred_edge(cl.detach(), g)   # the test's criterion, subsampled as the GT edges (TotalSeg: 50
            r2 = torch.rand(len(u), device=z.device)   # organs, an early prediction has edges almost everywhere)
            sel = (edge & (r < edges * len(u) / max(int(edge.sum()), 1))) | (u >= u.quantile(1 - unsure)) | (r > 1 - rand) \
                | (pe & (r2 < edges * len(u) / max(int(pe.sum()), 1)))
            f = self._cell_sums(cl[~sel], hist[~sel])
            if sel.any():
                lg = self.kernel_decode(z[sel], emb, sp[b], k, None).flatten(2).transpose(1, 2)   # [Ns, P, n_out]
                f = [u_ + v_ for u_, v_ in zip(f, sums(lg[val[sel]], lab[sel][val[sel]], 1.0, n_out))]
            fin = f if fin is None else [u_ + v_ for u_, v_ in zip(fin, f)]
            aux = a if aux is None else [u_ + v_ for u_, v_ in zip(aux, a)]
            self.last_refined = float(sel.float().mean())
        return fin, aux

    @torch.no_grad()
    def adaptive_predict(self, x: torch.Tensor, ctx: torch.Tensor, n_out: int) -> torch.Tensor:
        """A whole scan [C, *S] in one pass (no tiles: the cached tokens already carry the whole scan) -> class
        probabilities [n_out, *S]: the refine levels on the whole scan, the cell head everywhere, the voxels decoded
        only in the cells it is unsure of (max probability < tau) and their neighbours."""
        dev, S = x.device, tuple(x.shape[1:])
        c = torch.cat([ctx[:7].float().cpu(), torch.zeros(3)]).to(dev)   # corner 0: the tile is the scan
        # the refine levels' sequences cut every 65536 tokens along the curve, the longest a 16 cm training tile gives
        # (a whole-body 4 mm sequence, ~1e6 tokens, passed the Triton grid limit of the short convolution)
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            st, _, sp = self(x[None].float(), c[None], levels=True)
        finally:
            self.fine_seg = fs
        z, k, g, emb, _ = st[0]
        p = self.cellhead(z).float().softmax(1)
        sel = self._dilate(p.max(1).values < self.tau, g) | self._pred_edge(p, g)
        P = math.prod(k)
        fits = n_out * len(p) * P * 2 <= torch.cuda.mem_get_info(dev)[0] / 4
        pr = p.half().to(dev if fits else "cpu")[:, :, None].expand(-1, -1, P).contiguous()       # [N, n_out, P]
        for idx in sel.nonzero()[:, 0].split(1 << 15):
            if not len(idx):   # nothing unsure (split of an empty selection still yields one empty chunk)
                continue
            pr[idx.to(pr.device)] = self.kernel_decode(z[idx], emb, sp[0], k, None).float().softmax(1).flatten(2).half().to(pr.device)
        self.last_refined = float(sel.float().mean())
        out = pr.view(*g, n_out, *k).permute(3, 0, 4, 1, 5, 2, 6).reshape(n_out, *(a * b for a, b in zip(g, k)))
        return out[:, : S[0], : S[1], : S[2]]

    def _vox(self, vol, k, cells, sp):
        """kernel_skip's input: the cells' voxels (+ kernel_skip_mm of margin per side) and the window dims."""
        if not self.kskip:
            return None, None
        if not self.kskip_mm:
            return patchify_any(vol, k)[cells], list(k)
        m = [max(1, round(self.kskip_mm / float(s))) for s in sp]
        return windows(vol, k, m, cells), [a + 2 * b for a, b in zip(k, m)]

    def decode_cells(self, z, cells, vol, sp, k, g, emb):
        """The decoded cells (tokens z, raster indices `cells` on grid g of k-voxel cells) -> logits [n, n_out, *k]
        at their native voxels; level2: through their 2 mm children."""
        r = [2 if self.level2 and a >= 2 and a % 2 == 0 else 1 for a in k]
        if math.prod(r) == 1:
            return self.kernel_decode(z, emb, sp, k, None, *self._vox(vol, k, cells, sp))
        dev, k2 = z.device, [a // q for a, q in zip(k, r)]
        g2 = [a * q for a, q in zip(g, r)]
        pc = torch.stack([cells // (g[1] * g[2]), cells // g[2] % g[1], cells % g[2]], 1)
        off = torch.stack(torch.meshgrid(*(torch.arange(q, device=dev) for q in r), indexing="ij"), -1).view(-1, 3)
        xyz = (pc[:, None] * torch.tensor(r, device=dev) + off[None]).view(-1, 3)        # parent-major, children raster
        cidx = (xyz[:, 0] * g2[1] + xyz[:, 1]) * g2[2] + xyz[:, 2]
        t = self.up2(z).repeat_interleave(len(off), 0)
        pe, mm0 = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm
        pe.patch_mm = 2.0
        with torch.no_grad():
            im = pe(patchify_any(vol, k2)[cidx], pe.weights(sp.float(), k2))
        pe.patch_mm = mm0
        t = self.fine_mix([t + self.img2(im)], [xyz], [torch.tensor(g2, device=dev)], 0, self.fine2)[0]
        out = self.kernel_decode(t, emb, sp, k2, None, *self._vox(vol, k2, cidx, sp))       # [n R, o, *k2]
        o = out.shape[1]
        return out.view(len(z), *r, o, *k2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(len(z), o, *k)

    # ------------------------------------------------------------------------------------- decoder C (22/10)
    def c_logits(self, lv, vol, pts):
        """Point head at voxels pts [P, 3] of this tile / scan: the three levels' fields + the local intensities."""
        fs = []
        for j in range(3):
            if j < len(lv):
                z, k, g = lv[j]
                fs.append(_c_field(self.cfield[j](z.float()), g, k, pts))
            else:   # no 4 mm level (coarse voxels): zeros
                fs.append(torch.zeros(len(pts), self.cfield[j].out_features, device=pts.device))
        return self.chead(torch.cat(fs + [_c_image(vol, pts)], -1))

    def c_sums(self, x, y, ctx, n_out):
        """Training (decoder C): every level on every cell of the tiles, the cell heads' exact sums and the point head's
        importance-weighted sums on stratified voxels."""
        _, V, _ = self(x, ctx, levels=True)
        fin, aux = None, [None, None, None]
        add = lambda a, b: b if a is None else [u + v for u, v in zip(a, b)]
        heads = (self.head16, self.head8, self.head4)
        for b, lv in enumerate(self._lv):
            yy, vol = y[b, : V[b][0], : V[b][1], : V[b][2]], x[b, 0, : V[b][0], : V[b][1], : V[b][2]]
            for j, (z, k, _) in enumerate(lv):
                lab, val = self._cells(yy, k)
                aux[j] = add(aux[j], self._cell_sums(heads[j](z), self._hist(lab, val, n_out)))
            idx, w = _c_points(yy, n_out, self.c_points)
            S = yy.shape
            pts = torch.stack([idx // (S[1] * S[2]), idx // S[2] % S[1], idx % S[2]], 1)
            lg = self.c_logits(lv, vol, pts)
            fin = add(fin, wsums(lg, yy.flatten()[idx].long(), w, n_out))
        return fin, [a for a in aux if a is not None]

    @torch.no_grad()
    def c_predict(self, x, ctx, n_out):
        """A whole scan [C, *S] -> probabilities [n_out, *S]: every level on the whole scan, the point head on every
        voxel (chunks)."""
        dev, S = x.device, tuple(x.shape[1:])
        c = torch.cat([ctx[:7].float().cpu(), torch.zeros(3)]).to(dev)
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            self(x[None].float(), c[None], levels=True)
        finally:
            self.fine_seg = fs
        lv, vol, N = self._lv[0], x[0], math.prod(S)
        fits = n_out * N * 2 <= torch.cuda.mem_get_info(dev)[0] / 4
        out = torch.empty(n_out, N, dtype=torch.half, device=dev if fits else "cpu")
        for a in range(0, N, 1 << 19):
            i = torch.arange(a, min(a + (1 << 19), N), device=dev)
            pts = torch.stack([i // (S[1] * S[2]), i // S[2] % S[1], i % S[2]], 1)
            out[:, a: a + len(i)] = self.c_logits(lv, vol, pts).float().softmax(1).T.half().to(out.device)
        return out.view(n_out, *S)

    # --------------------------------------------------------------------------------- matryoshka decoding (21/10)
    def _hist(self, lab, val, n_out):
        return torch.zeros(len(lab), n_out, device=lab.device).scatter_add_(1, lab, val.float())

    def _sparse4(self, z8, vol, sp, k8, g8, sel8):
        """The 4 mm level on the children of the selected 8 mm cells only -> their tokens, dims, grid, dense indices."""
        from ..curves import grid_coords
        dev = z8.device
        idx = sel8.nonzero()[:, 0]
        rows = torch.full((len(z8),), -1, dtype=torch.long, device=dev)
        rows[idx] = torch.arange(len(idx), device=dev)
        kf = self.fine_dims(sp, 1)
        gf = [-(-v // a) for v, a in zip(vol.shape, kf)]
        xyz = grid_coords(torch.tensor(gf, device=dev)[None], math.prod(gf))[0]
        par = torch.minimum(((xyz + 0.5) * torch.tensor(kf, device=dev) / torch.tensor(k8, device=dev)).long(),
                            torch.tensor(g8, device=dev) - 1)
        keep = (rows[(par[:, 0] * g8[1] + par[:, 1]) * g8[2] + par[:, 2]] >= 0).nonzero()[:, 0]
        t, xyzk, gft, _ = self.to_fine(z8[idx], vol, sp, k8, g8, 1, keep=keep, zmap=rows)
        return self.fine_mix([t], [xyzk], [gft], 1)[0], kf, gf, keep, par[keep]

    def _select8(self, vol, k8, inside):
        """The quota of the 8 mm cells (inside the scan) whose voxel intensities vary most."""
        from ..data.volume import patchify
        s = patchify(vol.float(), k8).std(1)[: len(inside)]
        s = torch.where(inside, s, -torch.inf)
        sel = torch.zeros_like(inside)
        sel[s.argsort(descending=True)[: math.ceil(self.quota * int(inside.sum()))]] = True
        return sel & inside

    def _dense_pred(self, base: torch.Tensor, g8, gf, par, keep, fine: torch.Tensor):
        """Predicted classes on the finest grid: each fine cell its 8 mm parent's, the refined ones their own."""
        from ..curves import grid_coords
        dev = base.device
        xyz = grid_coords(torch.tensor(gf, device=dev)[None], math.prod(gf))[0]
        p = torch.minimum(xyz // 2, torch.tensor(g8, device=dev) - 1)
        out = base[(p[:, 0] * g8[1] + p[:, 1]) * g8[2] + p[:, 2]].clone()
        out[keep] = fine
        return out

    def matryoshka_sums(self, x: torch.Tensor, y: torch.Tensor, ctx: torch.Tensor, n_out: int, unsure: float = 0.05,
                        rand: float | None = None, edges: float = 0.1) -> tuple:
        """A batch of tiles -> (sums of the final prediction, [sums of the 16, 8, 4 mm heads]), each summed over the
        batch (segrun.sums_loss on each, added). Final: the 8 mm head where the 4 mm level is not read, the 4 mm head in
        its cells, the native voxels decoded in a subset of them (GT edges, predicted edges, unsure, random)."""
        if self.decoder_c:
            return self.c_sums(x, y, ctx, n_out)
        st, V, sp = self(x, ctx, levels=True, stop=1)
        rand = self.rand_dec if rand is None else rand
        add = lambda a, b: b if a is None else [u + v for u, v in zip(a, b)]
        fin, aux = None, [None, None, None]
        for b, (z8, k8, g8, emb8, cur) in enumerate(st):
            z16, k16, _ = self._z16[b]
            yy, vol = y[b, : V[b][0], : V[b][1], : V[b][2]], x[b, 0, : V[b][0], : V[b][1], : V[b][2]]
            lab16, val16 = self._cells(yy, k16)
            aux[0] = add(aux[0], self._cell_sums(self.head16(z16), self._hist(lab16, val16, n_out)))
            lab8, val8 = self._cells(yy, k8)
            h8, c8 = self._hist(lab8, val8, n_out), self.head8(z8)
            aux[1] = add(aux[1], self._cell_sums(c8, h8))
            self._cur = cur
            four = fine_ok(self.fine_dims(sp[b], 1))
            sel8 = self._select8(vol, k8, val8.any(1)) if four else torch.zeros(len(z8), dtype=torch.bool, device=z8.device)
            f = self._cell_sums(c8[~sel8], h8[~sel8])
            if sel8.any():
                z, k, g, keep, par = self._sparse4(z8, vol, sp[b], k8, g8, sel8)
                lab, val = self._cells(yy, k)
                lab, val = lab[keep], val[keep]
                h, c = self._hist(lab, val, n_out), self.head4(z)
                aux[2] = add(aux[2], self._cell_sums(c, h))
                emb, grid = self.fine[1][0].patch_embed, g
            else:   # no 4 mm level (coarse voxels): decode at 8 mm
                z, k, lab, val, h, c, emb = z8, k8, lab8, val8, h8, c8, emb8
                f = [0 * t for t in f]
            u = 1 - c.detach().float().softmax(1).max(1).values
            r, r2 = torch.rand(len(u), device=z.device), torch.rand(len(u), device=z.device)
            edge = (h > 0).sum(1) > 1
            if sel8.any():
                dense = self._dense_pred(c8.detach().argmax(1), g8, grid, par, keep, c.detach().argmax(1))
                pe = self._pred_edge_lab(dense, grid)[keep]
            else:
                pe = self._pred_edge(c.detach(), g8)
            dsel = (edge & (r < edges * len(u) / max(int(edge.sum()), 1))) | (u >= u.quantile(1 - unsure)) \
                | (r > 1 - rand) | (pe & (r2 < edges * len(u) / max(int(pe.sum()), 1)))
            f = add(f, self._cell_sums(c[~dsel], h[~dsel]))
            if dsel.any():
                cells = keep if sel8.any() else torch.arange(len(z), device=z.device)
                lg = self.decode_cells(z[dsel], cells[dsel], vol, sp[b], k, grid if sel8.any() else g8, emb)
                lg = lg.flatten(2).transpose(1, 2)
                f = add(f, sums(lg[val[dsel]], lab[dsel][val[dsel]], 1.0, n_out))
            fin = add(fin, f)
            self.last_refined = (float(sel8.float().mean()), float(dsel.float().mean()))
        return fin, [a for a in aux if a is not None]

    @staticmethod
    def _pred_edge_lab(lab: torch.Tensor, g) -> torch.Tensor:
        a = lab.float().view(1, 1, *g)
        return (F.max_pool3d(a, 3, 1, 1) != -F.max_pool3d(-a, 3, 1, 1)).flatten()

    @torch.no_grad()
    def matryoshka_predict(self, x: torch.Tensor, ctx: torch.Tensor, n_out: int) -> torch.Tensor:
        """A whole scan [C, *S] in one pass -> probabilities [n_out, *S]: 16 -> 8 mm on the whole scan, the 8 mm head
        everywhere, the 4 mm level in the quota of cells with the most varying intensities (its head there), the native
        voxels decoded in the 4 mm cells on the predicted boundary or unsure (max probability < tau)."""
        if self.decoder_c:
            return self.c_predict(x, ctx, n_out)
        dev, S = x.device, tuple(x.shape[1:])
        c = torch.cat([ctx[:7].float().cpu(), torch.zeros(3)]).to(dev)
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            st, _, sp = self(x[None].float(), c[None], levels=True, stop=1)
            z8, k8, g8, emb8, cur = st[0]
            vol, p8 = x[0], self.head8(z8).float().softmax(1)
            self._cur = cur
            four = fine_ok(self.fine_dims(sp[0], 1))
            inside = (patchify_any(vol, k8).abs().sum(1) > 0)[: len(z8)]
            if four:
                sel8 = self._select8(vol, k8, inside)
                z, k, g, keep, par = self._sparse4(z8, vol, sp[0], k8, g8, sel8)
                p = self.head4(z).float().softmax(1)
                emb = self.fine[1][0].patch_embed
                base = self._dense_pred(p8.argmax(1), g8, g, par, keep, p.argmax(1))
                pe = self._pred_edge_lab(base, g)[keep]
            else:
                z, k, g, p, emb = z8, k8, g8, p8, emb8
                keep = torch.arange(len(z8), device=dev)
                pe = self._pred_edge(p8, g8)
        finally:
            self.fine_seg = fs
        dsel = pe | (p.max(1).values < self.tau) | self.test_all
        P, N = math.prod(k), math.prod(g)
        fits = n_out * N * P * 2 <= torch.cuda.mem_get_info(dev)[0] / 4
        from ..curves import grid_coords
        xyz = grid_coords(torch.tensor(g, device=dev)[None], N)[0]
        q = torch.minimum(xyz * torch.tensor(k, device=dev) // torch.tensor(k8, device=dev), torch.tensor(g8, device=dev) - 1)
        pr = p8[(q[:, 0] * g8[1] + q[:, 1]) * g8[2] + q[:, 2]].half().to(dev if fits else "cpu")
        pr[keep.to(pr.device)] = p.half().to(pr.device)
        pr = pr[:, :, None].expand(-1, -1, P).contiguous()
        for idx in dsel.nonzero()[:, 0].split(1 << 15):
            if len(idx):
                lg = self.decode_cells(z[idx], keep[idx], vol, sp[0], k, g, emb)
                pr[keep[idx].to(pr.device)] = lg.float().softmax(1).flatten(2).half().to(pr.device)
        self.last_refined = float(dsel.float().mean())
        out = pr.view(*g, n_out, *k).permute(3, 0, 4, 1, 5, 2, 6).reshape(n_out, *(a * b for a, b in zip(g, k)))
        return out[:, : S[0], : S[1], : S[2]]


def _c_points(yy: torch.Tensor, k: int, n: int, gen=None) -> tuple:
    """Stratified voxels of a label tile (255 ignored): strata class x boundary (a 6-neighbour differs), n / (2 k) per
    stratum, uniform within it -> flat indices [P], weights [P] (voxels each point stands for: unbiased sums)."""
    dev, flat = yy.device, yy.flatten()
    a = yy.float()[None, None]
    edge = (F.max_pool3d(a, 3, 1, 1) != -F.max_pool3d(-a, 3, 1, 1)).flatten()
    m, N = max(1, n // (2 * k)), flat.numel()
    idx, w = [], []
    for c in range(k):
        for e in (False, True):
            mask = (flat == c) & (edge == e)
            cnt = int(mask.sum())
            if not cnt:
                continue
            take = min(cnt, m)
            if cnt / N > 0.01:   # large stratum: uniform draws, kept if inside (no nonzero of ~1e8 voxels)
                cand = torch.randint(N, (int(4 * take * N / cnt) + 16,), device=dev, generator=gen)
                cand = cand[mask[cand]][:take]
            else:
                nz = mask.nonzero()[:, 0]
                cand = nz[torch.randperm(cnt, device=dev, generator=gen)[:take]]
            idx.append(cand)
            w.append(torch.full((len(cand),), cnt / max(len(cand), 1), device=dev))
    return torch.cat(idx), torch.cat(w)


def _c_field(f: torch.Tensor, g, k, pts: torch.Tensor) -> torch.Tensor:
    """Tokens' features f [N, C] on grid g of k-voxel cells -> the continuous field at voxels pts [P, 3]: quadratic
    B-spline weights over the 27 nearest cell centres (partition of unity, C1 across cell borders)."""
    dev = f.device
    kk, gg = torch.tensor(k, device=dev, dtype=torch.float), torch.tensor(g, device=dev)
    u = (pts.float() + 0.5) / kk - 0.5                     # position in cell-centre units
    c = u.round()
    d = u - c                                              # [-0.5, 0.5]
    wts = torch.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2], -1)   # [P, 3 axes, 3]
    out = 0
    for i in range(3):
        for j in range(3):
            for l in range(3):
                n = (c.long() + torch.tensor([i - 1, j - 1, l - 1], device=dev)).clamp_min(0)
                n = torch.minimum(n, gg - 1)
                ww = wts[:, 0, i] * wts[:, 1, j] * wts[:, 2, l]
                out = out + ww[:, None] * f[(n[:, 0] * g[1] + n[:, 1]) * g[2] + n[:, 2]]
    return out


def _c_image(vol: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """The voxel's intensity and its 6 face neighbours' (clamped at the border) -> [P, 7]."""
    S = torch.tensor(vol.shape, device=vol.device)
    off = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device=vol.device)
    q = torch.minimum((pts[:, None] + off[None]).clamp_min(0), S - 1)
    return vol[q[..., 0], q[..., 1], q[..., 2]].float()


def windows(vol, k, m, idx) -> torch.Tensor:
    """Cells idx (raster on the grid of k-voxel cells, as patchify) with m voxels of margin per side (zeros past the
    scan) -> [n, prod(k + 2m)]."""
    g = [-(-n // a) for n, a in zip(vol.shape, k)]
    v = F.pad(vol.float(), [p for a, b, n in zip(reversed(k), reversed(m), reversed(vol.shape)) for p in (b, b + (-n % a))])
    w = v.unfold(0, k[0] + 2 * m[0], k[0]).unfold(1, k[1] + 2 * m[1], k[1]).unfold(2, k[2] + 2 * m[2], k[2])
    i = torch.stack([idx // (g[1] * g[2]), idx // g[2] % g[1], idx % g[2]], 1)
    return w[i[:, 0], i[:, 1], i[:, 2]].reshape(len(idx), -1)


def patchify_any(vol, k):
    from ..data.volume import patchify
    return patchify(vol.float(), k)

def wsums(logits: torch.Tensor, lab: torch.Tensor, w: torch.Tensor, k: int) -> list:
    """sums() with a weight per point (decoder C's importance weights)."""
    lp = logits.float().log_softmax(1)
    p, oh = lp.exp(), F.one_hot(lab, k).float()
    return [-(lp.gather(1, lab[:, None])[:, 0] * w).sum(), w.sum(), (w[:, None] * p * oh).sum(0),
            (w[:, None] * p).sum(0), (w[:, None] * oh).sum(0)]


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
                     refine_tau=float(self.args.get("refine_tau") or 0.9), matryoshka=bool(self.args.get("matryoshka")),
                     quota=float(self.args.get("refine_quota") or 0.5), head_w=float(self.args.get("head_w") or 1.0),
                     rand_dec=float(self.args.get("rand_dec") or 0.02), test_all=bool(self.args.get("test_decode_all")),
                     kernel_skip=int(self.args.get("kernel_skip") or 0),
                     kernel_skip_mm=float(self.args.get("kernel_skip_mm") or 0.0), level2=bool(self.args.get("level2")),
                     decoder_c=bool(self.args.get("decoder_c")), c_points=int(self.args.get("c_points") or 32768))
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
