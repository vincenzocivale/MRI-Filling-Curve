"""Our encoder (LeJEPA, causal GDN-2) as a benchmark model: the same `<out>/<id>.pt` as the external FMs, so
bench-extract / bench-probe / bench-segdec treat it like any of them. Runs in our own env (sfc-gdn2), `repo` = this repo.

- preprocessing: the pretraining one (data/volume.py): canonical orientation (axis permutation / flips only), native
  voxels, values / 99th percentile of the voxels > 0, patches of round(patch_mm / spacing) voxels per axis.
- features: model.encoder_extractor (the read of every evaluation): the whole patch grid along each curve of
  `args.curves` (default: the run's view_curves, averaged), forward + reversed. `map` = bi tokens on the patch grid
  [2d, gx, gy, gz]; `mean` = bi averaged over the foreground patches (the pyramid top, what volume probes read:
  canonical). Below it (multigrain, 27/10: the only reading under a head; args.multigrain false: off) 8 mm dense and
  4 / 2 mm in the most varied cells, g<mm>:mean / std per block, concatenated by bench.probe (see `multigrain`).
- meta: `source_to_input` (source voxel index -> canonical voxel index), returned by `seg_input`.
- segmentation (bench-segdec, fm/segrun.py): the frozen encoder + our decoder Dmix (`SegNet`: a coarse-to-fine cascade
  16 -> 8 -> 4 -> 2 mm read only under mixed cells, a point head at the native voxels; slurm/bench_full.sh `DM`).
  Input: the native canonical grid (`seg_input`), the spacing as `seg_context`; training tiles from the token cache's
  16 mm grid (SEG_PATCH voxels without one), whole scans at val / test.

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

from .base import Image, Wrapper
from .token_cache import LiveCache, TokenCache, fine_ok

SEG_PATCH = (160, 160, 160)   # voxels, as the Primus / nnFoundation ViT runs
MIX_MAX = 1 << 19   # tokens per refine-block call (fine_mix)
# decoder args of the variants removed on 27/10 (Primus' PatchDecode, decoder F / kernel decoding, adaptive, matryoshka
# without the cascade, decoder C without it, decoder D, zoom, level2, kernel_skip, block_plan ...): an error, not ignored
SEG_GONE = ("context", "tta", "neighbors", "dec_cpe", "kernel", "rough", "zoom", "fine_seg", "fine_mm", "adaptive",
            "test_decode_all", "kernel_skip", "kernel_skip_mm", "level2", "c_unweighted", "blocks", "live_curves",
            "cache_curves")


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
    """Our segmentation decoder, Dmix (22-27/10, notes/decoder_cascade.md; the other decoders removed 27/10): the frozen
    `encoder` read along `curves` at 16 mm (live, or its cached tokens), then a cascade of finer levels (8 / 4 / 2 mm,
    halvings of the patch) read only under the cells still mixed, each level's children = parent token + the frozen
    encoder's reading at that level's patch + the voxels' KernelEmbed, mixed along `fine_curves` by 2-block GDN-2
    refiners; a point head (decoder C2) on the voxels of the cells still mixed at g_min."""

    def __init__(self, encoder, curves: list[str], seed: int, patch_mm: float, n_out: int, refine: int = 3,
                 image: bool = False, fine_curves: list | None = None, erase: str = "channel", fine_encoder: bool = False,
                 token_cache: str | None = None, head_w: float = 1.0, c_points: int = 32768, c_local: bool = False,
                 cascade_g0: float = 16.0, cascade_gmin: float = 4.0, cascade_r: float = 0.995, cascade_eps: float = 0.01,
                 cascade_rare: float = 1e-3, cascade_neg: float = 2.0, cascade_topk: int = 2, fine_split: bool = False,
                 cascade_warm: int = 0, cascade_topk_eval=(8, 8, 16, 16), fine_live4: bool = False,
                 cascade_rank: bool = False, c_patch: bool = False, cascade_ring: bool = False):
        super().__init__()
        from ..model import Encoder, encoder_extractor

        # refine 3 (22/10 sera, user): a 2 mm level, read by the frozen encoder live under the mixed 4 mm cells (the token
        # cache holds 16 / 8 (/ 4) mm) and kept sparse (a dense 2 mm grid of a 0.5 mm scan: ~1.6e7 cells)
        assert refine in (2, 3), "SegNet (Dmix): refine 2 / 3"
        self.cascade = True   # segrun's switch to cascade_step / cascade_predict / cascade_val (externals lack it)
        self.encoder, self.patch_mm = encoder, patch_mm
        self.unique = encoder.bidir != "late"  # one stream: bi = h ++ h, the decoder takes h once
        width = encoder.dim * (1 + len(encoder.taps)) * (1 if self.unique else 2)
        # more information, not a bigger decoder (user, 16/10): a wider input is first projected to d per token
        self.proj = nn.Linear(width, encoder.dim) if width > encoder.dim else nn.Identity()
        c = min(width, encoder.dim)
        self.refine, self.image, self.seed = int(refine), image, seed
        self.fine_seg = 0   # 0 (training): one sequence per tile; the whole-scan passes cut them every 65536 tokens
        self.fine_mm = [patch_mm / 2 ** i for i in range(1, int(refine) + 1)]   # the refine levels' patch (mm)
        self.fine_curves = list(fine_curves or ["hilbert"])
        self.fine = nn.ModuleList(nn.ModuleList(Encoder(128, 2, 64, 2, bidir="alternate", patch_mm=self.fine_mm[i],
                                                        freq_decay=1.0) for _ in self.fine_curves)
                                  for i in range(self.refine))
        if erase == "head":   # HeadErase: refine blocks whose state cannot diverge
            for m in self.fine.modules():
                if hasattr(m, "b_proj") and hasattr(m, "head_k_dim"):
                    m.b_proj = HeadErase(m)
        self.up = nn.ModuleList([nn.Linear(c, 128)] + [nn.Linear(128, 128) for _ in range(self.refine - 1)])
        # fine_encoder (18/10): each fine level also gets the frozen encoder itself run at that level's patch (a model
        # trained at several resolutions reads every scale: the same tokens); seg_net makes them read last ++ taps directly
        self.fine_tok = nn.ModuleList(nn.Linear(c, 128) for _ in range(self.refine)) if fine_encoder else None
        self.enc_curves = list(curves)
        # decoder C2 (22/10, notes/adaptive_decoding.md "decoder C"): the levels' tokens spread as continuous fields
        # (quadratic B-splines over the 27 nearest cells: smooth across cell borders, nothing resampled) and a point head:
        # per voxel [each level's field; c_local: + the containing cell's own features and the voxel's position relative
        # to its centre] + the voxel's and its neighbours' intensities (+-1, c_local also +-2 voxels per axis) -> logits
        self.c_points, self.c_local = int(c_points), bool(c_local)
        CF = 32
        self.cfield = nn.ModuleList([nn.Linear(encoder.dim, CF)] + [nn.Linear(128, CF) for _ in range(max(2, self.refine))])
        L = len(self.cfield)
        # c_patch (1b, 27/10: 95 % of the cochlea's errors 1 voxel off the boundary, the CMB false positives far from any
        # lesion): the point head sees the voxels' intensities in two 5^3 cubes (stride 1 and 2) through a learned
        # linear map (a 5^3 convolution at the point: the U-Net's full-resolution stage), not 13 values along the axes
        self.cpatch = nn.Sequential(nn.Linear(250, 64), nn.GELU()) if c_patch else None
        img = 64 if c_patch else 13 if self.c_local else 7
        self.chead = nn.Sequential(nn.Linear((L * (2 * CF + 3) if self.c_local else L * CF) + img, 128),
                                   nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                   nn.Linear(128, n_out))
        self.extract = encoder_extractor(encoder, curves, "patch", seed, 0.0)   # the live 16 mm reading (no token cache)
        # token_cache (20/10, fm/token_cache.py): the encoder's tokens of the whole scan read once on nested grids; tiles
        # start on the 16 mm grid (segrun aligns them). "live:<folder>" (24/10, user): LiveCache, the volumes the folder
        # lacks read in the job and written to it
        self.cache = (LiveCache(token_cache[5:], self) if str(token_cache).startswith("live:") else TokenCache(token_cache)) \
            if token_cache else None
        self._cur, self.width = None, width
        self.head_w = float(head_w)   # the level heads' loss weight
        # cascade (22/10, notes/decoder_cascade.md, variant A): coarse to fine. Per level a cell head [pure class k |
        # mixed]; pure cells are filled, only the mixed ones' children are read (by the frozen encoder live, or from the
        # token cache) with a ring of one neighbouring cell of context, down to g_min; the point head runs only in the
        # cells still mixed there. "Mixed": more than eps of the cell off its majority class, or any voxel of a rare class
        # (voxel frequency < cascade_rare on the training tiles). Each level's threshold is set on val (cascade_val) so
        # that every foreground class keeps a share r of its voxels per level.
        lvl = [patch_mm] + self.fine_mm
        self.cj0, self.cjmin = lvl.index(float(cascade_g0)), lvl.index(float(cascade_gmin))
        self.cr, self.ceps, self.crare, self.cneg = float(cascade_r), float(cascade_eps), float(cascade_rare), float(cascade_neg)
        # landmarks (23/10, user: "si legge solo 8 celle a 2 mm"): segrun sets self.landmark; per level and landmark the
        # head's top cascade_topk cells (+ the true one in training) go down, the point head maps the last ones
        self.ctopk = int(cascade_topk)
        # cascade_topk_eval (24/10, user): the cells per landmark at inference, per level from g0 (the last value for the
        # finer ones); p19 val, rank of the true cell: 16 mm <= 6, 8 mm 95% <= 6, 4 / 2 mm 90% <= 27 / 18 (the ring
        # catches most of the rest). With 2 a cell 7th at 8 mm sent a landmark 37 mm away. segrun scales it on the
        # val MRE of the best state; training keeps cascade_topk + the true cell
        self.ctopk_eval = [int(t) for t in (cascade_topk_eval if isinstance(cascade_topk_eval, (list, tuple))
                                            else str(cascade_topk_eval).split(","))]
        # cascade_warm W (23/10, user): the first W training iterations (and the val passes among them) stop at 4 mm;
        # the 2 mm level is read once the heads choose the cells (an untrained head marks nearly every cell mixed)
        self.cwarm = int(cascade_warm)
        if self.cwarm:   # training calls, saved (a resume keeps it); only then: older saved states have no such key
            self.register_buffer("it_seen", torch.zeros((), dtype=torch.long))
        self.lhead = nn.ModuleList([nn.Linear(encoder.dim, n_out + 1)] + [nn.Linear(128, n_out + 1) for _ in range(refine)])
        # buffers, saved with the trained state: the levels' thresholds (set on val), the training tiles' voxel counts
        # per class (rare classes) and cell-target counts (class weights, frequency^-0.5)
        self.register_buffer("taus", torch.full((refine + 1,), 0.0))   # logit margins (0: probability 0.5)
        self.register_buffer("cfreq", torch.zeros(n_out))
        self.register_buffer("tcount", torch.zeros(n_out + 1))
        # cascade_rank (27/10, user: decoder Delta's test, paired with Dmix in one run): beside the mixed head, a score
        # per cell trained with ListNet on the gain of reading it further (class-weighted voxels outside its majority,
        # V. Civale's split ranker, 2fd5e4e) on detached tokens, so Dmix trains as before; the val pass goes down twice,
        # with the mixed head's margins and with the scores (own thresholds rtaus, the same calibration)
        self.crank, self.cuse = bool(cascade_rank), "mixed"
        # cascade_ring (1a, 27/10: 79 % of the CMB misses and half the cochlea's in cells filled whole): the cells next to
        # a mixed one go down too, never filled at that level (PointRend: refine the uncertain cells and their neighbours)
        self.cring = bool(cascade_ring)
        if self.crank:
            mk = lambda d: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 1))
            self.rhead = nn.ModuleList([mk(encoder.dim)] + [mk(128) for _ in range(refine)])
            self.register_buffer("rtaus", torch.full((refine + 1,), 0.0))
            self._rloss = None
        # fine_split (23/10, user): the encoder's live 2 mm reading as one sequence per connected group of the cells read
        # (no jump between far regions read as neighbours along one sequence)
        self.fine_split = bool(fine_split)
        # matryoshka's cell heads (21/10): not used by the cascade, kept so that saved Dmix states (.best.pt / .last.pt)
        # still load
        self.head16, self.head8, self.head4 = nn.Linear(encoder.dim, n_out), nn.Linear(128, n_out), nn.Linear(128, n_out)
        # fine_live4 (24/10, user: C2h dropped, the cascade reads 4 mm only under its chosen cells): the 4 mm level read by
        # the frozen encoder live like 2 mm, the cache holding 16 / 8 mm only (no 4 mm files)
        self.ncached = 1 if fine_live4 else 2
        if self.cache is not None and self.fine_tok is not None:
            self.cache.levels = (16.0, 8.0, 4.0)[: 1 + self.ncached]

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        """x [B, 1, *T] native voxels, spacing [B, 3] mm, or [B, 6] = spacing + the scan's shape (segrun `seg_extent`),
        or [B, 10] = + the volume's index in the token cache and the tile's corner -> per tile its 16 mm tokens
        (z [N, d], cell dims k, grid g, cache key or None), each tile's part of the scan V, spacing [B, 3]. Only the
        scan's own voxels are read: segrun right-pads a tile up to the tile size (up to 1.2 m of zeros on our native
        grids, never seen in pretraining)."""
        from ..data.volume import patch_dims, patch_grid, patchify

        T = tuple(x.shape[2:])
        cached = self.cache is not None and spacing.shape[1] >= 10   # spacing, shape, volume, corner
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
            bi = [self.cache.tokens(i, 16.0, c, grid[b].tolist(), k)[:, : self.width].to(x.device).float()
                  for b, (i, c, k) in enumerate(zip(ids, corners, ks))]   # the cache holds last ++ taps
        else:
            ks = [patch_dims(s, self.patch_mm) for s in spacing.tolist()]
            grid = [patch_grid(v, k).to(x.device) for v, k in zip(V, ks)]
            with torch.no_grad():   # one tile at a time
                bi = [self.extract({"patches": [patchify(xs[b], k)], "grid": grid[b][None],
                                    "k": torch.tensor([k], device=x.device), "spacing": spacing[b:b + 1].float()})["bi"][0]
                      for b, k in enumerate(ks)]
        st = []
        for b, k in enumerate(ks):
            g = grid[b].tolist()
            tok = bi[b][: int(grid[b].prod())]
            if self.unique and not cached:   # the cache holds h once
                tok = tok[:, : tok.shape[1] // 2]
            z = self.proj(tok).T.reshape(1, -1, *g)
            st.append((z[0].flatten(1).T, k, g, (ids[b], corners[b], k4s[b]) if cached else None))
        return st, V, spacing

    def fine_dims(self, sp, i) -> list:
        """Voxels per axis of a level-i fine cell: the cache's nested grids (k4 x 2^(1-i)), else the level's patch."""
        from ..data.volume import patch_dims
        if self._cur:
            return [a * 2 ** (1 - i) for a in self._cur[2]] if i < 2 else [max(1, round(a / 2)) for a in self._cur[2]]
        return patch_dims(sp.tolist(), self.fine_mm[i])

    def _fine_prep(self, z, vol, sp, k, g, i, keep) -> dict:
        """Level i's inputs on the children `keep` (raster indices on the level's grid) of the parents z [N, c] on grid g,
        up to the frozen encoder's live reading: the parent (+ cached) tokens and what the reading takes."""
        from ..curves import grid_coords
        from ..data.volume import patch_grid, patchify

        kf = self.fine_dims(sp, i)
        gl = patch_grid(vol.shape, kf).tolist()
        gf = _dt(gl, z.device)
        xyz = grid_coords(gf[None], math.prod(gl))[0][keep]                              # [n, 3]
        par = ((xyz + 0.5) * _dt(kf, z.device) / _dt(k, z.device)).long()
        par = torch.minimum(par, _dt(g, z.device) - 1)
        pidx = (par[:, 0] * g[1] + par[:, 1]) * g[2] + par[:, 2]
        t = self.up[i](z)[pidx]
        d = {"xyz": xyz, "gf": gf, "kf": kf, "vol": vol, "sp": sp, "i": i, "keep": keep, "read": None}
        if self._cur is not None and self.fine_tok is not None and i < self.ncached:   # fine_encoder from the cache: the
            ci, corner, _ = self._cur                     # whole-scan reading of this level, last block ++ taps as at 16 mm
            tf = self.cache.tokens(ci, 8.0 if i == 0 else 4.0, corner, gl, kf).to(z.device).float()[keep]
            t = t + self.fine_tok[i](self._fproj(i, tf))
        elif self.fine_tok is not None:   # the frozen encoder at this level's patch, live (_read_many); every child (the
            sel = torch.arange(len(xyz), device=z.device)   # extractor's reading without its h ++ h copies and statistics)
            if len(sel):
                pe, mm0 = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm
                pe.patch_mm = self.fine_mm[i]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x = pe(patchify(vol.float(), kf)[keep][sel], pe.weights(sp.float(), kf))
                    q = xyz[sel]
                    lab = self._groups(q, gf) if self.fine_split and i == 2 else torch.zeros(len(sel), dtype=torch.long, device=z.device)
                pe.patch_mm = mm0
                d["read"] = (x, q, lab, sel)
        d["t"] = t
        return d

    def _read_many(self, ds: list, i: int) -> None:
        """The frozen encoder's live reading of level i for several tiles / scans at once (24/10: one call per curve with
        every tile's groups packed as sequences, not one per tile: the same tokens, the ~16-block encoder's Triton kernels
        launched once per level and batch) -> d["bf"] of each _fine_prep dict d with a reading."""
        from ..curves import keys
        rs = [d for d in ds if d["read"] is not None]
        if not rs:
            return
        pe, mm0 = self.encoder.patch_embed, self.encoder.patch_embed.patch_mm
        pe.patch_mm = self.fine_mm[i]
        cl = getattr(self, "live_curves", None) or self.enc_curves
        groups, n = [[]], 0   # calls of at most MIX_MAX tokens (whole tiles), as fine_mix
        for d in rs:
            if groups[-1] and n + len(d["read"][0]) > MIX_MAX:
                groups.append([])
                n = 0
            groups[-1].append(d)
            n += len(d["read"][0])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for grp in groups:
                bf = [0] * len(grp)
                for c in cl:   # within each group along the curve
                    xs, ps, os_, cu, n0 = [], [], [], [0], 0
                    for d in grp:
                        x, q, lab, _ = d["read"]
                        kk = keys(c, q, d["gf"], self.seed)
                        o = (lab * (int(kk.max()) + 1) + kk).argsort()
                        xs.append(x[o]), ps.append(q[o]), os_.append(o)
                        cu += (n0 + torch.bincount(lab).cumsum(0)).tolist()   # one sequence per group
                        n0 += len(x)
                    h = self.encoder.flat(torch.cat(xs), torch.tensor(cu, dtype=torch.int32, device=xs[0].device),
                                          pos=torch.cat(ps)).float()
                    for m, (hh, o) in enumerate(zip(h.split([len(o) for o in os_]), os_)):
                        bf[m] = bf[m] + hh[o.argsort()]
                for d, b_ in zip(grp, bf):
                    d["bf"] = b_ / len(cl)
        pe.patch_mm = mm0

    def _fine_done(self, d: dict) -> tuple:
        """_fine_prep after the live reading: + the encoder's tokens, + the voxels' KernelEmbed (image: the level's own
        refine kernel) -> t, xyz, gf, kf."""
        from ..data.volume import patchify
        t, i, kf, vol, sp, keep = d["t"], d["i"], d["kf"], d["vol"], d["sp"], d["keep"]
        if d["read"] is not None:
            t = t.index_add(0, d["read"][3], self.fine_tok[i](self._fproj(i, d["bf"])))
        if self.image:
            e = self.fine[i][0].patch_embed
            t = t + e(patchify(vol.float(), kf)[keep], e.weights(sp.float(), kf))
        return t, d["xyz"], d["gf"], kf

    def _fproj(self, i, t):
        """Level i's tokens for its fine_tok: last ++ taps as they are (fine_direct, seg_net), else through proj -> d."""
        return t if i in getattr(self, "fine_direct", ()) else self.proj(t)

    @staticmethod
    def _groups(q, gf) -> torch.Tensor:
        """Cells q [n, 3] on grid gf -> the label of each one's 26-connected group (0 ..), groups in raster order."""
        from scipy import ndimage
        g = [int(a) for a in gf.tolist()]
        m = np.zeros(g, dtype=bool)
        qc = q.cpu().numpy()
        m[qc[:, 0], qc[:, 1], qc[:, 2]] = True
        lab = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))[0]
        return torch.as_tensor(lab[qc[:, 0], qc[:, 1], qc[:, 2]].astype(np.int64) - 1, device=q.device)

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

    # ------------------------------------------------------------------------------- cells and their labels
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

    def c_logits(self, lv, vol, pts):
        """Point head at voxels pts [P, 3] of this tile / scan: the three levels' fields + the local intensities."""
        fs = []
        for j in range(len(self.cfield)):
            C = self.cfield[j].out_features
            if j < len(lv):
                z, k, g = lv[j]
                f = _Proj(z, self.cfield[j]) if isinstance(z, SparseLevel) else self.cfield[j](z.float())
                fs.append(_c_field(f, g, k, pts))
                if self.c_local:
                    fs += list(_c_local(f, g, k, pts))
            else:   # no 4 mm level (coarse voxels): zeros
                fs.append(torch.zeros(len(pts), (2 * C + 3) if self.c_local else C, device=pts.device))
        img = self.cpatch(torch.cat([_c_cube(vol, pts, 1), _c_cube(vol, pts, 2)], -1)) if self.cpatch is not None else \
            _c_image(vol, pts, 2 if self.c_local else 1)
        return self.chead(torch.cat(fs + [img], -1))

    # ------------------------------------------------------------------------- cascade decoding (22/10, variant A)
    def _targets(self, hist: torch.Tensor, n_out: int) -> torch.Tensor:
        """Cells' label counts [M, n_out] -> targets [M]: the pure class, or n_out (mixed)."""
        n, (mx, am) = hist.sum(1), hist.max(1)
        tot = self.cfreq.sum()
        rare = (self.cfreq / tot.clamp_min(1) < self.crare) & (tot > 0)
        rare[0] = False
        other_rare = ((hist > 0) & rare[None]).scatter(1, am[:, None], False).any(1)
        return torch.where((n - mx > self.ceps * n) | other_rare, n_out, am)

    def _level_sparse_many(self, items: list, i: int) -> list:
        """Level i's tokens on the dense grid of its cells for several tiles / scans, items [(z, k, g, vol, sp, read, cur)]:
        the children of the `read` parents read (the encoder live or cached + the refine blocks, those children only, one
        sequence along each curve), the others their parent's token through up[i]; the live readings and refine blocks of
        all items in one call each (packed sequences, the same tokens as one call per tile) -> per item tokens [Nf, 128]
        (level 2: a SparseLevel), dims, grid, parent index of every child."""
        from ..curves import grid_coords
        pre = []
        for zd, kp, gp, vol, sp, read, cur in items:
            self._cur = cur
            dev = zd.device
            kf = self.fine_dims(sp, i)
            gf = [-(-v // a) for v, a in zip(vol.shape, kf)]
            xyz = grid_coords(_dt(gf, dev)[None], math.prod(gf))[0]
            par = torch.minimum(((xyz + 0.5) * _dt(kf, dev) / _dt(kp, dev)).long(), _dt(gp, dev) - 1)
            pidx = (par[:, 0] * gp[1] + par[:, 1]) * gp[2] + par[:, 2]
            keep = read[pidx].nonzero()[:, 0]
            pre.append((kf, gf, pidx, keep, self._fine_prep(zd, vol, sp, kp, gp, i, keep=keep) if len(keep) else None))
        self._read_many([p[4] for p in pre if p[4] is not None], i)
        live = [n for n, p in enumerate(pre) if p[4] is not None]
        done = {n: self._fine_done(pre[n][4]) for n in live}
        tks = {}
        if live:
            tks = dict(zip(live, self.fine_mix([done[n][0] for n in live], [done[n][1] for n in live],
                                               [done[n][2] for n in live], i)))
        out = []
        for n, ((zd, *_), (kf, gf, pidx, keep, _)) in enumerate(zip(items, pre)):
            tk = tks.get(n)
            if i == 2:   # the last level (2 mm): only the read cells (their 27 neighbours are read too: the ring)
                out.append((SparseLevel(keep, tk if tk is not None else zd.new_zeros(0, 128), zd, pidx, self.up[i]), kf, gf, pidx))
                continue
            dense = self.up[i](zd)[pidx]
            if tk is not None:
                dense = dense.index_copy(0, keep, tk.to(dense.dtype))
            out.append((dense, kf, gf, pidx))
        return out

    def _jmin(self, sp) -> int:
        """The last level read: g_min, the finest level whose cells hold >= 8 voxels, and 4 mm during the warm-up."""
        L = 1 + sum(1 for i in range(self.refine) if i == 0 or fine_ok(self.fine_dims(sp, i)))
        j = min(self.cjmin, L - 1)
        return min(j, 2) if self.cwarm and int(self.it_seen) <= self.cwarm else j

    def _cstart(self, z16, k16, g16, vol, sp, cur, yy=None) -> dict:
        """A tile / scan's state at the top of the cascade (its 16 mm tokens)."""
        self._cur = cur
        jmin = self._jmin(sp)
        return {"vol": vol, "sp": sp, "cur": cur, "yy": yy, "j": 0, "jmin": jmin, "j0": min(self.cj0, jmin),
                "z": z16, "k": k16, "g": g16, "lv": [(z16, k16, g16)], "info": [], "fill": [], "S": None, "done": False,
                "cand": torch.ones(len(z16), dtype=torch.bool, device=z16.device)}

    def _cheads(self, st: dict, n_out: int) -> None:
        """The current level's head on its candidate cells (the cells without signal: background)."""
        self._cur = st["cur"]
        z, k, j = st["z"], st["k"], st["j"]
        cnt, hist = self._cell_counts(st, k, n_out)
        inside = (cnt > 0)[: len(z)]
        st["fill"].append((j, (st["cand"] & ~inside).nonzero()[:, 0], None))
        st["cand"], st["inside"] = st["cand"] & inside, inside
        st["hist"] = hist
        st["ci"] = st["lg"] = None
        if j >= st["j0"]:
            st["ci"] = st["cand"].nonzero()[:, 0]
            with torch.autocast(z.device.type, enabled=False):   # fp32 logits: bf16 ones tie the margins (12.0, 7.875)
                st["lg"] = self.lhead[j](z[st["ci"]].float())
            # confidence of the best pure class as a logit margin (its logit - logsumexp of the others), not a probability:
            # trained heads saturate the softmax at exactly 1.0 and a threshold of 1.0 then filled every tied cell past the
            # calibrated prefix (CMB, 24/10: val F1 0 from 3750 on, its lesion cells filled as background at 16 / 8 mm)
            lg = st["lg"].detach().float()
            st["a"] = lg[:, :n_out].argmax(1)
            st["s"] = lg.gather(1, st["a"][:, None])[:, 0] - lg.scatter(1, st["a"][:, None], float("-inf")).logsumexp(1)
            if getattr(self, "crank", False):   # the ranker's score (high: read further); "rank": the fill order is -score
                with torch.autocast(z.device.type, enabled=False):
                    st["r"] = self.rhead[j](z[st["ci"]].detach().float())[:, 0]
                if self.cuse == "rank":   # filled with the mixed head's best pure class, in the ranker's order
                    st["s"] = -st["r"].detach()

    def _cell_counts(self, st: dict, k, n_out: int) -> tuple:
        """Per cell of k voxels of the tile / scan: its non-zero voxels and its label counts [N, n_out] (None without
        labels). The 16 / 8 mm cells (4 k4, 2 k4: the cache's nested grids) are sums of the 4 mm ones, computed once
        per tile (27/10: every level patchified the whole tile again)."""
        memo, key = st.setdefault("counts", {}), tuple(int(a) for a in k)
        if key in memo:
            return memo[key]
        k4 = [int(a) for a in st["cur"][2]] if st["cur"] else None
        f = key[0] // k4[0] if k4 and key[0] % k4[0] == 0 else 0
        if f in (2, 4) and list(key) == [f * a for a in k4]:
            gf = [-(-v // a) for v, a in zip(st["vol"].shape, k4)]
            pool = lambda t: F.pad(t.view(*gf, -1), [0, 0] + [q for a in reversed(gf) for q in (0, -a % f)]) \
                .view(-(-gf[0] // f), f, -(-gf[1] // f), f, -(-gf[2] // f), f, -1).sum((1, 3, 5)).flatten(0, 2)
            c4, h4 = self._cell_counts(st, k4, n_out)
            memo[key] = (pool(c4[:, None].float())[:, 0], None if h4 is None else pool(h4))
        else:
            cnt = (patchify_any(st["vol"], k) != 0).sum(1)
            memo[key] = (cnt, None if st["yy"] is None else self._hist(*self._cells(st["yy"], k), n_out))
        return memo[key]

    def _capply(self, st: dict, n_out: int, train: bool = False, upto=None) -> None:
        """Pure / mixed with this level's threshold; the mixed cells' children read (the next level), or done."""
        self._capply_many([st], n_out, train, upto)

    def _capply_many(self, sts: list, n_out: int, train: bool = False, upto=None) -> None:
        """_capply of several tiles / scans at the same level, their next level read together (_level_sparse_many)."""
        go = [st for st in sts if self._cdecide(st, n_out, train, upto)]
        if not go:
            return
        res = self._level_sparse_many([(st["z"], st["k"], st["g"], st["vol"], st["sp"],
                                        self._dilate(st["S"], st["g"]) & st["inside"], st["cur"]) for st in go], go[0]["j"])
        for st, (zf, kf, gf, pidx) in zip(go, res):
            st["lv"].append((zf, list(kf), gf))
            st.update(cand=st["S"][pidx], z=zf, k=kf, g=gf, j=st["j"] + 1)

    def _cdecide(self, st: dict, n_out: int, train: bool = False, upto=None) -> bool:
        """_capply's choice: the level's pure / mixed cells (st["S"]: mixed) -> whether its children are read."""
        j, cand = st["j"], st["cand"]
        if st["ci"] is None:   # above g0: every cell goes down
            S = cand
        else:
            ci, hc = st["ci"], None if st["hist"] is None else st["hist"][st["ci"]]
            mix = st["s"] < (self.rtaus if getattr(self, "cuse", "mixed") == "rank" else self.taus)[j]
            if train:   # the true mixed cells + at most cneg x as many predicted mixed (hard negatives)
                tm = self._targets(hc, n_out) == n_out
                fp = mix & ~tm
                m = max(64, int(self.cneg * int(tm.sum())))
                if int(fp.sum()) > m:
                    fp &= torch.rand(len(fp), device=fp.device) < m / int(fp.sum())
                mix = tm | fp
            if not train and hc is not None and getattr(self, "oracle", False):   # segrun SFC_CASCADE_ORACLE: no
                mix = mix | (self._targets(hc, n_out) == n_out)                    # truly mixed cell is ever filled
            if self.cring and not train and mix.any():   # + the ring: the candidates next to a mixed cell (inference only:
                # in training it lifted the cneg cap on the cells read, 27 per isolated lesion cell at every level)
                D = torch.zeros_like(cand)
                D[ci[mix]] = True
                mix = (self._dilate(D, st["g"]) & cand)[ci]
            st["info"].append((j, ci, st["lg"], hc))
            st.setdefault("rinfo", []).append(st.get("r"))
            st["fill"].append((j, ci[~mix], st["a"][~mix]))
            S = torch.zeros_like(cand)
            S[ci[mix]] = True
        st["S"] = S
        if j == st["jmin"] or j == upto or not S.any():
            st["done"] = True
            return False
        return True

    def _cascade(self, z16, k16, g16, vol, sp, cur, n_out, yy=None, train=False) -> dict:
        """One tile / scan down the levels with the current thresholds -> its final state: lv (the levels' dense tokens,
        c_logits), info [(j, candidate cells, head logits, their label counts)], S (the cells still mixed at the last
        level read, dims k, grid g), fill [(j, cells, class or None = background)]."""
        return self._cascade_many([(z16, k16, g16, vol, sp, cur, yy)], n_out, train)[0]

    def _cascade_many(self, items: list, n_out: int, train: bool = False) -> list:
        """_cascade of several tiles [(z16, k16, g16, vol, sp, cur, yy)] level by level, each level read for all at once."""
        sts = [self._cstart(*it) for it in items]
        while not all(st["done"] for st in sts):
            act = [st for st in sts if not st["done"]]
            for st in act:
                self._cheads(st, n_out)
            self._capply_many(act, n_out, train)
        return sts

    def _whole(self, x, ctx):
        """A whole scan's 16 mm tokens (the tile is the scan) -> z, k, g, cur, spacing."""
        c = torch.cat([ctx[:7].float().cpu(), torch.zeros(3)]).to(x.device)
        st, _, sp = self(x[None].float(), c[None])
        z, k, g, cur = st[0]
        return z, k, g, cur, sp[0]

    def cascade_sums(self, x, y, ctx, n_out):
        """Training (encoder frozen): the level heads on their candidate cells (weighted CE + soft Dice, targets pure k /
        mixed, weights frequency^-0.5), the point head on stratified voxels of the cells still mixed at g_min (true
        mixed + predicted ones), unweighted."""
        parts = list(zip(x, y, ctx)) if isinstance(x, list) else [(x, y, ctx)]   # lists: tiles of several shapes
        tiles = []
        for xp, yp, cp in parts:
            if self.cwarm:
                self.it_seen += 1
            st, V, sp = self(xp, cp)
            for b, (z, k, g, cur) in enumerate(st):
                yy, vol = yp[b, : V[b][0], : V[b][1], : V[b][2]], xp[b, 0, : V[b][0], : V[b][1], : V[b][2]]
                ok = yy < n_out
                self.cfreq += torch.bincount(yy[ok].flatten().long(), minlength=n_out).float()
                tiles.append((z, k, g, vol, sp[b], cur, yy))
        # every tile of the batch down the levels together (24/10: one live reading per level, not per tile)
        add = lambda a, b: b if a is None else [u + v for u, v in zip(a, b)]
        fin, aux = None, [None] * len(self.lhead)
        rl, nl = 0.0, 0
        wc = torch.where(self.cfreq > 0, self.cfreq.clamp_min(1).rsqrt(), torch.zeros_like(self.cfreq))
        for (z, k, g, vol, _, cur, yy), c in zip(tiles, self._cascade_many(tiles, n_out, train=True)):
            lv, S, kl, gl = c["lv"], c["S"], c["k"], c["g"]
            for (j, ci, lg, hc), r in zip(c["info"], c.get("rinfo", [None] * len(c["info"]))):
                keep = hc.sum(1) > 0
                if r is not None and int(keep.sum()) > 1:   # ListNet: the cells' scores against their gains in the tile
                    hw = hc[keep].float() * wc
                    gain = hw.sum(1) - hw.max(1).values
                    if float(gain.sum()) > 0:
                        rl, nl = rl - (gain / gain.sum() * r[keep].float().log_softmax(0)).sum(), nl + 1
                t = self._targets(hc[keep], n_out)
                self.tcount += torch.bincount(t, minlength=n_out + 1).float()
                f = self.tcount / self.tcount.sum()
                w = f.clamp_min(1e-6) ** -0.5
                w = w / w[f > 0].mean()
                aux[j] = add(aux[j], wsums(lg[keep], t, w[t], n_out + 1))
            if S.any():
                idx, _ = _c_points(yy, n_out, self.c_points, region=_blocks(S, gl, kl, yy.shape))
                if len(idx):
                    Sh = yy.shape
                    pts = torch.stack([idx // (Sh[1] * Sh[2]), idx // Sh[2] % Sh[1], idx % Sh[2]], 1)
                    lab = yy.flatten()[idx].long()
                    fin = add(fin, wsums(self.c_logits(lv, vol, pts), lab, torch.ones(len(idx), device=pts.device), n_out))
            self.last_refined = float(S.float().mean())
        if fin is None:   # nothing mixed in the batch
            dev = parts[0][0].device
            fin = [torch.zeros((), device=dev), torch.zeros((), device=dev)] + [torch.zeros(n_out, device=dev)] * 3
        if getattr(self, "crank", False):   # segrun.cascade_step adds it to the loss (its own parameters, detached input)
            r0 = sum(p.sum() * 0 for p in self.rhead.parameters())   # every ranker parameter in the graph (DDP, unused levels)
            self._rloss = (self._rloss if self._rloss is not None else 0) + r0 + (rl / nl if nl else 0)
        return fin, [a for a in aux if a is not None]

    @torch.no_grad()
    def _cfinish(self, st: dict, n_out: int) -> torch.Tensor:
        """A scan's final state -> probabilities [n_out, *S]: the filled cells one-hot (each voxel the first level that
        decided it), the point head on the voxels of the cells still mixed."""
        x0, lv = st["vol"], st["lv"]
        dev, S = x0.device, tuple(x0.shape)
        lab = torch.full(S, -1, dtype=torch.long, device=dev)
        for j, cells, cls in st["fill"]:
            if not len(cells):
                continue
            kj, gj = lv[j][1], lv[j][2]
            cv = torch.full((math.prod(gj),), -1, dtype=torch.long, device=dev)
            cv[cells] = 0 if cls is None else cls
            bv = _blocks(cv, gj, kj, S)
            lab = torch.where((lab < 0) & (bv >= 0), bv, lab)
        N = math.prod(S)
        fits = n_out * N * 2 <= torch.cuda.mem_get_info(dev)[0] / 4
        out = torch.zeros(n_out, N, dtype=torch.half, device=dev if fits else "cpu")
        lf = lab.flatten()
        done = (lf >= 0).nonzero()[:, 0]
        out[lf[done].to(out.device), done.to(out.device)] = 1
        rest = (lf < 0).nonzero()[:, 0]   # the cells mixed at the last level read
        for i in rest.split(1 << 17 if self.cpatch is not None else 1 << 19):
            if len(i):
                pts = torch.stack([i // (S[1] * S[2]), i // S[2] % S[1], i % S[2]], 1)
                out[:, i.to(out.device)] = self.c_logits(lv, x0, pts).float().softmax(1).T.half().to(out.device)
        self.last_refined = float(len(rest) / N)
        return out.view(n_out, *S)

    @torch.no_grad()
    def cascade_predict(self, x, ctx, n_out):
        """A whole scan -> probabilities [n_out, *S] with the thresholds set on val."""
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            z, k, g, cur, sp = self._whole(x, ctx)
            return self._cfinish(self._cascade(z, k, g, x[0], sp, cur, n_out), n_out)
        finally:
            self.fine_seg = fs

    @torch.no_grad()
    def cascade_val(self, vols, n_out, amp) -> list:
        """The val pass (segrun's val volumes): every volume down the levels together, each level's threshold set from
        all of them before going on, then the predictions [*S] (argmax) for the val Dice: one pass, no separate
        calibration. Threshold: the lowest (fewest cells read further) at which every foreground class keeps a share r
        of its voxels at that level (lost: a cell filled with another class). States wait on the CPU between levels."""
        dev = self.taus.device
        mv = lambda o, d: o.to(d) if torch.is_tensor(o) or isinstance(o, SparseLevel) else \
            [mv(t, d) for t in o] if isinstance(o, (list, tuple)) else o
        tot = sum(torch.bincount(v["y"][v["y"] < n_out].flatten().long(), minlength=n_out).float() for v in vols).to(dev)
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            sts = []
            for v in vols:
                x = v["x"].to(dev).float()
                with amp():
                    z, k, g, cur, sp = self._whole(x, v["ctx"].to(dev))
                sts.append({a: mv(b, "cpu") for a, b in self._cstart(z, k, g, x[0], sp, cur, v["y"].to(dev)).items()})
            for j in range(self.cjmin + 1):
                live = [i for i, st in enumerate(sts) if not st["done"] and st["j"] == j]
                if not live:
                    break
                for i in live:
                    st = {a: mv(b, dev) for a, b in sts[i].items()}
                    with amp():
                        self._cheads(st, n_out)
                    sts[i] = {a: mv(b, "cpu") for a, b in st.items()}
                if j >= self.cj0:
                    ss = [sts[i]["s"] for i in live if sts[i]["ci"] is not None]
                    if ss:
                        s = torch.cat(ss).to(dev)
                        ls = torch.cat([sts[i]["hist"][sts[i]["ci"]].scatter(1, sts[i]["a"][:, None], 0.0)
                                        for i in live if sts[i]["ci"] is not None]).to(dev)
                        o = s.argsort(descending=True)
                        okp = (ls[o].cumsum(0)[:, 1:] <= (1 - self.cr) * tot[None, 1:]).all(1)
                        m = int(okp.long().cumprod(0).sum())   # the longest prefix of most confident cells that may be filled
                        taus = self.rtaus if getattr(self, "cuse", "mixed") == "rank" else self.taus
                        taus[j] = float(s[o[m - 1]]) if m else float("inf")
                        print(f"[cascade] {getattr(self, 'cuse', 'mixed')} level {j}: tau {float(taus[j]):.4f}, filled {m}/{len(s)} cells",
                              flush=True)
                for i in live:
                    st = {a: mv(b, dev) for a, b in sts[i].items()}
                    with amp():
                        self._capply(st, n_out)
                    sts[i] = {a: mv(b, "cpu") for a, b in st.items()}
            preds = []
            for st in sts:
                st = {a: mv(b, dev) for a, b in st.items()}
                with amp():
                    preds.append(self._cfinish(st, n_out).argmax(0).to(dev))
            return preds
        finally:
            self.fine_seg = fs

    # ------------------------------------------------------------- landmark cascade (23/10, heatmap regression)
    def _lm_levels(self, z, k, g, vol, sp, cur, n_out, pts=None):
        """A tile / scan down the levels for landmarks: at each level the head's logits [cells, landmarks] on the candidate
        cells, the top ctopk (training) / ctopk_eval[level] cells per landmark (+ the true cell, pts [landmarks, 3] voxels or NaN, in training) go down:
        their children are read with a ring of one cell. -> lv, [(logits, targets or None)], the last level's chosen cells
        (dense mask), its dims and grid."""
        return self._lm_levels_many([(z, k, g, vol, sp, cur, pts)], n_out)[0]

    def _lm_levels_many(self, items: list, n_out: int) -> list:
        """_lm_levels of several tiles [(z, k, g, vol, sp, cur, pts)] level by level, each level read for all at once."""
        sts = []
        for z, k, g, vol, sp, cur, pts in items:
            self._cur = cur
            inside = (patchify_any(vol, k).abs().sum(1) > 0)[: len(z)]
            sts.append({"z": z, "k": k, "g": g, "vol": vol, "sp": sp, "cur": cur, "pts": pts, "jmin": self._jmin(sp),
                        "lv": [(z, k, g)], "info": [], "cand": inside, "inside": inside, "out": None})
        j = 0
        while any(st["out"] is None for st in sts):
            go = []
            for st in sts:
                if st["out"] is not None:
                    continue
                z, k, g, pts = st["z"], st["k"], st["g"], st["pts"]
                ci = st["cand"].nonzero()[:, 0]
                lg = self.lhead[j](z[ci])[:, 1:n_out].float()
                sel = torch.zeros(len(z), dtype=torch.bool, device=ci.device)
                tgt = None
                if pts is not None:
                    kk, gg = _dt(k, ci.device), _dt(g, ci.device)
                    ok = ~pts.isnan().any(1)
                    cell = torch.minimum((pts[ok] // kk).long(), gg - 1)
                    cell = (cell[:, 0] * g[1] + cell[:, 1]) * g[2] + cell[:, 2]
                    row = torch.full((len(z),), -1, dtype=torch.long, device=ci.device)
                    row[ci] = torch.arange(len(ci), device=ci.device)
                    tgt = torch.zeros_like(lg)
                    for c, r in zip(ok.nonzero()[:, 0].tolist(), row[cell].tolist()):
                        if r >= 0:
                            tgt[r, c] = 1
                    sel[cell] = True
                tk = self.ctopk if pts is not None else self.ctopk_eval[min(j, len(self.ctopk_eval) - 1)]
                top = lg.detach().topk(min(tk, len(ci)), 0).indices.flatten()
                sel[ci[top]] = True
                st["info"].append((lg, tgt))
                if j == st["jmin"]:
                    st["out"] = (st["lv"], st["info"], sel, k, g)
                else:
                    go.append((st, sel))
            if go:
                res = self._level_sparse_many([(st["z"], st["k"], st["g"], st["vol"], st["sp"],
                                                self._dilate(sel, st["g"]) & st["inside"], st["cur"]) for st, sel in go], j)
                for (st, sel), (zf, kf, gf, pidx) in zip(go, res):
                    st["lv"].append((zf, list(kf), gf))
                    inside = (patchify_any(st["vol"], kf).abs().sum(1) > 0)[: len(pidx)]
                    st.update(cand=sel[pidx] & inside, inside=inside, z=zf, k=kf, g=gf)
            j += 1
        return [st["out"] for st in sts]

    def _lm_region(self, sel, k, g, S):
        """The voxels [P, 3] of the chosen last-level cells and their ring of one cell."""
        return _blocks(self._dilate(sel, g), g, k, S).nonzero()

    @staticmethod
    def _focal(logits, h):
        """CenterNet's penalty-reduced focal loss (segrun.focal_heatmap) of logits against heatmap targets h."""
        pos = h > 0.999
        lp, ln = F.logsigmoid(logits), F.logsigmoid(-logits)
        p = lp.exp()
        neg = (p ** 2 * (1 - h) ** 4 * ln)[~pos]
        if not pos.any():
            return -neg.mean()
        return -(((1 - p) ** 2 * lp)[pos].sum() + neg.sum()) / pos.sum()

    def landmark_sums(self, x, h, ctx, n_out):
        """Training on landmark heatmaps h [B, landmarks, *T] (segrun.heatmaps: 1 at each landmark's voxel): the level
        heads (balanced BCE: the cell holding the landmark against the other candidates) and the point head's focal loss
        on the voxels of the cells reached and their ring. x, h, ctx may be lists (tiles of several shapes): every tile
        of the batch goes down the levels together (24/10: one live reading per level, not per tile)."""
        parts = list(zip(x, h, ctx)) if isinstance(x, list) else [(x, h, ctx)]
        tiles, heat = [], []
        for xp, hp, cp in parts:
            if self.cwarm:
                self.it_seen += 1
            st, V, sp = self(xp, cp)
            for b, (z, k, g, cur) in enumerate(st):
                vol, hb = xp[b, 0, : V[b][0], : V[b][1], : V[b][2]], hp[b, :, : V[b][0], : V[b][1], : V[b][2]].float()
                pts = torch.full((hb.shape[0], 3), float("nan"), device=xp.device)
                for c in range(hb.shape[0]):
                    w = (hb[c] > 0.999).nonzero()
                    if len(w):
                        pts[c] = w[0].float()
                tiles.append((z, k, g, vol, sp[b], cur, pts))
                heat.append(hb)
        loss = 0.0
        for (_, _, _, vol, _, _, _), hb, (lv, info, sel, kl, gl) in zip(tiles, heat, self._lm_levels_many(tiles, n_out)):
            for lg, tgt in info:
                pos = tgt > 0
                ll = -F.logsigmoid(-lg)[~pos].mean()
                if pos.any():
                    ll = ll - F.logsigmoid(lg)[pos].mean()
                loss = loss + self.head_w * ll
            q = self._lm_region(sel, kl, gl, vol.shape)
            if len(q) > (1 << 16):
                q = q[torch.randperm(len(q), device=q.device)[: 1 << 16]]
            lg = self.c_logits(lv, vol, q)[:, 1:n_out].float()
            loss = loss + self._focal(lg, hb[:, q[:, 0], q[:, 1], q[:, 2]].T)
        return loss / len(tiles)

    @torch.no_grad()
    def landmark_predict(self, x, ctx, n_out):
        """A whole scan -> heatmaps [n_out, *S] (sigmoid; 0 outside the cells reached): segrun.landmark takes each map's
        maximum."""
        dev, S = x.device, tuple(x.shape[1:])
        fs, self.fine_seg = self.fine_seg, self.fine_seg or 1 << 16
        try:
            z, k, g, cur, sp = self._whole(x, ctx)
            lv, _, sel, kl, gl = self._lm_levels(z, k, g, x[0], sp, cur, n_out)
            out = torch.zeros(n_out, *S, dtype=torch.half, device=dev)
            q = self._lm_region(sel, kl, gl, S)
            for qq in q.split(1 << 18):
                out[1:, qq[:, 0], qq[:, 1], qq[:, 2]] = self.c_logits(lv, x[0], qq)[:, 1:n_out].float().sigmoid().T.half()
            self.last_refined = float(len(q) / math.prod(S))
            return out
        finally:
            self.fine_seg = fs

    def _hist(self, lab, val, n_out):
        return torch.zeros(len(lab), n_out, device=lab.device).scatter_add_(1, lab, val.float())


def _c_points(yy: torch.Tensor, k: int, n: int, gen=None, region: torch.Tensor | None = None) -> tuple:
    """Stratified voxels of a label tile (255 ignored): strata class x boundary (a 6-neighbour differs), n / (2 k) per
    stratum, uniform within it -> flat indices [P], weights [P] (voxels each point stands for: unbiased sums).
    region [*S] bool: only these voxels (the boundaries still those of the whole tile)."""
    dev, flat = yy.device, yy.flatten()
    a = yy.float()[None, None]
    edge = (F.max_pool3d(a, 3, 1, 1) != -F.max_pool3d(-a, 3, 1, 1)).flatten()
    if region is not None:
        flat = torch.where(region.flatten(), flat, torch.full_like(flat, 255))
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
    if not idx:
        return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, device=dev)
    return torch.cat(idx), torch.cat(w)


class SparseLevel:
    """A level's tokens kept only where read: rows `keep` (sorted dense indices on the level's grid) -> t [n, 128];
    any other cell its parent's token through `up` (parent tokens zp dense, pidx: every cell's parent). Indexed like
    the dense tensor (z[idx]), len = the grid's cells."""

    def __init__(self, keep, t, zp, pidx, up):
        self.keep, self.t, self.zp, self.pidx, self.up = keep, t, zp, pidx, up
        self.device = t.device

    def __len__(self):
        return len(self.pidx)

    def __getitem__(self, idx):
        pos = torch.searchsorted(self.keep, idx).clamp_max(max(len(self.keep) - 1, 0))
        hit = (self.keep[pos] == idx) if len(self.keep) else torch.zeros_like(idx, dtype=torch.bool)
        out = self.up(self.zp[self.pidx[idx]]).to(self.t.dtype) if not bool(hit.all()) else None
        if out is None:
            return self.t[pos]
        return torch.where(hit[:, None], self.t[pos] if len(self.t) else out, out) if len(self.t) else out

    def to(self, d):
        return SparseLevel(*(a.to(d) for a in (self.keep, self.t, self.zp, self.pidx)), self.up)


class _Proj:
    """cfield over a SparseLevel, gathered on demand (decoder C's fields index the cells they need)."""

    def __init__(self, z, lin):
        self.z, self.lin, self.device = z, lin, z.device

    def __getitem__(self, idx):
        return self.lin(self.z[idx].float())


_DEV = {}


def _dt(vals, dev, dtype=torch.long) -> torch.Tensor:
    """A small constant (cell dims, grid, offsets) on the device, made once: torch.tensor(list, device=cuda) is a
    synchronous pageable copy that drains the GPU queue (27/10 profile: ~115 per tile per iteration, ~25 % of the GPU
    time of the Dmix step, mostly _c_field's 27 offsets)."""
    key = (tuple(vals) if not isinstance(vals, torch.Tensor) else tuple(vals.tolist()), str(dev), dtype)
    if (t := _DEV.get(key)) is None:
        t = _DEV[key] = torch.tensor(key[0], device=dev, dtype=dtype)
    return t


_OFF27 = [v for i in range(3) for j in range(3) for l in range(3) for v in (i - 1, j - 1, l - 1)]


def _blocks(cell: torch.Tensor, g, k, S) -> torch.Tensor:
    """A per-cell array [prod(g)] on the grid g of k-voxel cells -> per voxel [*S] (each voxel its cell's value)."""
    v = cell.view(*g)
    for d in range(3):
        v = v.repeat_interleave(int(k[d]), d)
    return v[: S[0], : S[1], : S[2]]


def _c_field(f: torch.Tensor, g, k, pts: torch.Tensor) -> torch.Tensor:
    """Tokens' features f [N, C] on grid g of k-voxel cells -> the continuous field at voxels pts [P, 3]: quadratic
    B-spline weights over the 27 nearest cell centres (partition of unity, C1 across cell borders)."""
    dev = f.device
    kk, gg = _dt(k, dev, torch.float), _dt(g, dev)
    u = (pts.float() + 0.5) / kk - 0.5                     # position in cell-centre units
    c = u.round()
    d = u - c                                              # [-0.5, 0.5]
    wts = torch.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2], -1)   # [P, 3 axes, 3]
    off = _dt(_OFF27, dev).view(27, 3)
    out, cl = 0, c.long()
    for o in range(27):   # the 27 neighbours in turn (all at once: a [27, P, C] gather, too large at 2 mm)
        i, j, l = o // 9, o // 3 % 3, o % 3
        n = torch.minimum((cl + off[o]).clamp_min(0), gg - 1)
        ww = wts[:, 0, i] * wts[:, 1, j] * wts[:, 2, l]
        out = out + ww[:, None] * f[(n[:, 0] * g[1] + n[:, 1]) * g[2] + n[:, 2]]
    return out


def _c_local(f: torch.Tensor, g, k, pts: torch.Tensor) -> tuple:
    """The containing cell's features [P, C] and the voxel's offset from that cell's centre, in cells [P, 3]."""
    kk, gg = _dt(k, f.device, torch.float), _dt(g, f.device)
    u = (pts.float() + 0.5) / kk
    n = torch.minimum(u.floor().long(), gg - 1)
    return f[(n[:, 0] * g[1] + n[:, 1]) * g[2] + n[:, 2]], u - n.float() - 0.5


def _c_cube(vol: torch.Tensor, pts: torch.Tensor, step: int) -> torch.Tensor:
    """The 5^3 voxels around each point at stride `step` (clamped at the border) -> [P, 125]."""
    S = _dt(vol.shape, vol.device)
    off = _dt([step * (v - 2) for o in range(125) for v in (o // 25, o // 5 % 5, o % 5)], vol.device).view(125, 3)
    q = torch.minimum((pts[:, None] + off[None]).clamp_min(0), S - 1)
    return vol[q[..., 0], q[..., 1], q[..., 2]].float()


def _c_image(vol: torch.Tensor, pts: torch.Tensor, r: int = 1) -> torch.Tensor:
    """The voxel's intensity and the voxels at +-1 .. +-r along each axis (clamped at the border) -> [P, 1 + 6 r]."""
    S = _dt(vol.shape, vol.device)
    off = _dt([v for o in [[0, 0, 0]] + [[s * d * (a == 0), s * d * (a == 1), s * d * (a == 2)]
                                         for d in range(1, r + 1) for a in range(3) for s in (1, -1)] for v in o],
              vol.device).view(-1, 3)
    q = torch.minimum((pts[:, None] + off[None]).clamp_min(0), S - 1)
    return vol[q[..., 0], q[..., 1], q[..., 2]].float()


def patchify_any(vol, k):
    from ..data.volume import patchify
    return patchify(vol.float(), k)


def wsums(logits: torch.Tensor, lab: torch.Tensor, w: torch.Tensor, k: int) -> list:
    """Points' logits [M, k], labels [M] and weights [M] -> [CE sum, weight sum, per class: soft intersection, predicted
    sum, target sum] (segrun.sums_loss: CE / weights + 1 - mean soft Dice of the classes > 0)."""
    lp = logits.float().log_softmax(1)
    p, oh = lp.exp(), F.one_hot(lab, k).float()
    return [-(lp.gather(1, lab[:, None])[:, 0] * w).sum(), w.sum(), (w[:, None] * p * oh).sum(0),
            (w[:, None] * p).sum(0), (w[:, None] * oh).sum(0)]


FINE = (8.0, 4.0, 2.0)   # multigrain levels below the 16 mm extraction


def tricov(t: torch.Tensor) -> torch.Tensor:
    """Tokens [n, d] -> their covariance (population), upper triangle fp16: bench.probe.second_order's exact input."""
    t = t.float() - t.float().mean(0)
    c = t.T @ t / max(len(t), 1)
    i = torch.triu_indices(len(c), len(c), device=c.device)
    return c[i[0], i[1]].half()


@torch.no_grad()
def multigrain(enc, curves: list, seed: int, vol: torch.Tensor, sp: torch.Tensor, thr: float, above: dict,
               keep: dict | None = None) -> dict:
    """The features every head reads on our model (user, 27/10; segmentation decoders keep their own reading): below the
    16 mm extraction, 8 mm on every cell, then 4 and 2 mm only under the cells of the level above whose voxels vary most
    (intensity std of the foreground cells: label-free, the same for every task), as many cells as the level above has
    foreground cells (each level costs about the 8 mm one). The cells read as one sequence per 26-connected group along each curve
    (averaged), as the cascade's live reading. Per level the foreground mean and std of each block (last ++ taps, one
    copy of h) -> g<mm>:mean, g<mm>:std_b<k>, g<mm>:tricov_b<k>, ... (covariance); a level the scan cannot
    hold (cells under 8 voxels, nothing left) repeats the level above (`above`: the 16 mm ones, same names).
    bench.probe concatenates the levels."""
    from ..curves import grid_coords, keys
    from ..data.volume import patch_dims, patch_grid, patchify

    assert enc.bidir != "late", "multigrain: one-stream encoders (bi = h ++ h)"
    dev, d, pe, mm0 = vol.device, enc.dim, enc.patch_embed, enc.patch_embed.patch_mm
    names = ["", *(f"_b{t + 1}" for t in enc.taps)]
    out, prev, budget, kp, gp = {}, None, None, None, None   # prev: the level above's cell scores (-inf: not a parent)
    for mm in FINE:
        k = patch_dims(sp.tolist(), mm)
        g = _dt(patch_grid(vol.shape, k).tolist(), dev)
        p = patchify(vol, k)
        xyz = grid_coords(g[None], len(p))[0]
        if prev is None:   # 8 mm: every cell
            sel = torch.arange(len(p), device=dev)
        else:              # the children of the top parents, at most `budget` cells
            par = torch.minimum(((xyz + 0.5) * _dt(k, dev) / _dt(kp, dev)).long(), gp - 1)
            pidx = (par[:, 0] * gp[1] + par[:, 1]) * gp[2] + par[:, 2]
            order = prev.argsort(descending=True)
            n = torch.bincount(pidx, minlength=len(prev))[order].cumsum(0)
            top = order[(n <= budget) & (prev[order] > -torch.inf)]
            sel = torch.isin(pidx, top).nonzero()[:, 0]
        if not fine_ok(k) or not len(sel) or len(sel) > MIX_MAX:   # (whole-body scans at 8 mm: the level above's)
            prev = None
            break
        q, fg = xyz[sel], p[sel].mean(1) > thr
        nfg, fg = int(fg.sum()), fg if fg.any() else torch.ones_like(fg)
        lab = SegNet._groups(q, g) if len(sel) < len(p) else torch.zeros(len(sel), dtype=torch.long, device=dev)
        pe.patch_mm = mm
        h = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = pe(p[sel], pe.weights(sp, k))
            cu = torch.cat([lab.new_zeros(1), torch.bincount(lab).cumsum(0)]).int()
            for c in curves:
                kk = keys(c, q, g, seed)
                o = (lab * (int(kk.max()) + 1) + kk).argsort()
                h = h + enc.flat(x[o], cu, pos=q[o]).float()[o.argsort()]
        pe.patch_mm = mm0
        if keep is not None:   # fm/dense_dec: each level's cells read and their tokens (the decoder's multigrain input)
            keep[mm] = (q.short().cpu(), (h / len(curves)).half().cpu(), list(k))
        h = h[fg] / len(curves)
        for j, nm in enumerate(names):
            s, m = torch.std_mean(h[:, j * d: (j + 1) * d], 0, correction=0)
            out[f"g{int(mm)}:mean{nm}"], out[f"g{int(mm)}:std{nm}"] = m.half(), s.half()
            out[f"g{int(mm)}:tricov{nm}"] = tricov(h[:, j * d: (j + 1) * d])
        prev = torch.full((len(p),), -torch.inf, device=dev)
        prev[sel[fg]] = p[sel[fg]].float().std(1)
        budget, kp, gp = max(nfg, 1), k, g
    for mm in FINE:   # the levels not read: the level above's
        for nm, v in above.items():
            if f"g{int(mm)}:{nm}" not in out:
                out[f"g{int(mm)}:{nm}"] = out.get(f"g{int(mm) * 2}:{nm}", v)
    return out


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
        a = self.args
        if a.get("dense_dec"):   # fm/dense_dec (28/10): the dense control decoder on the cached 16 / 8 mm tokens
            from .dense_dec import DenseDec
            net = DenseDec(enc, str(a["token_cache"]).removeprefix("live:"), n_out, a.get("dense_mg"))
            net.cache.sig = {"checkpoint": str(self.ckpt) if pretrained else "random", "taps": list(enc.taps),
                             "curves": list(self.curves), "blocks": None}
            return net, ["encoder"], SEG_PATCH
        gone = [n for n in SEG_GONE if a.get(n)] + [f"{n}={a[n]}" for n, ok in (("encoder", None), ("fine_head", "mix"),
                                                                                  ("image_kernel", "own")) if a.get(n) not in (None, ok)]
        assert not gone, f"sfc segmentation: decoder args removed in the 27/10 cleanup (Dmix only): {gone}"
        assert all(a.get(n) for n in ("decoder_c", "matryoshka", "cascade")), \
            "sfc segmentation: only the Dmix cascade remains (args.decoder_c, args.matryoshka, args.cascade; slurm/bench_full.sh DM)"
        net = SegNet(enc, self.curves, int(self.run_cfg["seed"]), self.run_cfg["data"]["patch_mm"], n_out,
                     refine=int(a.get("refine") or 0), image=bool(a.get("image")), fine_curves=a.get("fine_curves"),
                     erase=a.get("erase") or "channel", fine_encoder=bool(a.get("fine_encoder")),
                     token_cache=a.get("token_cache"), head_w=float(a.get("head_w") or 1.0),
                     c_points=int(a.get("c_points") or 32768), c_local=bool(a.get("c_local")),
                     cascade_g0=float(a.get("cascade_g0") or 16.0), cascade_gmin=float(a.get("cascade_gmin") or 4.0),
                     cascade_r=float(a.get("cascade_r") or 0.995), cascade_eps=float(a.get("cascade_eps", 0.01)),
                     cascade_rare=float(a.get("cascade_rare", 1e-3)), cascade_neg=float(a.get("cascade_neg") or 2.0),
                     cascade_topk=int(a.get("cascade_topk") or 2), fine_split=bool(a.get("fine_split")),
                     cascade_warm=int(a.get("cascade_warm") or 0),
                     cascade_topk_eval=a.get("cascade_topk_eval") or (8, 8, 16, 16),
                     fine_live4=bool(a.get("fine_live4")), cascade_rank=bool(a.get("cascade_rank")),
                     c_patch=bool(a.get("c_patch")), cascade_ring=bool(a.get("cascade_ring")))
        if isinstance(net.cache, TokenCache):   # checked at its first read (segrun also builds a random net it never runs)
            net.cache.sig = {"checkpoint": str(self.ckpt) if pretrained else "random", "taps": list(enc.taps),
                             "curves": list(self.curves), "blocks": None}   # (blocks: always None since block_plan's removal)
        if enc.taps and getattr(net, "fine_tok", None) is not None:   # each finer level reads last ++ taps (the depth
            u = 1 if net.unique else 2        # quartiles concatenated as the probe, user 27/10) straight into its fine_tok (a
            net.fine_direct = set()           # projection to d first was a second linear map in a row: ~2M parameters for nothing)
            for i in range(len(net.fine_tok)):
                net.fine_tok[i] = nn.Linear((1 + len(enc.taps)) * enc.dim * u, 128)
                net.fine_direct.add(i)
        # the live levels (4 / 2 mm) read along one curve, not the mean of the run's curves (27/10 profile: the live
        # reading was ~half of the step, 3 curves x 16 blocks per level; the Dmix runs of 24/10 read one, block_plan's)
        net.live_curves = [a.get("live_curve") or "hilbert"]
        return net, ["encoder"], SEG_PATCH

    def extractor(self):
        if self._extract is None:
            from ..model import encoder_extractor

            gone = [a for a in ("tta", "cov_basis", "per_curve") if a in self.args]   # removed 27/10 (no probe read them)
            assert not gone, f"sfc extraction: args removed in the 27/10 cleanup: {gone}"
            enc = self._encoder().to(self.device).eval().requires_grad_(False)
            if self.args.get("encoder") != "none":  # also the depth quartiles n/4, n/2, 3n/4 (probe menu: mean_b<block>;
                n = len(enc.blocks)                  # 27/10, the rule of every token model in the benchmark)
                enc.taps = tuple(self.args.get("probe_taps") or (n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1))
            self._taps, self._dim = enc.taps, enc.dim
            obj = self.run_cfg["objective"]
            self._extract = encoder_extractor(enc, self.curves, "both", int(self.run_cfg["seed"]),
                                              obj.get("fg_threshold", 0.05), crop=bool(self.args.get("crop")),
                                              embed_only=self.args.get("encoder") == "none")
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
        vol = prepared["volume"].to(dev).float() * a + c
        x = patchify(vol, k.tolist())
        out = self.extractor()({"patches": [x], "grid": grid[None].to(dev), "k": k[None].to(dev),
                                "spacing": prepared["spacing"][None].to(dev)})
        d, n = self._dim, len(self._taps) + 1

        def block(v, j):  # [..., 2dn] = fwd (last block, taps) ++ bwd (same) -> block j's fwd ++ bwd [..., 2d]
            return torch.cat([v[..., j * d: (j + 1) * d], v[..., (n + j) * d: (n + j + 1) * d]], -1)

        bi, g = block(out["bi"][0, : len(x)], 0), grid.tolist()
        feats = {"map": bi.T.reshape(bi.shape[1], *g), "fg": out["fg"][0, : len(x)].view(*g)}  # last block's tokens
        for stat in ("mean", "std"):  # per block: last (no suffix), taps (_b<block>)
            feats |= {stat + ("" if j == 0 else f"_b{self._taps[j - 1] + 1}"): block(out[stat][0], j) for j in range(n)}
        # multigrain (27/10): the head features below 16 mm; off for another patch size (args.patch_mm) or encoder none
        if self.args.get("multigrain", True) and not self.args.get("patch_mm") and self.args.get("encoder") != "none":
            above = {s: v[: self._dim].half() for s, v in feats.items() if s.startswith(("mean", "std"))}
            fgk = out["fg"][0, : len(x)]   # 16 mm covariances per block, foreground tokens (exact: bench.probe.second_order)
            fgk = fgk if fgk.any() else torch.ones_like(fgk)
            for j in range(n):
                nm = "" if j == 0 else f"_b{self._taps[j - 1] + 1}"
                feats[f"tricov{nm}"] = above[f"tricov{nm}"] = tricov(block(out["bi"][0, : len(x)], j)[fgk, : self._dim])
            if pe.patch_mm == self.run_cfg["data"]["patch_mm"]:
                feats |= multigrain(self._extract_enc, self.curves, int(self.run_cfg["seed"]), vol,
                                    prepared["spacing"].to(dev), self.run_cfg["objective"].get("fg_threshold", 0.05), above)
            else:   # read at fallback_mm (whole-body scans): every level repeats it
                feats |= {f"g{int(m)}:{s}": v for m in FINE for s, v in above.items()}
        return {"features": feats, "canonical": "mean", "derived": [],
                "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}}

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__}


build = SFC
