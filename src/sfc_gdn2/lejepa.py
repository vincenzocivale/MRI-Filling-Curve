"""LeJEPA over groups of patches seen through different serializations: one masked-prediction task
repeated at every spatial scale (the H-JEPA pattern, Zhang et al. 2026, arXiv 2610.06805, with space in
place of time): patch tokens, cells of `levels` patches per edge, and the whole view on top. Each scale has
its own projector (latent space) and SIGReg. The volume summary is the top of the token pyramid, not a
separate objective on the last causal state (v7: that global term competed with the token term and lost:
age R^2 below raw intensities).

Per volume, boxes of its own patch grid ("groups": a few global, most local; a patch is a block of the
scan's native voxels, ~patch_mm per side, embedded by a continuous kernel: `model.KernelEmbed`). Each
group yields two views read along different curves (random curve of `view_curves` x cube symmetry,
traced within the view's box), each with its own box jitter, intensity augmentation and, with
probability `thick_prob`, simulated thick slices (native slices averaged into slabs of one of
`thick_mm` along a random axis, when that is thicker than the scan's own); view A also has `mask_ratio` of its
foreground patches replaced by the mask token, in aligned cubes of one of `mask_cells` patches per edge
(drawn per view, at most half the view's shortest edge). Each view is read forwards and backwards by the same
causal encoder. With `max_view_tokens` set, views hold at most that many patches (larger boxes shrink,
aspect kept); None (default) lets a global view span a whole volume. The patch embedding is recomputed in
backward in chunks of `embed_chunk_voxels`, so memory follows tokens, not native voxels (up to ~73k per patch),
bounds the tokens per step whatever the field of view.

- top (`global`): the mean bi token over the view's foreground patches -> `glob` head; invariance between
  A and B + SIGReg (per view, over the batch).
- token (`tok_self`, LeJEPA + iBOT++ as TIPSv2; 0 = off): in a fraction `tok_self_p` (global, local) of the groups
  view B is view A unmasked (same draws); there every foreground `bi` token (forward ++ backward output at that
  patch, as in the segmentation probe) of masked A, masked and visible, and the same token of B -> `tok` head in one
  call (same BatchNorm statistics), pulled
  together (symmetric, no stop-grad, no teacher, as LeJEPA's invariance). Foreground only (as Vol-JEPA's head mask):
  background patches stay in the sequence (with no position embedding their count is spatial signal) but are never
  masked, paired or sampled for SIGReg. SIGReg on B's foreground tokens, `tokens_per_volume` per volume (only
  the estimate's sample count: the weight is fixed by `sigreg_samples`).
- cells (`levels`, e.g. 2, 4, 8 patches = 32, 64, 128 mm per edge, aligned on the patch grid, so A and B
  share them whatever their boxes): mean foreground bi token per cell; every cell whose foreground is
  entirely masked in A is pulled to the same cell in B (`cell<c>` head); SIGReg on `tokens_per_volume` of
  B's cells per volume.
- volume (LeJEPA's multi-crop over the volume's boxes): the top of every view of every box -> `volume` head; each
  view pulled to the mean of the volume's global views; SIGReg per view slot, one sample per volume.
- session: the same over the scans of one session in the batch (the loader batches each scan with a random other
  scan of its session): what all its sequences share (anatomy, age, sex), not the acquisition signature that
  alone tells a volume's boxes from the rest of the batch. Its own head; SIGReg one sample per session.
  `top_levels` lists which of the two are on (both by default; [] for the A/B check).
View position (model.view_pos, `model.ViewPos`): each token gets its coordinates inside its own view's box after
the view's symmetry, so A and B place a patch differently and position alone cannot pair them.
loss = (1 - lam) * wmean(invariances) + lam * wmean(SIGRegs), one of each per scale, weight `w_top` for the view,
volume and session terms, 1 for cells and `tok_self` for the token term.
SIGReg sees every rank's samples. Its statistic grows with their count M (a test statistic), which made every
term's weight its sample count (view ~1152 vs session ~72 per step): each is rescaled by sigreg_samples / M, so
every term weighs as one of sigreg_samples samples (768: the token term lam was tuned on) whatever the batch,
`tokens_per_volume`, groups or number of GPUs.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.distributed.nn
from torch import nn
from torch.utils.checkpoint import checkpoint

from .curves import keys, symmetries, transform
from .data.dataset import per_patch
from .model import Encoder


def gather_rows(x: torch.Tensor) -> torch.Tensor:
    """Differentiable all-gather along dim 1 of tensors whose dim 1 differs per rank (batches of whole scans
    hold different numbers of scans): pad to the largest, gather, keep each rank's own rows."""
    n = torch.tensor([x.shape[1]], device=x.device)
    counts = [torch.zeros_like(n) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(counts, n)
    m = int(max(counts))
    parts = torch.distributed.nn.functional.all_gather(nn.functional.pad(x, (0, 0, 0, m - x.shape[1])))
    return torch.cat([p[:, : int(c)] for p, c in zip(parts, counts)], 1)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularisation: Epps-Pulley test of random 1D projections
    against N(0, 1), averaged over slices (Balestriero & LeCun, LeJEPA, 2025)."""

    def __init__(self, slices: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, t_max, knots)
        w = torch.full((knots,), 2 * t_max / (knots - 1))
        w[[0, -1]] /= 2  # trapezoid on [0, t_max], doubled for the symmetric half
        phi = torch.exp(-t.square() / 2)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("w", w * phi, persistent=False)

    def forward(self, z: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """z: [..., M, D] -> scalar (mean over leading dims and slices). Scaled by M like a test
        statistic (LeJEPA's); LeJEPA.forward rescales it to a fixed M."""
        with torch.autocast(z.device.type, enabled=False):  # cos/sin of bf16 (or tf32) projections is too coarse
            z = z.float()
            a = torch.randn(z.shape[-1], self.slices, device=z.device, generator=generator)
            a = a / a.norm(dim=0)
            xt = (z.double() @ a.double()).float().unsqueeze(-1) * self.t   # [..., M, S, T]; fp64: no tf32 either way
            err = (xt.cos().mean(-3) - self.phi).square() + xt.sin().mean(-3).square()
            return ((err @ self.w) * z.shape[-2]).mean()


class Projector(nn.Sequential):
    """BatchNorm always uses batch statistics (no running stats): the loss is defined on batch
    statistics in training, so evaluating with running ones would score a different function
    (it inflated val SIGReg ~1.4x at equal train/val data)."""

    def __init__(self, d_in: int, hidden: int, d_out: int):
        bn = lambda: nn.BatchNorm1d(hidden, track_running_stats=False)
        super().__init__(nn.Linear(d_in, hidden), bn(), nn.GELU(),
                         nn.Linear(hidden, hidden), bn(), nn.GELU(), nn.Linear(hidden, d_out))


class Protos(nn.Linear):
    """iBOT/DINO's last layer: cosine of the l2-normalised embedding with K l2-normalised prototypes -> [.., K]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x.float(), dim=-1) @ nn.functional.normalize(self.weight.float(), dim=-1).T


class LeJEPA(nn.Module):
    def __init__(self, encoder: Encoder, view_curves: list[str], seed: int = 17, groups_global: int = 2,
                 groups_local: int = 6, global_frac: tuple = (0.3, 1.0), local_edge: tuple = (3, 8),
                 jitter: float = 0.25, max_view_tokens: int | None = None, embed_chunk_voxels: int = 1 << 24,
                 max_packed_tokens: int = 1 << 16, checkpoint_tokens: int = 100_000,
                 thick_prob: float = 0.0,
                 thick_mm: tuple = (4.0, 8.0), mask_ratio: tuple = (0.2, 0.5), gamma: float = 0.3, scale: float = 0.1,
                 shift: float = 0.05, noise: float = 0.02, fg_threshold: float = 0.05, proj_hidden: int = 1024,
                 proj_dim: int = 128, tokens_per_volume: int = 64, lam: float = 0.05, sigreg_slices: int = 256,
                 sigreg_knots: int = 17, levels: tuple = (), mask_cells: tuple = (1,), sigreg_samples: int = 768,
                 w_top: float = 1.0, top_levels: tuple = ("volume", "session"), rec_weight: float = 0.0,
                 rec_cells: int = 4, rec_eps: float = 0.01, rec_visible: float = 0.0,
                 rec_head_cells: int = 0, rec_extra: bool = False, rec_contrast: float = 0.0, rec_edge: float = 0.0,
                 tok_self: float = 0.0, tok_self_p: tuple = (0.5, 0.5),
                 tok_self_ema: float = 0.0, tok_protos: int = 4096,
                 meta: float = 0.0, meta_ridge: float = 0.1, meta_min_mm: float = 8.0):
        super().__init__()
        self.encoder, self.gg, self.gl, self.k, self.lam = encoder, groups_global, groups_local, 2, lam
        self.curves, self.seed = list(view_curves), seed
        self.max_view_tokens = float("inf") if max_view_tokens is None else max_view_tokens
        self.embed_chunk_voxels, self.max_packed_tokens = embed_chunk_voxels, max_packed_tokens
        self.checkpoint_tokens = checkpoint_tokens
        self.thick_prob, self.thick_mm = thick_prob, tuple(float(t) for t in thick_mm)
        self.tokens_per_volume, self.sigreg_samples, self.w_top = tokens_per_volume, sigreg_samples, w_top
        self.global_frac, self.local_edge, self.jitter, self.mask_ratio = global_frac, local_edge, jitter, mask_ratio
        self.gamma, self.scale, self.shift, self.noise, self.fg_threshold = gamma, scale, shift, noise, fg_threshold
        self.levels = [int(c) for c in levels]
        # tok_self w (29/10, user: LeJEPA + iBOT++, TIPSv2): per group, with probability tok_self_p (global, local), view B
        # is view A unmasked (same box, curve, symmetry and augmentation; only its noise sample differs) instead of
        # another view; on those groups every foreground token of masked A, masked and visible, is pulled to the same
        # token of B, both with gradient (symmetric, no teacher, no stop-grad, as LeJEPA's invariance), SIGReg on B's
        # tokens, tokens_per_volume per volume. The other terms use every group as before. No extra reading. The only
        # token term (the cross-view one, A masked -> another B, removed 29/10: off since 23/10, no_tok better).
        # Weight w in both means; logged as inv_token(_m, _v) / sigreg_token.
        # tok_self_ema m > 0 (29/10, user: the EMA branch on the same pairs, TIPSv2's head-only EMA): no SIGReg on the term;
        # head + tok_protos prototypes (Protos), B's token through an EMA copy of both (momentum m -> 1 on a cosine,
        # ema_update after each optimizer step) without gradient, centred (EMA 0.9 of its mean logits) and sharpened
        # (temperature 0.04) as DINO/iBOT against A's at 0.1, cross-entropy; weight w added to the loss outside the means
        # (inv_token = the CE; tok_entropy: the target's entropy, ~0 = collapse onto a prototype)
        self.tok_self, self.tok_self_p = float(tok_self), tuple(tok_self_p)
        # meta w (30/10, user: post-training "understand the tasks", no labels): every unmasked B view of a scan at
        # >= meta_min_mm is an episode. Label-free targets per foreground token on a 2^3 sub-grid of its patch: the
        # scan's z-scored intensity of its session mate on the same grid in another contrast (cross; manifest
        # p31 pairs: session = session#grid), FLAIR hyperintensity (z > 2.5), 3 intensity classes (1D k-means). The
        # view's tokens split by a random plane: a ridge (closed form, strength meta_ridge x mean |x|^2) fitted on one
        # side, 1 - R^2 on the other, back-propagated through the solution into the encoder (MetaOptNet). Per rank, no
        # collective; weight w added to the loss; logged as meta and meta_r2_<family>.
        self.meta, self.meta_ridge, self.meta_min_mm = float(meta), float(meta_ridge), float(meta_min_mm)
        self.register_buffer("mask_cells", torch.tensor(mask_cells), persistent=False)
        self.glob = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.tok = Projector(2 * encoder.dim, proj_hidden, proj_dim) if tok_self else None
        self.tok_self_ema, self.tok_t = tok_self_ema, None
        if tok_self and tok_self_ema:
            self.tok = nn.Sequential(self.tok, Protos(proj_dim, tok_protos, bias=False))
            self.tok_t = copy.deepcopy(self.tok).requires_grad_(False)
            self.register_buffer("tok_center", torch.zeros(tok_protos))
        self.cell = nn.ModuleList(Projector(2 * encoder.dim, proj_hidden, proj_dim) for _ in self.levels)
        self.top = nn.ModuleDict({n: Projector(2 * encoder.dim, proj_hidden, proj_dim) for n in top_levels})
        self.rec_weight, self.rec_cells, self.rec_eps, self.rec_visible = rec_weight, rec_cells, rec_eps, rec_visible
        # rec_head_cells C (28/10, cascade of sub-token targets): the heads predict C^3 cells, pooled to the current
        # rec_cells (C divisible by it), so grain_plateau raises rec_cells mid-run with no parameter changing shape (DDP,
        # Adam); a checkpoint with fewer cells -> expand_rec_heads (each new cell = its parent: the same pooled output)
        self.rec_head_cells = int(rec_head_cells) or rec_cells
        mk = lambda n: nn.Sequential(nn.Linear(2 * encoder.dim, proj_hidden), nn.GELU(), nn.Linear(proj_hidden, n))
        self.rec = mk(self.rec_head_cells ** 3) if rec_weight or rec_visible else None
        # rec_extra: on the rec rows also the patch's mean and log-std on the volume's scale (rec_contrast: what rec's
        # per-patch normalisation drops, a token brighter than its neighbours) and its gradient magnitude in the cells
        # (rec_edge: where the edges run inside the token). Both heads always run (logged before their stage; every
        # parameter in the graph for DDP); their weights, 0 until grain_plateau opens their stage, scale the loss
        self.rec_contrast, self.rec_edge = rec_contrast, rec_edge
        self.rec_con_head = mk(2) if rec_extra and self.rec is not None else None
        self.rec_edge_head = mk(self.rec_head_cells ** 3) if rec_extra and self.rec is not None else None
        self.sigreg = SIGReg(sigreg_slices, sigreg_knots)
        perms, flips = symmetries()
        self.register_buffer("perms", perms, persistent=False)
        self.register_buffer("flips", flips, persistent=False)

    @classmethod
    def from_config(cls, cfg: dict) -> LeJEPA:
        return cls(Encoder.from_config(cfg), seed=int(cfg["seed"]), **cfg["objective"])

    def _u(self, shape, lo, hi, gen):
        return torch.rand(shape, device=self.perms.device, generator=gen) * (hi - lo) + lo

    def _cap(self, e: torch.Tensor, cap: float) -> torch.Tensor:
        """Box edges [...,3] (float) shrunk, aspect kept, to at most cap patches."""
        s = (cap / e.prod(-1, keepdim=True)).clamp(max=1) ** (1 / 3)
        return torch.where(s < 1, (e * s).floor().clamp_min(1), e)

    def boxes(self, fg: torch.Tensor, grid: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """fg [B,N], grid [B,3] -> group boxes (corner, edge) [B,G,3], globals first.
        Centres on foreground patches; global edges span global_frac of the foreground bounding box
        (volume fraction), local edges are log-uniform in local_edge per axis. No [B,N,3] coordinate table (25/10: at
        2 mm cells it was GBs, the reason max_patches_per_scan coarsened every brain to ~5 mm)."""
        b, g = len(fg), grid[:, None]
        lo, hi = [], []
        for i, (gx, gy, gz) in enumerate(grid.tolist()):   # per scan: the foreground's extent along each axis
            f = fg[i, : gx * gy * gz].view(gx, gy, gz)
            a = [f.any(2).any(1), f.any(2).any(0), f.any(1).any(0)]
            has = a[0].any()
            lo.append(torch.stack([torch.where(has, x.int().argmax(), 1 << 30) for x in a]))
            hi.append(torch.stack([torch.where(has, len(x) - 1 - x.flip(0).int().argmax(), -1) for x in a]))
        extent = (torch.stack(hi) - torch.stack(lo) + 1).clamp_min(1)                     # [B,3]
        frac = self._u((b, self.gg, 1), *self.global_frac, gen) ** (1 / 3)
        e_glob = self._cap((extent[:, None] * frac).round(), self.max_view_tokens)
        lo, hi = torch.tensor(self.local_edge, dtype=torch.float).log()
        # ponytail: independent per-axis edges (aspect in [3/8, 8/3]); add an explicit aspect prior if needed
        e_loc = self._u((b, self.gl, 3), lo, hi, gen).exp().round()
        edge = torch.cat([e_glob, e_loc], 1).clamp_min(1).minimum(g).long()               # [B,G,3]
        valid = torch.arange(fg.shape[1], device=fg.device) < grid.prod(1, keepdim=True)
        j = torch.multinomial(fg.float() + 1e-6 * valid, self.gg + self.gl, replacement=True, generator=gen)
        gy, gz = g[..., 1], g[..., 2]
        centre = torch.stack([j // (gy * gz), j // gz % gy, j % gz], -1)                     # [B,G,3]
        return (centre - edge // 2).clamp(min=0).minimum(g - edge), edge

    def jittered(self, corner, edge, grid, gen):
        """K views per group: each box rescaled and shifted by up to +-jitter of its edge -> [B,G,K,3]."""
        g, shape = grid[:, None, None], (*edge.shape[:2], self.k, 3)
        e = (edge[:, :, None] * self._u(shape, -self.jitter, self.jitter, gen).exp()).round().clamp_min(1)
        e = self._cap(e, self.max_view_tokens).minimum(g).long()
        c = corner[:, :, None] + (self._u(shape, -self.jitter, self.jitter, gen) * edge[:, :, None]).round().long()
        return c.clamp(min=0).minimum(g - e), e

    def serialize(self, grid, corner, edge, n, tot, gen, src):
        """Boxes (corner, edge) [V,3] in patch grids [V,3], n [V] patches per box (tot in all, host) -> every view's
        patches in the order of a random curve of `view_curves` under a random cube symmetry, traced within the box,
        packed view after view (no padding): view [T], patch index in its scan's raster [T], coordinates [T,3] in the
        scan and in the box after the symmetry (the view position)."""
        dev, nv = corner.device, len(corner)
        sym = torch.randint(len(self.perms), (nv,), device=dev, generator=gen)[src]
        curve = torch.randint(len(self.curves), (nv,), device=dev, generator=gen)[src]
        tv = torch.repeat_interleave(torch.arange(nv, device=dev), n, output_size=tot)
        j, e = torch.arange(tot, device=dev) - (n.cumsum(0) - n)[tv], edge[tv]
        box = torch.stack([j // (e[:, 1] * e[:, 2]), j // e[:, 2] % e[:, 1], j % e[:, 2]], -1)   # raster of the box
        pxyz, g = corner[tv] + box, grid[tv]
        local, dims = transform(box, e, self.perms[sym[tv]], self.flips[sym[tv]])
        key = torch.zeros_like(j)
        for i, name in enumerate(self.curves):
            key = torch.where(curve[tv] == i, keys(name, local, dims, self.seed, gen), key)
        order = key.argsort()                                   # keys are distinct within a box
        order = order[tv[order].argsort(stable=True)]           # views kept in place, each along its curve
        pxyz = pxyz[order]
        return tv, (pxyz[:, 0] * g[:, 1] + pxyz[:, 1]) * g[:, 2] + pxyz[:, 2], pxyz, local[order]

    @staticmethod
    def cell_ids(p: torch.Tensor, grid: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Patch coords p [...,3] in grids [...,3] -> id [...] of the aligned cube of c [...] patches per edge holding
        each patch (ids < the grid's patch count)."""
        gc = (grid + c[..., None] - 1) // c[..., None]
        q = p.clamp_min(0) // c[..., None]
        return (q[..., 0] * gc[..., 1] + q[..., 1]) * gc[..., 2] + q[..., 2]

    @staticmethod
    def thick(v: torch.Tensor, k, a: int, f: int) -> torch.Tensor:
        """Patches [R,P] of one scan (k voxels per axis): native slices along axis a averaged in consecutive groups of
        f (a thick-slice acquisition)."""
        g = torch.arange(k[a], device=v.device) // f
        m = (g[:, None] == g[None, :]).float()
        x = v.unflatten(-1, tuple(k)).movedim(1 + a, -1)
        with torch.autocast(v.device.type, enabled=False):              # voxel values stay fp32
            return (x @ (m / m.sum(0))).movedim(-1, 1 + a).flatten(-3)

    def _embed(self, x, pid, vid, w, aug, k, slabs, seed):
        """Patches `pid` of one scan's [N,P] voxel values, each augmented with the parameters of its view `vid`
        (aug: per-view tensors; noise drawn from `seed`, so a recomputation in backward is identical); `slabs`:
        (first row, end, axis, f) of the rows whose view simulates thick slices (`thick`) -> [R,d]."""
        gamma, scale, shift, noise = (a[vid] for a in aug)
        v = x[pid]
        v = v.sign() * v.abs() ** gamma[:, None] * scale[:, None] + shift[:, None]
        g = torch.Generator(v.device).manual_seed(seed)
        v = v + torch.randn(v.shape, device=v.device, generator=g) * noise[:, None]
        for s, e, a, f in slabs:                           # only those rows (before: every row, every axis and slab)
            v[s:e] = self.thick(v[s:e], k, a, f)
        return self.encoder.patch_embed(v, w)

    def read(self, vols, ws, fg, n_patch, vol, corner, edge, masked, gen, src=None):
        """Views [V] (scan `vol`, box) -> their tokens packed view after view, each along its curve (no padding):
        view [T], patch index in its scan [T], patch coords [T,3], foreground [T], mask [T], bi tokens [T,2d]; tokens
        per view [V] and per-view offsets [V] (device) and the tokens per view (host). Masking applies only to
        foreground patches of views with `masked`; the backward pass reads each view in reverse. Tokens are embedded
        per scan (its own voxels per patch and kernel weights `ws`). Host syncs: a handful per call. src [V]: the view whose
        draws (symmetry, curve, augmentation, thick slices) each view takes (itself by default; tok_self: a B view tied
        to its A)."""
        nv, dev = len(corner), corner.device
        src = torch.arange(nv, device=dev) if src is None else src
        n = edge.prod(-1)
        vol_h, lens = torch.stack([vol, n]).tolist()
        tot = sum(lens)
        tv, idx, pxyz, vpos = self.serialize(vols["grid"][vol], corner, edge, n, tot, gen, src)
        fgv = fg[vol[tv], idx]
        u = lambda lo, hi: self._u((nv, 1, 1), lo, hi, gen)[src]
        gamma, scale, shift, noise = u(-self.gamma, self.gamma).exp(), u(1 - self.scale, 1 + self.scale), \
            u(-self.shift, self.shift), u(0, self.noise)
        on = (torch.rand(nv, device=dev, generator=gen) < self.thick_prob)[src]
        axis = torch.randint(3, (nv,), device=dev, generator=gen)[src]
        mm = torch.randint(len(self.thick_mm), (nv,), device=dev, generator=gen)[src]
        # masked in aligned cubes of one of mask_cells per view (<= half its shortest edge): big cubes are the
        # coarse levels' prediction targets
        cells = self.mask_cells[None]
        ok_c = cells <= (edge.amin(-1, keepdim=True) // 2).clamp_min(1)
        c = cells[0][torch.rand(ok_c.shape, device=dev, generator=gen).masked_fill(~ok_c, -1).argmax(1)]
        cid = self.cell_ids(pxyz, vols["grid"][vol[tv]], c[tv])
        # one uniform per (view, cube): a hash of a per-view seed, not a [views x patches] table (GBs at 2 mm cells)
        draw = _hash01(torch.randint(1 << 30, (nv,), device=dev, generator=gen)[tv], cid)
        mask = masked[tv] & fgv & (draw < u(*self.mask_ratio).view(-1)[tv])
        aug = (gamma.view(-1), scale.view(-1), shift.view(-1), noise.view(-1))
        host = torch.cat([on.long(), axis, mm, torch.randint(1 << 62, (1,), device=dev, generator=gen)]).tolist()
        on_h, axis_h, mm_h, seed = host[:nv], host[nv:2 * nv], host[2 * nv:3 * nv], host[-1]   # one host sync
        offs = [0]
        for ln in lens:
            offs.append(offs[-1] + ln)
        spans, slabs = {}, {}                               # per scan: its token range (its views are contiguous)
        for i, b in enumerate(vol_h):
            s = spans.setdefault(b, [offs[i], offs[i]])
            assert s[1] == offs[i], "views of a scan must be contiguous"
            s[1] = offs[i + 1]
            sp = vols["spacing_list"][b]
            if on_h[i] and (f := round(self.thick_mm[mm_h[i]] / sp[axis_h[i]])) >= 2:   # else already that thick
                slabs.setdefault(b, []).append((offs[i], offs[i + 1], axis_h[i], f))
        parts = []
        for b, (s0, s1) in spans.items():                  # one scan at a time: its voxels per patch and kernel weights
            x, k = vols["patches"][b], vols["k_list"][b]
            chunk = max(1, self.embed_chunk_voxels // x.shape[1])
            for c in range(0, s1 - s0, chunk):             # recomputed in backward: memory ~ tokens, not voxels
                lo, hi = s0 + c, min(s0 + c + chunk, s1)
                sl = [(max(a, lo) - lo, min(z, hi) - lo, ax, f) for a, z, ax, f in slabs.get(b, []) if a < hi and z > lo]
                parts.append(checkpoint(self._embed, x, idx[lo:hi], tv[lo:hi], ws[b], aug, k, sl,
                                        seed + b * 1_000_003 + c, use_reentrant=False))
        t = self.encoder.apply_mask(torch.cat(parts), mask)
        if self.encoder.pos is not None:   # masked patches too: the mask token learns where it is
            t = t + self.encoder.pos(vpos).to(t.dtype)
        off = n.cumsum(0) - n
        rev = 2 * off[tv] + n[tv] - 1 - torch.arange(tot, device=dev)                      # an involution
        late = self.encoder.bidir == "late"                # else one stream, both ways inside the blocks
        seqs, lens2 = (torch.cat([t, t.index_select(0, rev)]), lens + lens) if late else (t, lens)   # fwd, then bwd
        pseq = torch.cat([vpos, vpos.index_select(0, rev)]) if late else vpos   # patch coordinates (rope3d, dwconv)
        cu = nn.functional.pad((torch.cat([n, n]) if late else n).cumsum(0), (1, 0))
        groups, cur, size = [], [], 0
        for r, ln in enumerate(lens2):     # packed calls of <= max_packed_tokens: GDN-2's backward peak follows the call
            if cur and size + ln > self.max_packed_tokens:
                groups.append(cur)
                cur, size = [], 0
            cur.append(r)
            size += ln
        groups.append(cur)
        ckpt = self.encoder.grad_checkpoint and len(seqs) > self.checkpoint_tokens   # recompute only for big scans
        sizes = [sum(lens2[g[0]:g[-1] + 1]) for g in groups]
        parts = seqs.split(sizes)                                          # split: one cat in backward (slices:
        h = torch.cat([self.encoder.flat(x, (cu[g[0]:g[-1] + 2] - cu[g[0]]).int(), ckpt, p)   # a zero-filled [2T,d]
                       for x, g, p in zip(parts, groups, pseq.split(sizes))])             # gradient per call)
        fwd, bwd = h.split(tot) if late else (h, h)
        bi = torch.cat([fwd, bwd.index_select(0, rev) if late else bwd], -1)   # index_select: backward index_add
        return tv, idx, pxyz, fgv, mask, bi, n, off, lens

    def _episodes(self, pv, v, idx, pxyz, fgv, bi, n_h, gen, acc: dict) -> None:
        """objective.meta: one episode per unmasked (B) view of the read; 1 - R^2 per family appended to acc."""
        dev = bi.device
        mms = pv.get("patch_mm") or [self.encoder.patch_embed.patch_mm] * len(pv["patches"])
        ses, mod, flair = pv["session"].tolist(), pv["modality"].tolist(), pv["flair"].tolist()
        stats = {}

        def scan(b):   # z-scored voxels and 3 intensity-class centres of scan b (its foreground), once
            if b not in stats:
                x = pv["patches"][b].float()
                f = x[x > self.fg_threshold]
                mu, sd = f.mean(), f.std().clamp_min(1e-6)
                z = ((f[torch.randint(len(f), (min(len(f), 1 << 17),), device=dev, generator=gen)] - mu) / sd)
                c = torch.quantile(z, torch.tensor([0.2, 0.5, 0.8], device=dev))
                for _ in range(15):
                    a = (z[:, None] - c).abs().argmin(1)
                    c = torch.stack([z[a == j].mean() if (a == j).any() else c[j] for j in range(3)])
                stats[b] = (mu, sd, c)
            return stats[b]

        def target(b, pid):   # z-scored voxels of scan b's patches pid -> [R, 1, *k] and its foreground
            mu, sd, _ = scan(b)
            x = pv["patches"][b][pid].float()
            return ((x - mu) / sd).view(-1, 1, *pv["k_list"][b]), (x > self.fg_threshold).view(-1, 1, *pv["k_list"][b])

        pool = lambda t: nn.functional.adaptive_avg_pool3d(t.float(), 2).flatten(1)
        offs = [0]
        for ln in n_h:
            offs.append(offs[-1] + ln)
        with torch.autocast(dev.type, enabled=False):
            for j in range(1, len(n_h), 2):   # B views
                b = int(v[j])
                if mms[b] < self.meta_min_mm:
                    continue
                r = torch.arange(offs[j], offs[j + 1], device=dev)
                r = r[fgv[r]]
                if len(r) < 64:
                    continue
                pid = idx[r]
                with torch.no_grad():
                    z, fgm = target(b, pid)
                    fam = {"tissue": torch.cat([pool(((z - scan(b)[2].view(1, 3, 1, 1, 1)).abs().argmin(1, keepdim=True)
                                                       == j_) & fgm) for j_ in range(3)], 1)}
                    if flair[b]:
                        fam["hyper"] = pool((z > 2.5) & fgm)
                    mate = [m for m in range(len(ses)) if m != b and ses[m] == ses[b] and mod[m] != mod[b]
                            and pv["k_list"][m] == pv["k_list"][b] and pv["patches"][m].shape == pv["patches"][b].shape]
                    if mate:
                        zm, fm = target(mate[0], pid)
                        fam["cross"] = pool(zm * fm)
                    u = torch.randn(3, device=dev, generator=gen)
                    side = (pxyz[r].float() - pxyz[r].float().mean(0)) @ u
                    s = side < side.median()
                if s.sum() < 32 or (~s).sum() < 32:
                    continue
                x = bi[r].float()
                xs, xq = x[s], x[~s]
                m0 = xs.mean(0)
                xs, xq = xs - m0, xq - m0
                y = torch.cat(list(fam.values()), 1)
                ys, yq = y[s], y[~s]
                y0 = ys.mean(0)
                lam = self.meta_ridge * xs.square().sum(1).mean()
                if len(xs) <= xs.shape[1]:   # kernel form (fewer support tokens than dimensions)
                    a = torch.linalg.solve(xs @ xs.T + lam * torch.eye(len(xs), device=dev), ys - y0)
                    pred = xq @ (xs.T @ a) + y0
                else:
                    pred = xq @ torch.linalg.solve(xs.T @ xs + lam * torch.eye(xs.shape[1], device=dev), xs.T @ (ys - y0)) + y0
                c0 = 0
                for f, t in fam.items():
                    sl = slice(c0, c0 + t.shape[1])
                    c0 += t.shape[1]
                    sst = (yq[:, sl] - yq[:, sl].mean(0)).square().sum()
                    if sst > 1e-3 * len(yq):   # a target with no variance on the query side: no episode
                        acc.setdefault(f, []).append(((pred[:, sl] - yq[:, sl]).square().sum() / sst).clamp(max=4))

    def _grid_ctx(self, vols: dict) -> tuple:
        """One cut of the batch's scans -> (vols with host lists, grid [B,3], n_patch, foreground [B,N], kernel
        weights per scan)."""
        grid = vols["grid"]
        b, n_patch = len(grid), int(grid.prod(1).max())
        fg = per_patch(vols, n_patch, lambda x: x.mean(-1)) > self.fg_threshold            # [B,N]
        vols = {**vols, "k_list": vols["k"].tolist(), "spacing_list": vols["spacing"].tolist()}
        memo, pe = {}, self.encoder.patch_embed     # one kernel weight matrix per distinct (spacing, k, cell mm)
        mm0 = pe.patch_mm
        mms = vols.get("patch_mm") or [mm0] * b    # per scan (repatch with max_patches_per_scan): its own cell size
        for sp, k, m in zip(vols["spacing_list"], vols["k_list"], mms):
            if (key := (*sp, *k, m)) not in memo:
                pe.patch_mm = m
                memo[key] = pe.weights(torch.tensor(sp, device=grid.device), k)
        pe.patch_mm = mm0
        ws = [memo[(*sp, *k, m)] for sp, k, m in zip(vols["spacing_list"], vols["k_list"], mms)]
        return vols, grid, n_patch, fg, ws

    def forward(self, vols: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """vols (`dataset.to_device`): patches = per scan [N_b, P_b] native voxel values, spacing [B,3] mm,
        k [B,3] voxels per patch axis, grid [B,3] patch grid, session [B] (any ids)."""
        gen, b = generator, len(vols["grid"])
        fine = self._grid_ctx(vols)
        # data.global_mm (25/10, user): at a finer step the global views are read on the scans cut at that cell
        # (vols["global"], repatched by the trainer: the whole volume, as in the trunk, not a few cm at the token cap),
        # the local ones at the step's grain; the volume and session levels pull every view to the mean of the
        # globals, so the fine views are tied to the coarse reading of the same scan (cross-grain invariance)
        coarse = self._grid_ctx(vols["global"]) if "global" in vols else fine
        vols, grid, _, fg, _ = fine
        def rec_target(scan, pid, vols):
            """Sub-patch targets of the rows (scan [R], patch [R]): each patch's native voxels in rec_cells^3 cells,
            normalised per patch ("rec"); with rec_extra also "contrast" (the patch's mean and log-std, both on the
            scan's foreground scale) and "edge" (log1p of the gradient magnitude on that scale, central differences
            inside the patch, in the same cells). Only the rows a loss uses (22/10: all patches of the batch held a copy
            of every voxel at 4 mm cells, ~10 GB and OOM)."""
            out, pool = {}, lambda t: nn.functional.adaptive_avg_pool3d(t, self.rec_cells).flatten(1)
            with torch.no_grad(), torch.autocast(grid.device.type, enabled=False):
                for b in scan.unique().tolist():
                    s = (scan == b).nonzero()[:, 0]
                    x, k = vols["patches"][b], vols["k_list"][b]
                    xs = x[pid[s]].view(-1, 1, *k).float()
                    t = pool(xs)
                    got = {"rec": (t - t.mean(1, keepdim=True)) / (t.var(1, keepdim=True, unbiased=False) + self.rec_eps).sqrt()}
                    if self.rec_con_head is not None:
                        nz = x[x != 0].float()
                        vm, vs = (nz.mean(), nz.std().clamp_min(1e-6)) if len(nz) > 1 else (x.float().mean(), x.float().std().clamp_min(1e-6))
                        f = xs.flatten(1)
                        got["contrast"] = torch.stack([(f.mean(1) - vm) / vs, (f.std(1, unbiased=False) / vs + self.rec_eps).log()], 1)
                        q = nn.functional.pad(xs, (1, 1, 1, 1, 1, 1), mode="replicate")
                        g = ((q[:, :, 2:, 1:-1, 1:-1] - q[:, :, :-2, 1:-1, 1:-1]).square()
                             + (q[:, :, 1:-1, 2:, 1:-1] - q[:, :, 1:-1, :-2, 1:-1]).square()
                             + (q[:, :, 1:-1, 1:-1, 2:] - q[:, :, 1:-1, 1:-1, :-2]).square()).sqrt() / (2 * vs)
                        got["edge"] = pool(g).log1p()
                    for n, t in got.items():
                        out.setdefault(n, t.new_empty(len(scan), t.shape[1]))[s] = t
            return out
        rec, recv, con, edg = [], [], [], []
        corner, edge = self.jittered(*self.boxes(fg, grid, gen), grid, gen)               # [B,G,2,3]
        if coarse is not fine:   # the global boxes on the coarse grid (the fine draw's globals unused)
            gc, ge = self.jittered(*self.boxes(coarse[3], coarse[1], gen), coarse[1], gen)
            corner, edge = (torch.cat([g[:, : self.gg], f[:, self.gg:]], 1) for g, f in ((gc, corner), (ge, edge)))
        vol = torch.arange(b, device=grid.device)[:, None, None].expand(-1, corner.shape[1], 2)
        same = torch.zeros(corner.shape[:2], dtype=torch.bool, device=grid.device)                    # [B,G]
        if self.tok_self:   # groups whose B is A unmasked: A's box
            pg = torch.tensor([self.tok_self_p[0]] * self.gg + [self.tok_self_p[1]] * self.gl, device=grid.device)
            same = torch.rand(same.shape, device=grid.device, generator=gen) < pg
            # >= 1 tied group per part on every rank: the token head's SyncBatchNorm and gathers are collectives, a
            # rank without one deadlocked the others (NCCL timeout, p30 29/10)
            for s in (0, self.gg)[: 1 + bool(self.gl)]:
                same[0, s] |= ~same[:, s: s + (self.gg if s == 0 else self.gl)].any()
            corner, edge = (torch.where(same[..., None, None], t[:, :, :1], t) for t in (corner, edge))
        zg, lens, tops, spred, stgt, smask, svol = [], [], [], [], [], [], []
        meta = {}   # family -> [1 - R^2 per episode]
        cpred, ctgt, cz, cvol = ([[] for _ in self.levels] for _ in range(4))
        for part in (slice(0, self.gg), slice(self.gg, None)):                             # globals / locals
            pv, pgrid, pn, pfg, pws = coarse if part.start == 0 else fine
            c, e, v = (t[:, part].reshape(-1, *t.shape[3:]) for t in (corner, edge, vol))  # views A,B interleaved
            nv, dev = len(c), c.device
            masked = torch.arange(nv, device=dev) % 2 == 0
            sg = same[:, part].reshape(-1)                                                 # per group
            src = torch.arange(nv, device=dev) - (sg.repeat_interleave(2) & ~masked).long()   # tied B -> its A
            tv, idx, pxyz, fgv, mask, bi, n, off, n_h = self.read(pv, pws, pfg, pn, v, c, e, masked, gen, src)
            if self.meta:
                self._episodes(pv, v, idx, pxyz, fgv, bi, n_h, gen, meta)
            has = torch.zeros(nv, device=dev).index_add_(0, tv, fgv.float()) > 0
            w = (fgv | ~has[tv]).float()                                                   # foreground (all if none)
            top = torch.zeros(nv, bi.shape[-1], device=dev).index_add_(0, tv, bi.float() * w[:, None]) \
                / torch.zeros(nv, device=dev).index_add_(0, tv, w)[:, None]
            zg.append(self.glob(top).float().view(b, -1, 2, self.glob[-1].out_features))
            tops.append(top.view(b, -1, top.shape[-1]))
            for i, (cs, head) in enumerate(zip(self.levels, self.cell)):
                p, q, zb, vb = self.cells(bi, tv, fgv, mask, pxyz, pgrid[v[tv]], cs, pn, head)
                cpred[i].append(p)
                ctgt[i].append(q)
                cz[i].append(zb)
                cvol[i].append(v[1::2][vb])
            if self.tok_self and bool(sg.any()):   # tied groups: every foreground token of A against the same in B
                ra, rb = (_ranges(off[k::2], n[k::2], sum(n_h[k::2])) for k in (0, 1))
                # the same patches in A and B (same box), in their own curve orders: aligned by (group, patch)
                ra, rb = (r[sg[tv[r] // 2] & fgv[r]] for r in (ra, rb))
                ra, rb = (r[(tv[r] // 2 * pn + idx[r]).argsort()] for r in (ra, rb))
                # one head call: both under the same BatchNorm statistics; recomputed in backward for big scans
                if self.tok_t is not None:   # EMA: A through the head, B through its EMA copy, no gradient
                    x = bi.index_select(0, ra)
                    spred.append((checkpoint(self.tok, x, use_reentrant=False) if len(x) > self.checkpoint_tokens
                                  else self.tok(x)).float())
                    with torch.no_grad():
                        stgt.append(self.tok_t(bi.index_select(0, rb)).float())
                else:
                    x = torch.cat([bi.index_select(0, ra), bi.index_select(0, rb)])
                    z = (checkpoint(self.tok, x, use_reentrant=False) if len(x) > self.checkpoint_tokens else self.tok(x)).float()
                    spred.append(z[: len(ra)])
                    stgt.append(z[len(ra):])
                smask.append(mask[ra])
                svol.append(v[tv[ra]])
            lens.append(e.prod(-1).float().mean())
            # masked tokens (A's foreground) -> their patch's sub-patch layout (MAE-style); rec_visible: B's foreground
            for w, r, acc in ((self.rec_weight, mask, rec), (self.rec_visible, fgv & (tv % 2 == 1), recv)):
                if w:
                    r = r.nonzero()[:, 0]
                    if len(r):
                        h, tg = bi.index_select(0, r), rec_target(v[tv[r]], idx[r], pv)
                        acc.append((self.rec_cells_of(self.rec(h)) - tg["rec"]).square().mean(1))
                        if self.rec_con_head is not None:
                            con.append((self.rec_con_head(h).float() - tg["contrast"]).square().mean(1))
                            edg.append((self.rec_cells_of(self.rec_edge_head(h)) - tg["edge"]).square().mean(1))
        zg = torch.cat(zg, 1)                                                              # [B,G,2,D]
        inv_g = (zg - zg.mean(2, keepdim=True)).square().mean()

        def sig(z):  # every rank's samples (differentiable gather), weighed as sigreg_samples whatever their count
            z = gather_rows(z) if torch.distributed.is_initialized() else z
            return self.sigreg(z, gen) * (self.sigreg_samples / z.shape[-2])

        sig_g = sig(zg.permute(2, 0, 1, 3).reshape(2, -1, zg.shape[-1]))
        out = {}
        for cs, p, q, z, vb in zip(self.levels, cpred, ctgt, cz, cvol):
            p, q, z, vb = torch.cat(p), torch.cat(q), torch.cat(z), torch.cat(vb)
            out[f"inv_cell{cs}"] = (p - q).square().mean(-1).sum() / max(len(p), 1) / 4
            out[f"sigreg_cell{cs}"] = sig(z[self._sample(vb, torch.ones_like(vb, dtype=torch.bool), b, gen)][None])
            out[f"pairs_cell{cs}"] = torch.tensor(float(len(p)), device=z.device)
        tops = torch.cat(tops, 1)                                                          # [B,V,2d], globals first
        groups = {"volume": torch.arange(b, device=grid.device),
                  "session": vols["session"].unique(return_inverse=True)[1]}
        for name, head in self.top.items():
            grp = groups[name]
            n = int(grp.max()) + 1
            z = head(tops.flatten(0, 1)).float().view(b, -1, head[-1].out_features)       # [B,V,D]
            cnt = torch.bincount(grp, minlength=n).clamp_min(1)[:, None]
            centre = z.new_zeros(n, z.shape[-1]).index_add_(0, grp, z[:, : 2 * self.gg].mean(1)) / cnt
            out[f"inv_{name}"] = (z - centre[grp][:, None]).square().mean()
            per = z.new_zeros(n, *z.shape[1:]).index_add_(0, grp, z) / cnt[..., None]      # one sample per group
            out[f"sigreg_{name}"] = sig(per.transpose(0, 1))                               # [V,n,D]: per view slot
        inv = [inv_g, *(out[f"inv_cell{c}"] for c in self.levels), *(out[f"inv_{n}"] for n in self.top)]
        sigs = [sig_g, *(out[f"sigreg_cell{c}"] for c in self.levels), *(out[f"sigreg_{n}"] for n in self.top)]
        w = [self.w_top, *(1.0 for _ in self.levels), *(self.w_top for _ in self.top)]
        tok_loss = 0
        if spred:
            zs, zt, m = torch.cat(spred), torch.cat(stgt), torch.cat(smask)
            if self.tok_t is None:   # symmetric pull (= mean ||z - mean||^2) + SIGReg on B's tokens, per volume
                d = (zs - zt).square().mean(1) / 4
                vs = torch.cat(svol)
                out["sigreg_token"] = sig(zt[self._sample(vs, torch.ones_like(vs, dtype=torch.bool), b, gen)][None])
            else:                    # centred, sharpened EMA target; cross-entropy
                q = ((zt - self.tok_center) / 0.04).softmax(-1)
                d = -(q * (zs / 0.1).log_softmax(-1)).sum(-1)
                out["tok_entropy"] = -(q * q.clamp_min(1e-12).log()).sum(-1).mean()
                if self.training:
                    c = zt.mean(0)
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(c)
                        c /= torch.distributed.get_world_size()
                    self.tok_center.mul_(0.9).add_(0.1 * c)
            out["inv_token"] = d.mean()
            out["inv_token_m"] = d[m].mean() if m.any() else d.sum() * 0
            out["inv_token_v"] = d[~m].mean() if (~m).any() else d.sum() * 0
            if self.tok_t is None:
                inv.append(out["inv_token"]), sigs.append(out["sigreg_token"]), w.append(self.tok_self)
            else:
                tok_loss = self.tok_self * out["inv_token"]
        wmean = lambda xs: sum(a * x for a, x in zip(w, xs)) / sum(w)
        if meta:
            out["meta"] = torch.cat([torch.stack(x) for x in meta.values()]).mean()
            out |= {f"meta_r2_{f}": 1 - torch.stack(x).detach().mean() for f, x in meta.items()}
        for k, r in (("rec", rec), ("rec_visible", recv), ("rec_contrast", con), ("rec_edge", edg)):
            if r:
                out[k] = (r := torch.cat(r)).sum() / max(len(r), 1)
        return {"loss": (1 - self.lam) * wmean(inv) + self.lam * wmean(sigs) + tok_loss + self.rec_weight * out.get("rec", 0)
                + self.rec_visible * out.get("rec_visible", 0) + self.rec_contrast * out.get("rec_contrast", 0)
                + self.rec_edge * out.get("rec_edge", 0) + self.meta * out.get("meta", 0),
                "inv_global": inv_g, "sigreg_global": sig_g, **out, "len_global": lens[0], "len_local": lens[1]}

    def rec_cells_of(self, p: torch.Tensor) -> torch.Tensor:
        """A rec head's rec_head_cells^3 prediction [R, C^3] -> the current rec_cells^3 (mean of each cell's children)."""
        C = self.rec_head_cells
        p = p.float()
        return p if C == self.rec_cells else nn.functional.adaptive_avg_pool3d(p.view(-1, 1, C, C, C), self.rec_cells).flatten(1)

    @staticmethod
    def expand_rec_heads(sd: dict, cells: int) -> dict:
        """A state dict's rec heads (last Linear of rec / rec_edge_head, c^3 outputs) -> cells^3 outputs, each new cell a
        copy of its parent (c | cells), so rec_cells_of pooled back to c gives the old prediction exactly (28/10)."""
        for key in [k for k in sd if k.split(".")[-3:-1] in (["rec", "2"], ["rec_edge_head", "2"])]:
            c = round(sd[key].shape[0] ** (1 / 3))
            if c != cells:
                i, f = torch.arange(cells), cells // c
                parent = ((i[:, None, None] // f) * c + i[None, :, None] // f) * c + i[None, None, :] // f
                sd[key] = sd[key][parent.flatten().to(sd[key].device)]
        return sd

    @torch.no_grad()
    def ema_update(self, progress: float = 0.0):
        """tok_self_ema: the target head follows the trained one after each optimizer step, momentum tok_self_ema -> 1
        on a cosine over training (progress in [0, 1]), as DINO/iBOT."""
        if self.tok_t is not None:
            m = 1 - (1 - self.tok_self_ema) * (1 + math.cos(math.pi * progress)) / 2
            for t, p in zip(self.tok_t.parameters(), self.tok.parameters()):
                t.lerp_(p, 1 - m)

    def _sample(self, vol: torch.Tensor, fg: torch.Tensor, b: int, gen) -> torch.Tensor:
        """`tokens_per_volume` row indices per volume, drawn among its foreground rows (with replacement: a fixed
        shape for the cross-GPU gather); a volume with no foreground row falls back to all its rows, with none to any.
        Vectorised (before: a loop over volumes, two host syncs each)."""
        cnt = torch.zeros(b, dtype=torch.long, device=vol.device).index_add_(0, vol, torch.ones_like(vol))
        nfg = torch.zeros_like(cnt).index_add_(0, vol, fg.long())
        order = (vol * 2 + (~fg).long()).argsort(stable=True)          # by volume, its foreground rows first
        pool = torch.where(nfg > 0, nfg, cnt)[:, None]
        r = torch.rand(b, self.tokens_per_volume, device=vol.device, generator=gen)
        j = torch.where(pool > 0, (cnt.cumsum(0) - cnt)[:, None] + (r * pool).long().clamp_max(pool - 1),
                        (r * len(vol)).long().clamp_max(len(vol) - 1))
        return order[j.flatten()]

    def cells(self, bi, tv, sel, mask, pxyz, grid, c, n_patch, head):
        """Packed tokens of views A (even) and B (odd): bi [T,2d], their view tv [T], foreground `sel` [T], mask
        [T], patch coords [T,3] in grids [T,3] -> per cube of c patches per edge: projected mean token of A at cubes
        whose foreground is entirely masked in A, of B at the same cubes, of B at every cube it holds, and the pair
        index of the latter."""
        cid = self.cell_ids(pxyz, grid, torch.full_like(tv, c))
        pair, isa = tv // 2 * n_patch + cid, tv % 2 == 0
        sa, sb = sel & isa, sel & ~isa
        pa, pb = pair[sa], pair[sb]
        u, inv = torch.unique(torch.cat([pa, pb]), return_inverse=True)
        ia, ib = inv[: len(pa)], inv[len(pa):]
        n_a, n_b = torch.bincount(ia, minlength=len(u)), torch.bincount(ib, minlength=len(u))
        mean = lambda x, i, n: x.new_zeros(len(u), x.shape[-1]).index_add_(0, i, x) / n.clamp_min(1)[:, None]
        m_a, m_b = mean(bi[sa].float(), ia, n_a), mean(bi[sb].float(), ib, n_b)
        full = (torch.bincount(ia, weights=mask[sa].float(), minlength=len(u)) == n_a) & (n_a > 0)
        ok, has_b = full & (n_b > 0), n_b > 0
        m_a = m_a[ok]
        z = head(torch.cat([m_a, m_b[has_b]])).float()
        z_a, z_b = z[: len(m_a)], z[len(m_a):]
        return z_a, z_b[(has_b.cumsum(0) - 1)[ok]], z_b, u[has_b] // n_patch


def _hash01(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Integers a, b >= 0 (a < 2^31) -> uniform-looking floats in [0, 1): a 31-bit integer hash (0x45d9f3b rounds)."""
    m = (1 << 31) - 1
    x = (a * 1_000_003 + b) & m
    for _ in range(3):
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & m
    return (x >> 7).float() / (1 << 24)   # 24 bits: exact in float32, so < 1


def _ranges(start: torch.Tensor, n: torch.Tensor, total: int) -> torch.Tensor:
    """Indices of the ranges [start_i, start_i + n_i) back to back; total = sum(n), known on the host (no sync)."""
    r = torch.repeat_interleave(torch.arange(len(n), device=n.device), n, output_size=total)
    return start[r] + torch.arange(total, device=n.device) - (n.cumsum(0) - n)[r]
