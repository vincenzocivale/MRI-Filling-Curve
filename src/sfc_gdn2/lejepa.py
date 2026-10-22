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
- token: for every masked patch of A that B also contains, the `bi` token (forward ++ backward
  output at that patch, as in the segmentation probe) of A and of B -> `tok` head, pulled together
  (symmetric, no stop-grad, as LeJEPA's invariance); `tok_joint_bn`: both through one head call, so its
  BatchNorm normalises them with the same statistics (off: two calls, A's masked tokens and all of B's apart);
  Token targets are foreground only (as Vol-JEPA's head mask): background patches stay in the
  sequence (with no position embedding their count is spatial signal) but are never masked, paired or
  sampled for SIGReg, so near-identical air tokens neither give free pairs nor a degenerate mass.
  SIGReg on B's foreground token embeddings, `tokens_per_volume` per volume (only the estimate's sample count:
  the weight is fixed by `sigreg_samples`; when SIGReg still scaled with it, ~20k tokens gave per-patch noise and
  #volumes let a per-volume code through).
- cells (`levels`, e.g. 2, 4, 8 patches = 32, 64, 128 mm per edge, aligned on the patch grid, so A and B
  share them whatever their boxes): mean foreground bi token per cell; every cell whose foreground is
  entirely masked in A is pulled to the same cell in B (`cell<c>` head); SIGReg on `tokens_per_volume` of
  B's cells per volume.
- volume (LeJEPA's multi-crop over the volume's boxes): the top of every view of every box -> `volume` head; each
  view pulled to the mean of the volume's global views; SIGReg per view slot, one sample per volume.
- session: the same over the scans of one session in the batch (the loader batches each scan with a random other
  scan of its session): what all its sequences share (anatomy, age, sex), not the acquisition signature that
  alone tells a volume's boxes from the rest of the batch. Its own head; SIGReg one sample per session.
  `top_levels` lists which of the two are on (both by default; [] for the A/B check). `volume_locals` false: the
  volume level over the global views only (a local crop cannot see the volume's size, which the pull asks it to).
View position (model.view_pos, `model.ViewPos`): each token gets its coordinates inside its own view's box after
the view's symmetry, so A and B place a patch differently and position alone cannot pair them.
loss = (1 - lam) * wmean(invariances) + lam * wmean(SIGRegs), one of each per scale, weight `w_top` for the view,
volume and session terms and 1 for token and cells.
SIGReg sees every rank's samples. Its statistic grows with their count M (a test statistic), which made every
term's weight its sample count (view ~1152 vs session ~72 per step): each is rescaled by sigreg_samples / M, so
every term weighs as one of sigreg_samples samples (768: the token term lam was tuned on) whatever the batch,
`tokens_per_volume`, groups or number of GPUs.
"""
from __future__ import annotations

import torch
import torch.distributed.nn
from torch import nn
from torch.utils.checkpoint import checkpoint

from .curves import grid_coords, keys, symmetries, transform
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
        def bn():
            return nn.BatchNorm1d(hidden, track_running_stats=False)
        super().__init__(nn.Linear(d_in, hidden), bn(), nn.GELU(),
                         nn.Linear(hidden, hidden), bn(), nn.GELU(), nn.Linear(hidden, d_out))


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
                 w_top: float = 1.0, w_tok: float = 1.0, top_levels: tuple = ("volume", "session"), rec_weight: float = 0.0,
                 rec_cells: int = 4, rec_eps: float = 0.01, rec_visible: float = 0.0, coarse_max: int = 1,
                 coarse_prob: float = 0.3, cov_dim: int = 0, tok_joint_bn: bool = False,
                 volume_locals: bool = True, child: int = 0, child_adaptive: bool = False,
                 child_frac: tuple = (0.25, 0.25), child_explore: float = 0.1, child_max: int = 216,
                 coarse_view: bool = False, tok_visible: float = 0.0, tok_self: float = 0.0):
        super().__init__()
        self.encoder, self.gg, self.gl, self.k, self.lam = encoder, groups_global, groups_local, 2, lam
        self.curves, self.seed = list(view_curves), seed
        self.max_view_tokens = float("inf") if max_view_tokens is None else max_view_tokens
        self.embed_chunk_voxels, self.max_packed_tokens = embed_chunk_voxels, max_packed_tokens
        self.checkpoint_tokens, self.tok_joint_bn, self.volume_locals = checkpoint_tokens, tok_joint_bn, volume_locals
        self.thick_prob, self.thick_mm = thick_prob, tuple(float(t) for t in thick_mm)
        self.coarse_max, self.coarse_prob = int(coarse_max), coarse_prob
        self.tokens_per_volume, self.sigreg_samples, self.w_top = tokens_per_volume, sigreg_samples, w_top
        self.w_tok = w_tok   # weight of the token (masked patch) term in both means; 0: its ablation (22/10)
        self.global_frac, self.local_edge, self.jitter, self.mask_ratio = global_frac, local_edge, jitter, mask_ratio
        self.gamma, self.scale, self.shift, self.noise, self.fg_threshold = gamma, scale, shift, noise, fg_threshold
        self.levels = [int(c) for c in levels]
        # iBOT++ (TIPSv2, 22/10 screens): token supervision beyond A's masked tokens. tok_visible w: A's visible foreground
        # tokens also pulled to the same patch in B (cheap; B differs in box, augmentation and curve); tok_self w: view A
        # read a second time unmasked (same box, augmentation, curve: the generator replayed) and every foreground token of
        # masked A pulled to its unmasked reading (TIPSv2's setting: same view, context removed). Weight w in the
        # invariance mean, beside the masked-token term (1); logged as inv_token_vis / inv_token_self(_m, _v).
        self.tok_visible, self.tok_self = float(tok_visible), float(tok_self)
        self.register_buffer("mask_cells", torch.tensor(mask_cells), persistent=False)
        self.glob = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.tok = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.cell = nn.ModuleList(Projector(2 * encoder.dim, proj_hidden, proj_dim) for _ in self.levels)
        self.top = nn.ModuleDict({n: Projector(2 * encoder.dim, proj_hidden, proj_dim) for n in top_levels})
        self.rec_weight, self.rec_cells, self.rec_eps, self.rec_visible = rec_weight, rec_cells, rec_eps, rec_visible
        self.rec = nn.Sequential(nn.Linear(2 * encoder.dim, proj_hidden), nn.GELU(),
                                 nn.Linear(proj_hidden, rec_cells ** 3)) if rec_weight or rec_visible else None
        self.sigreg = SIGReg(sigreg_slices, sigreg_knots)
        # child n (18/10, user): the pyramid also goes down. In the B view of each scan's first local group (at most
        # child_max patches) patches are read finer: n = 1, every patch as its 8 sub-blocks of patch_mm/2; n = 2,
        # also 64 sub-blocks of patch_mm/4. Each sub-block is embedded from the same native voxels (nothing resampled) by
        # the patch kernel zoomed onto it, and the view is read once by the encoder at mixed resolution along the
        # Hilbert curve of the finest grid (a cell's sub-blocks are consecutive on it). A's masked token predicts, through a slot
        # query per sub-block, the tokens of its sub-blocks (invariance + SIGReg, as the cells). child_adaptive: only
        # the patches (then sub-blocks) with the highest learned score are refined, child_frac of them, child_explore
        # of those at random; the score learns the sub-block error measured where it refined (the model goes finer
        # where its prediction of the detail fails).
        self.child, self.child_adaptive, self.child_frac = int(child), child_adaptive, tuple(child_frac)
        self.child_explore, self.child_max = child_explore, child_max
        # coarse_view (18/10, user): the pyramid's input also goes up. Each scan's first global B box is read once more
        # at twice the patch (2^3 patches' voxels as one block, the patch kernel zoomed out onto it, plus a learned
        # scale vector: ~8x fewer tokens) and its foreground mean joins the volume level as one more view
        self.coarse_view = coarse_view
        self.coarse_scale = nn.Parameter(torch.zeros(encoder.dim)) if coarse_view else None
        if self.child:
            d2 = 2 * encoder.dim
            # one kernel for every scale: a sub-block is embedded by the patch kernel zoomed onto it (its voxels at f
            # times their spacing span one patch), plus a learned vector per scale so the encoder knows which it reads
            self.child_scale = nn.Parameter(torch.zeros(self.child, encoder.dim))
            self.child_slot = nn.ParameterList(nn.Parameter(torch.randn(f ** 3, d2) * 0.02) for f in (2, 4)[: self.child])
            self.child_head = Projector(d2, proj_hidden, proj_dim)
            self.child_score = nn.ModuleList(nn.Linear(d2, 1) for _ in (2, 4)[: self.child]) if child_adaptive else None
        # cov_dim > 0: a second-order view level (view_cov), invariant across A/B and SIGReg'd like the view mean
        self.cov_proj = nn.Linear(2 * encoder.dim, cov_dim, bias=False) if cov_dim else None
        self.cov = Projector(cov_dim * (cov_dim + 1) // 2, proj_hidden, proj_dim) if cov_dim else None
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

    def boxes(self, fg: torch.Tensor, xyz: torch.Tensor, grid: torch.Tensor,
              gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """fg [B,N], coords [B,N,3], grid [B,3] -> group boxes (corner, edge) [B,G,3], globals first.
        Centres on foreground patches; global edges span global_frac of the foreground bounding box
        (volume fraction), local edges are log-uniform in local_edge per axis."""
        b, g = len(fg), grid[:, None]
        big = torch.where(fg[..., None], xyz, 1 << 30).amin(1), torch.where(fg[..., None], xyz, -1).amax(1)
        extent = (big[1] - big[0] + 1).clamp_min(1)                                       # [B,3]
        frac = self._u((b, self.gg, 1), *self.global_frac, gen) ** (1 / 3)
        e_glob = self._cap((extent[:, None] * frac).round(), self.max_view_tokens)
        lo, hi = torch.tensor(self.local_edge, dtype=torch.float).log()
        # ponytail: independent per-axis edges (aspect in [3/8, 8/3]); add an explicit aspect prior if needed
        e_loc = self._u((b, self.gl, 3), lo, hi, gen).exp().round()
        edge = torch.cat([e_glob, e_loc], 1).clamp_min(1).minimum(g).long()               # [B,G,3]
        centre = xyz.gather(1, torch.multinomial(fg.float() + 1e-6 * (xyz[..., 0] >= 0), self.gg + self.gl, replacement=True,
                                                 generator=gen)[..., None].expand(-1, -1, 3))
        return (centre - edge // 2).clamp(min=0).minimum(g - edge), edge

    def jittered(self, corner, edge, grid, gen):
        """K views per group: each box rescaled and shifted by up to +-jitter of its edge -> [B,G,K,3]."""
        g, shape = grid[:, None, None], (*edge.shape[:2], self.k, 3)
        e = (edge[:, :, None] * self._u(shape, -self.jitter, self.jitter, gen).exp()).round().clamp_min(1)
        e = self._cap(e, self.max_view_tokens).minimum(g).long()
        c = corner[:, :, None] + (self._u(shape, -self.jitter, self.jitter, gen) * edge[:, :, None]).round().long()
        return c.clamp(min=0).minimum(g - e), e

    def serialize(self, grid, corner, edge, n, tot, gen):
        """Boxes (corner, edge) [V,3] in patch grids [V,3], n [V] patches per box (tot in all, host) -> every view's
        patches in the order of a random curve of `view_curves` under a random cube symmetry, traced within the box,
        packed view after view (no padding): view [T], patch index in its scan's raster [T], coordinates [T,3] in the
        scan and in the box after the symmetry (the view position)."""
        dev, nv = corner.device, len(corner)
        sym = torch.randint(len(self.perms), (nv,), device=dev, generator=gen)
        curve = torch.randint(len(self.curves), (nv,), device=dev, generator=gen)
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

    def read(self, vols, ws, fg, n_patch, vol, corner, edge, masked, gen):
        """Views [V] (scan `vol`, box) -> their tokens packed view after view, each along its curve (no padding):
        view [T], patch index in its scan [T], patch coords [T,3], foreground [T], mask [T], bi tokens [T,2d]; tokens
        per view [V] and per-view offsets [V] (device) and the tokens per view (host). Masking applies only to
        foreground patches of views with `masked`; the backward pass reads each view in reverse. Tokens are embedded
        per scan (its own voxels per patch and kernel weights `ws`). Host syncs: a handful per call."""
        nv, dev = len(corner), corner.device
        n = edge.prod(-1)
        vol_h, lens = torch.stack([vol, n]).tolist()
        tot = sum(lens)
        tv, idx, pxyz, vpos = self.serialize(vols["grid"][vol], corner, edge, n, tot, gen)
        fgv = fg[vol[tv], idx]
        u = lambda lo, hi: self._u((nv, 1, 1), lo, hi, gen)
        gamma, scale, shift, noise = u(-self.gamma, self.gamma).exp(), u(1 - self.scale, 1 + self.scale), \
            u(-self.shift, self.shift), u(0, self.noise)
        on = torch.rand(nv, device=dev, generator=gen) < self.thick_prob
        axis = torch.randint(3, (nv,), device=dev, generator=gen)
        mm = torch.randint(len(self.thick_mm), (nv,), device=dev, generator=gen)
        # masked in aligned cubes of one of mask_cells per view (<= half its shortest edge): big cubes are the
        # coarse levels' prediction targets
        cells = self.mask_cells[None]
        ok_c = cells <= (edge.amin(-1, keepdim=True) // 2).clamp_min(1)
        c = cells[0][torch.rand(ok_c.shape, device=dev, generator=gen).masked_fill(~ok_c, -1).argmax(1)]
        cid = self.cell_ids(pxyz, vols["grid"][vol[tv]], c[tv])
        draw = torch.rand(nv, n_patch, device=dev, generator=gen)[tv, cid]
        mask = masked[tv] & fgv & (draw < u(*self.mask_ratio).view(-1)[tv])
        aug = (gamma.view(-1), scale.view(-1), shift.view(-1), noise.view(-1))
        coarse = []      # coarse_max > 1: in coarse_prob of the views, each axis averaged in blocks of 1..coarse_max
        if self.coarse_max > 1:  # native voxels (a coarser acquisition, nothing interpolated); off: no draw at all
            c_on = torch.rand(nv, device=dev, generator=gen) < self.coarse_prob
            f3 = torch.randint(1, self.coarse_max + 1, (nv, 3), device=dev, generator=gen)
            coarse = [torch.where(c_on[:, None], f3, 1).flatten()]
        host = torch.cat([on.long(), axis, mm, *coarse, torch.randint(1 << 62, (1,), device=dev, generator=gen)]).tolist()
        on_h, axis_h, mm_h, seed = host[:nv], host[nv:2 * nv], host[2 * nv:3 * nv], host[-1]   # one host sync
        co_h = host[3 * nv:6 * nv] if coarse else None
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
            for a in range(3) if co_h else ():
                if (f := co_h[3 * i + a]) >= 2:
                    slabs.setdefault(b, []).append((offs[i], offs[i + 1], a, f))
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

    def forward(self, vols: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """vols (`dataset.to_device`): patches = per scan [N_b, P_b] native voxel values, spacing [B,3] mm,
        k [B,3] voxels per patch axis, grid [B,3] patch grid, session [B] (any ids)."""
        grid, gen = vols["grid"], generator
        b, n_patch = len(grid), int(grid.prod(1).max())
        xyz = grid_coords(grid, n_patch)                                                  # [B,N,3]
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
        def rec_target(scan, pid):
            """Sub-patch targets of the rows (scan [R], patch [R]): each patch's native voxels in rec_cells^3 cells,
            normalised per patch. Only the rows a loss uses (22/10: all patches of the batch held a copy of every voxel
            at 4 mm cells, ~10 GB and OOM)."""
            out = None
            with torch.no_grad(), torch.autocast(grid.device.type, enabled=False):
                for b in scan.unique().tolist():
                    s = (scan == b).nonzero()[:, 0]
                    x, k = vols["patches"][b], vols["k_list"][b]
                    t = nn.functional.adaptive_avg_pool3d(x[pid[s]].view(-1, 1, *k), self.rec_cells).flatten(1)
                    t = (t - t.mean(1, keepdim=True)) / (t.var(1, keepdim=True, unbiased=False) + self.rec_eps).sqrt()
                    out = t.new_empty(len(scan), t.shape[1]) if out is None else out
                    out[s] = t
            return out
        rec, recv = [], []
        corner, edge = self.jittered(*self.boxes(fg, xyz, grid, gen), grid, gen)           # [B,G,2,3]
        vol = torch.arange(b, device=grid.device)[:, None, None].expand(-1, corner.shape[1], 2)
        zg, pred, tgt, ztok, tvol, tfg, lens, tops, zc = [], [], [], [], [], [], [], [], []
        vpred, vtgt, spred, stgt, smask = [], [], [], [], []
        cpred, ctgt, cz, cvol = ([[] for _ in self.levels] for _ in range(4))
        for part in (slice(0, self.gg), slice(self.gg, None)):                             # globals / locals
            c, e, v = (t[:, part].reshape(-1, *t.shape[3:]) for t in (corner, edge, vol))  # views A,B interleaved
            nv, dev = len(c), c.device
            masked = torch.arange(nv, device=dev) % 2 == 0
            if self.tok_self:   # the same draws again for the unmasked reading
                gen_self = torch.Generator(gen.device)
                gen_self.set_state(gen.get_state())
            tv, idx, pxyz, fgv, mask, bi, n, off, n_h = self.read(vols, ws, fg, n_patch, v, c, e, masked, gen)
            if self.tok_self:   # the target reading, without gradient (TIPSv2's teacher pass; memory as one read)
                with torch.no_grad():
                    bi_u = self.read(vols, ws, fg, n_patch, v, c, e, torch.zeros_like(masked), gen_self)[5]
            has = torch.zeros(nv, device=dev).index_add_(0, tv, fgv.float()) > 0
            w = (fgv | ~has[tv]).float()                                                   # foreground (all if none)
            top = torch.zeros(nv, bi.shape[-1], device=dev).index_add_(0, tv, bi.float() * w[:, None]) \
                / torch.zeros(nv, device=dev).index_add_(0, tv, w)[:, None]
            zg.append(self.glob(top).float().view(b, -1, 2, self.glob[-1].out_features))
            tops.append(top.view(b, -1, top.shape[-1]))
            if self.cov is not None:
                zc.append(self.cov(self.view_cov(bi, tv, w, off, n_h)).float().view(b, -1, 2, self.cov[-1].out_features))
            for i, (cs, head) in enumerate(zip(self.levels, self.cell)):
                p, q, zb, vb = self.cells(bi, tv, fgv, mask, pxyz, grid[v[tv]], cs, n_patch, head)
                cpred[i].append(p)
                ctgt[i].append(q)
                cz[i].append(zb)
                cvol[i].append(v[1::2][vb])
            ia = _ranges(off[0::2], n[0::2], sum(n_h[0::2]))                               # A's tokens, B's tokens
            ib = _ranges(off[1::2], n[1::2], sum(n_h[1::2]))
            row_b = torch.full((nv // 2, n_patch), -1, device=dev)                         # (pair, patch) -> row in B
            row_b[tv[ib] // 2, idx[ib]] = torch.arange(len(ib), device=dev)
            j_b = row_b[tv[ia] // 2, idx[ia]]                                              # A's patch -> its row in B
            pair = mask[ia] & (j_b >= 0)                                                   # mask is foreground only
            # token head recomputed in backward for big scans: whole-volume views give ~10^5 tokens x 1024 hidden
            # (deterministic: batch-statistics BatchNorm)
            head = (lambda x: checkpoint(self.tok, x, use_reentrant=False)) if len(bi) > self.checkpoint_tokens \
                else self.tok
            xa = bi.index_select(0, ia[pair])                                              # A's masked, paired
            if self.tok_joint_bn:  # one call: A's predictions and B's tokens under the same BatchNorm statistics
                z = head(torch.cat([xa, bi.index_select(0, ib)])).float()
                za, z_b = z[: len(xa)], z[len(xa):]
            else:  # two calls (runs before 16/10): B's statistics hold its background, A's only masked foreground,
                z_b, za = head(bi.index_select(0, ib)).float(), head(xa).float()   # and a perfect one scores > 0
            pred.append(za)
            tgt.append(z_b[j_b[pair]])
            if self.tok_visible:   # A's visible foreground tokens, paired with B
                vis = ~mask[ia] & fgv[ia] & (j_b >= 0)
                vpred.append(head(bi.index_select(0, ia[vis])).float())
                vtgt.append(z_b[j_b[vis]])
            if self.tok_self:      # every foreground token of masked A against its unmasked reading
                fa = fgv[ia]
                spred.append(head(bi.index_select(0, ia[fa])).float())
                stgt.append(head(bi_u.index_select(0, ia[fa])).float())
                smask.append(mask[ia][fa])
            if self.child and part.start == self.gg:
                child_out = self.sub_blocks(vols, ws, bi, tv, pxyz, ia[pair], c, e, v, gen)
            ztok.append(z_b)
            tvol.append(v[tv[ib]])
            tfg.append(fgv[ib])
            lens.append(e.prod(-1).float().mean())
            # masked tokens (A's foreground) -> their patch's sub-patch layout (MAE-style); rec_visible: B's foreground
            for w, r, acc in ((self.rec_weight, mask, rec), (self.rec_visible, fgv & (tv % 2 == 1), recv)):
                if w:
                    r = r.nonzero()[:, 0]
                    if len(r):
                        acc.append((self.rec(bi.index_select(0, r)).float() - rec_target(v[tv[r]], idx[r])).square().mean(1))
        zg = torch.cat(zg, 1)                                                              # [B,G,2,D]
        inv_g = (zg - zg.mean(2, keepdim=True)).square().mean()
        # symmetric, as LeJEPA: both tokens pulled to their mean, no stop-grad (with a stop-grad target and
        # no EMA teacher the target drifted and inv_token grew 0.19 -> 2.7); SIGReg prevents collapse
        inv_t = (torch.cat(pred) - torch.cat(tgt)).square().mean() / 4                     # = mean ||z - mean||^2
        ztok, tvol, tfg = torch.cat(ztok), torch.cat(tvol), torch.cat(tfg)
        # equal foreground tokens per volume (with replacement: fixed shape for the cross-GPU gather); a
        # volume whose B views hold no foreground (boxes are centred on it, so only after jitter) falls back to all
        keep = self._sample(tvol, tfg, b, gen)

        def sig(z):  # every rank's samples (differentiable gather), weighed as sigreg_samples whatever their count
            z = gather_rows(z) if torch.distributed.is_initialized() else z
            return self.sigreg(z, gen) * (self.sigreg_samples / z.shape[-2])

        sig_g, sig_t = sig(zg.permute(2, 0, 1, 3).reshape(2, -1, zg.shape[-1])), sig(ztok[keep][None])
        out = {}
        if zc:
            zc = torch.cat(zc, 1)
            out["inv_cov"] = (zc - zc.mean(2, keepdim=True)).square().mean()
            out["sigreg_cov"] = sig(zc.permute(2, 0, 1, 3).reshape(2, -1, zc.shape[-1]))
        for cs, p, q, z, vb in zip(self.levels, cpred, ctgt, cz, cvol):
            p, q, z, vb = torch.cat(p), torch.cat(q), torch.cat(z), torch.cat(vb)
            out[f"inv_cell{cs}"] = (p - q).square().mean(-1).sum() / max(len(p), 1) / 4
            out[f"sigreg_cell{cs}"] = sig(z[self._sample(vb, torch.ones_like(vb, dtype=torch.bool), b, gen)][None])
            out[f"pairs_cell{cs}"] = torch.tensor(float(len(p)), device=z.device)
        if self.coarse_view:   # scans too finely sampled for the coarse block keep their fine view (one more copy)
            ct, big = self.coarse_tops(vols, fg, corner, edge)
            fine = tops[0][:, 1].clone()
            if ct is not None:
                fine[[i for i in range(b) if i not in big]] = ct
            tops.append(fine[:, None])
        tops = torch.cat(tops, 1)                                                          # [B,V,2d], globals first
        groups = {"volume": torch.arange(b, device=grid.device),
                  "session": vols["session"].unique(return_inverse=True)[1]}
        for name, head in self.top.items():
            grp = groups[name]
            n = int(grp.max()) + 1
            t = tops if name != "volume" or self.volume_locals else tops[:, : 2 * self.gg]  # off: globals only
            z = head(t.flatten(0, 1)).float().view(b, -1, head[-1].out_features)          # [B,V,D]
            cnt = torch.bincount(grp, minlength=n).clamp_min(1)[:, None]
            centre = z.new_zeros(n, z.shape[-1]).index_add_(0, grp, z[:, : 2 * self.gg].mean(1)) / cnt
            out[f"inv_{name}"] = (z - centre[grp][:, None]).square().mean()
            per = z.new_zeros(n, *z.shape[1:]).index_add_(0, grp, z) / cnt[..., None]      # one sample per group
            out[f"sigreg_{name}"] = sig(per.transpose(0, 1))                               # [V,n,D]: per view slot
        with torch.no_grad():                                   # between-volume share of token-embedding variance
            zt, tv = ztok[keep], tvol[keep]
            means = torch.zeros(b, zt.shape[1], device=zt.device).index_add_(0, tv, zt)
            means /= torch.bincount(tv, minlength=b).clamp_min(1)[:, None]
            vol_share = 1 - (zt - means[tv]).var(0).sum() / zt.var(0).sum()
        cov = ["cov"] if self.cov is not None else []
        ch = []
        if self.child:
            for name, (zp, zt, score_loss) in child_out.items():
                out[f"inv_{name}"] = (zp - zt).square().mean() / 4 if len(zp) else zp.sum() * 0
                pick = torch.randperm(len(zt), device=zt.device, generator=gen)[: self.tokens_per_volume * b]
                out[f"sigreg_{name}"] = sig(zt[pick][None])
                out[f"n_{name}"] = torch.tensor(float(len(zp)), device=zt.device)
                if score_loss is not None:
                    out[f"score_{name}"] = score_loss
                ch.append(name)
        extra, w_extra = [], []
        if self.tok_visible:
            out["inv_token_vis"] = (torch.cat(vpred) - torch.cat(vtgt)).square().mean() / 4
            extra.append(out["inv_token_vis"]), w_extra.append(self.tok_visible)
        if self.tok_self:
            d = (torch.cat(spred) - torch.cat(stgt)).square().mean(1) / 4
            m = torch.cat(smask)
            out["inv_token_self"] = d.mean()
            out["inv_token_self_m"], out["inv_token_self_v"] = d[m].mean() if m.any() else d.sum() * 0, \
                d[~m].mean() if (~m).any() else d.sum() * 0
            extra.append(out["inv_token_self"]), w_extra.append(self.tok_self)
        inv = [inv_g, inv_t, *(out[f"inv_cell{c}"] for c in self.levels), *(out[f"inv_{n}"] for n in [*self.top, *cov]),
               *(out[f"inv_{n}"] for n in ch), *extra]
        sigs = [sig_g, sig_t, *(out[f"sigreg_cell{c}"] for c in self.levels),
                *(out[f"sigreg_{n}"] for n in [*self.top, *cov]), *(out[f"sigreg_{n}"] for n in ch)]
        w = [self.w_top, self.w_tok, *(1.0 for _ in self.levels), *(self.w_top for _ in [*self.top, *cov]), *(1.0 for _ in ch), *w_extra]
        ws_ = w[: len(sigs)]   # the extra invariances add no SIGReg (their embeddings are the token head's)
        wmean = lambda xs: sum(a * x for a, x in zip(w if len(xs) == len(w) else ws_, xs)) / sum(w if len(xs) == len(w) else ws_)
        for k, r in (("rec", rec), ("rec_visible", recv)):
            if r:
                out[k] = (r := torch.cat(r)).sum() / max(len(r), 1)
        score = sum(out[f"score_{n}"] for n in ch if f"score_{n}" in out)   # trains only the score heads
        if self.child:  # every child parameter in the graph each step (DDP; a 4 mm block may be absent in a batch)
            score = score + 0 * sum(q.view(-1)[0] for q in [self.child_scale, *self.child_slot,
                                                              *self.child_head.parameters(),
                                                              *(self.child_score.parameters() if self.child_score else [])])
        return {"loss": (1 - self.lam) * wmean(inv) + self.lam * wmean(sigs) + self.rec_weight * out.get("rec", 0) + score
                + self.rec_visible * out.get("rec_visible", 0),
                "inv_global": inv_g, "inv_token": inv_t, "sigreg_global": sig_g, "sigreg_token": sig_t, **out,
                "vol_share": vol_share, "len_global": lens[0], "len_local": lens[1]}

    def view_cov(self, bi, tv, w, off, n_h):
        """Order-free second-order reading of every view (notes/loss_stato.md: the covariance of the tokens holds age
        and sex that their mean drops): its foreground tokens (weights w) projected to cov_dim, their covariance, its
        square root by Newton-Schulz with trace normalisation (iSQRT-COV) -> upper triangle [V, r(r+1)/2]."""
        y, nv = self.cov_proj(bi).float(), len(n_h)                       # [T, r]: no fp32 copy of bi
        with torch.autocast(bi.device.type, enabled=False):
            s = torch.zeros(nv, device=y.device).index_add_(0, tv, w)
            m = torch.zeros(nv, y.shape[1], device=y.device).index_add_(0, tv, y * w[:, None]) / s[:, None]
            pad = y.new_zeros(nv, max(n_h), y.shape[1])                    # views padded: one bmm for every view
            pad[tv, torch.arange(len(tv), device=y.device) - off[tv]] = (y - m[tv]) * w[:, None].sqrt()
            eye = torch.eye(y.shape[1], device=y.device).expand(nv, -1, -1)
            c = pad.transpose(1, 2) @ pad / s[:, None, None] + 1e-4 * eye
            tr = c.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]
            a, z = c / tr, eye
            for _ in range(10):                                            # ~1 % from the exact root (5: ~20 %)
                t = 0.5 * (3 * eye - z @ a)
                a, z = a @ t, t @ z
            iu = torch.triu_indices(y.shape[1], y.shape[1], device=y.device)
            return (a * tr.sqrt())[:, iu[0], iu[1]]

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

    def coarse_tops(self, vols, fg, corner, edge):
        """Each scan's first global B box at twice the patch (coarse_view) -> its foreground mean token [B,2d]."""
        dev, seqs, fgs, memo = corner.device, [], [], {}
        off = torch.stack(torch.meshgrid(*(torch.arange(2, device=dev),) * 3, indexing="ij"), -1).view(-1, 3)
        big = []
        for b in range(len(corner)):
            c0, e0, g = corner[b, 0, 1], edge[b, 0, 1], vols["grid"][b]
            k, sp = vols["k_list"][b], vols["spacing_list"][b]
            if 8 * k[0] * k[1] * k[2] > 1 << 18:   # a 2^3-patch block of > 64^3 voxels: its kernel matrix alone is GBs
                big.append(b)
                continue
            ec = (e0 + 1) // 2
            cc = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev) for n in ec.tolist()), indexing="ij"), -1).view(-1, 3)
            pc = c0 + cc[:, None] * 2 + off[None]                                            # [Nc,8,3] its patches
            inside = ((pc < g) & (pc < c0 + e0)).all(-1)
            pid = ((pc[..., 0] * g[1] + pc[..., 1]) * g[2] + pc[..., 2]).clamp(max=int(g.prod()) - 1)
            if (key := (*sp, *k)) not in memo:
                memo[key] = self.encoder.patch_embed.weights(torch.tensor(sp, device=dev) / 2, [2 * n for n in k])
            w, x = memo[key], vols["patches"][b]

            def emb(pid, inside, w=w, x=x, k=k):   # 2^3 patches' voxels as one x-major block -> its token
                blk = (x[pid].float() * inside[..., None]).view(len(pid), 2, 2, 2, *k)
                return self.encoder.patch_embed(blk.permute(0, 1, 4, 2, 5, 3, 6).reshape(len(pid), -1), w)
            n = max(1, self.embed_chunk_voxels // w.shape[0])     # chunks recomputed in backward: memory ~ blocks
            t = torch.cat([checkpoint(emb, pid[i:i + n], inside[i:i + n], use_reentrant=False)
                           for i in range(0, len(pid), n)]) + self.coarse_scale
            order = keys("hilbert", cc, ec, self.seed).argsort()
            seqs.append(t[order])
            fgs.append((fg[b][pid] & inside).any(1)[order])
        lens = [len(q) for q in seqs]
        if not lens:
            return None, big
        cu = nn.functional.pad(torch.tensor(lens, device=dev).cumsum(0), (1, 0)).int()
        tops = []
        for h, m in zip(self.encoder.flat(torch.cat(seqs), cu, self.encoder.grad_checkpoint).split(lens), fgs):
            h = h[m] if bool(m.any()) else h
            tops.append(torch.cat([h.mean(0), h.mean(0)]))                                 # one stream: bi = h ++ h
        return torch.stack(tops).float(), big

    def sub_blocks(self, vols, ws, bi, tv, pxyz, pa, corner, edge, vol, gen):
        """`child` levels (see __init__) for the locals part: A's masked paired rows `pa` ->
        {name: (predicted [M,D], target [M,D], score loss or None)} for child8 (and child4)."""
        dev, gl = bi.device, self.gl
        pv = tv[pa] + 1                                                    # B view of each pair
        first = pv % (2 * gl) == 1                                         # the first local group of its scan
        views = torch.unique(pv[first]).tolist()
        F = 2 ** self.child
        memo = {}   # kernel weights per (spacing, scale, sub-block shape): few distinct, each einsum kept for backward
        up8 = torch.tensor([(a // 2) * 4 + (b2 // 2) * 2 + c // 2 for a in range(4) for b2 in range(4) for c in range(4)], device=dev)
        seqs, metas = [], []
        for j in views:
            b, c0, e0 = int(vol[j]), corner[j], edge[j]
            k, sp = vols["k_list"][b], vols["spacing_list"][b]
            if int(e0.prod()) > self.child_max or min(k) < 2:
                continue
            g = vols["grid"][b]
            rows = pa[first & (pv == j)]                                   # A rows paired in this view
            box = torch.stack(torch.meshgrid(*(torch.arange(n, device=dev) for n in e0.tolist()), indexing="ij"), -1)
            pc = (c0 + box.view(-1, 3))                                    # every patch of the view (context)
            pid = (pc[:, 0] * g[1] + pc[:, 1]) * g[2] + pc[:, 2]
            loc = ((pxyz[rows] - c0) * torch.tensor([e0[1] * e0[2], e0[2], 1], device=dev)).sum(-1)
            lv = [torch.zeros(len(pc), dtype=torch.bool, device=dev) for _ in range(self.child)]
            ok4 = min(k) >= 4
            # which patches (and sub-blocks) go finer: all of the paired ones, or the top scored (+ some at random)
            if self.child_adaptive:
                s1 = self.child_score[0](bi[rows].detach()).squeeze(-1).float()   # A's own token, no grad to the encoder
                n1 = max(1, round(self.child_frac[0] * len(rows)))
                r = torch.rand(len(rows), device=dev, generator=gen) < self.child_explore
                top = (s1 + 1e4 * r).topk(min(n1, len(rows))).indices
                lv[0][loc[top]] = True
            else:
                lv[0][loc] = True
            x = vols["patches"][b][pid].float().view(-1, *k)
            toks, origin, size, owner, slot1, slot2 = [], [], [], [], [], []
            sub4 = None
            if self.child == 2 and ok4:   # which 8 mm sub-blocks of the refined patches go to 4 mm
                ref = lv[0].nonzero()[:, 0]
                if self.child_adaptive:
                    s2 = self.child_score[1](bi[rows].detach()[:, None] + self.child_slot[0].detach()[None]).float()
                    sc = torch.full((len(pc), 8), -1e9, device=dev)
                    sc[loc] = s2.squeeze(-1)
                    sc = sc[ref].flatten()
                    n2 = max(1, round(self.child_frac[1] * len(sc)))
                    r = torch.rand(len(sc), device=dev, generator=gen) < self.child_explore
                    pick = (sc + 1e4 * r).topk(min(n2, len(sc))).indices
                else:
                    pick = torch.arange(len(ref) * 8, device=dev)
                sub4 = torch.zeros(len(pc), 8, dtype=torch.bool, device=dev)
                sub4[ref[pick // 8], pick % 8] = True
            for f, name in ((1, None), (2, "2"), (4, "4")):
                sizes = [[n // f + (r < n % f) for r in range(f)] for n in k]
                if f == 1:
                    sel = ~lv[0]
                    t = self.encoder.patch_embed(x[sel].flatten(1), ws[b])
                    toks.append(t); origin.append(box.view(-1, 3)[sel] * F); size.append(F); owner.append(sel.nonzero()[:, 0])
                    slot1.append(torch.full((int(sel.sum()),), -1, device=dev)); slot2.append(slot1[-1])
                    continue
                if f == 4 and sub4 is None:
                    continue
                emb, zs = self.encoder.patch_embed, self.child_scale[0 if f == 2 else 1]
                parts = x.split(sizes[0], 1)
                for ox in range(f):
                    py = parts[ox].split(sizes[1], 2)
                    for oy in range(f):
                        pz = py[oy].split(sizes[2], 3)
                        for oz in range(f):
                            c8 = int(up8[(ox * 4 + oy) * 4 + oz]) if f == 4 else ox * 4 + oy * 2 + oz
                            sel = lv[0] & (sub4[:, c8] if f == 4 else (~sub4[:, c8] if sub4 is not None else True))
                            if not bool(sel.any()):
                                continue
                            shp = (sizes[0][ox], sizes[1][oy], sizes[2][oz])
                            if (key := (*sp, f, *shp)) not in memo:
                                memo[key] = emb.weights(torch.tensor(sp, device=dev) * f, list(shp))
                            w = memo[key]
                            toks.append(emb(pz[oz][sel].flatten(1), w) + zs)
                            o = box.view(-1, 3)[sel] * F + torch.tensor([ox, oy, oz], device=dev) * (F // f)
                            origin.append(o); size.append(F // f); owner.append(sel.nonzero()[:, 0])
                            s8 = torch.full((int(sel.sum()),), c8, device=dev)
                            slot1.append(s8 if f == 2 else torch.full_like(s8, -1))
                            slot2.append(torch.full_like(s8, (ox * f + oy) * f + oz) if f == 4 else torch.full_like(s8, -1))
            t = torch.cat(toks)
            # order: the Hilbert key of each token's first cell on the finest grid (blocks are contiguous on it)
            dims = e0 * F
            key = torch.cat([min_key(o, s, dims, self.seed) for o, s in zip(origin, size)])
            order = key.argsort()
            seqs.append(t[order])
            metas.append((rows, loc, torch.cat(owner)[order], torch.cat(slot1)[order], torch.cat(slot2)[order], lv, sub4))
        if not seqs:
            z = bi.new_zeros(0, self.child_head[-1].out_features)
            return {f"child{8 // (2 ** i)}": (z, z, None) for i in range(self.child)}
        lens = [len(q) for q in seqs]
        cu = nn.functional.pad(torch.tensor(lens, device=dev).cumsum(0), (1, 0)).int()
        h = self.encoder.flat(torch.cat(seqs), cu, self.encoder.grad_checkpoint).split(lens)
        out = {}
        for lvl, (name, f) in enumerate((("child8", 2), ("child4", 4))[: self.child]):
            q_in, t_in, err_at = [], [], []
            for (rows, loc, owner, s1, s2, lv, sub4), hv in zip(metas, h):
                slot = s1 if f == 2 else s2
                m = slot >= 0
                if not bool(m.any()):
                    continue
                pos = torch.full((int(loc.max()) + 1 if len(loc) else 1,), -1, device=dev)
                pos[loc] = torch.arange(len(loc), device=dev)
                own = owner[m]
                r = pos[own.clamp(max=len(pos) - 1)]
                keep = (own < len(pos)) & (r >= 0)                         # sub-blocks of A's masked paired patches
                hb = torch.cat([hv[m][keep], hv[m][keep]], -1)             # one stream: bi = h ++ h
                q_in.append(bi[rows[r[keep]]] + self.child_slot[lvl][slot[m][keep]])
                t_in.append(hb)
                err_at.append((rows[r[keep]], slot[m][keep]))
            if not q_in:
                z = bi.new_zeros(0, self.child_head[-1].out_features)
                out[name] = (z, z, None)
                continue
            q_in, t_in = torch.cat(q_in), torch.cat(t_in)
            z = self.child_head(torch.cat([q_in, t_in])).float()
            zp, zt = z[: len(q_in)], z[len(q_in):]
            score_loss = None
            if self.child_adaptive:   # the score learns the measured error of the sub-blocks it chose to refine
                er = (zp - zt).detach().square().mean(-1)
                ar = torch.cat([a for a, _ in err_at])
                if f == 2:
                    per = torch.zeros(len(bi), device=dev).index_add_(0, ar, er) / \
                        torch.zeros(len(bi), device=dev).index_add_(0, ar, torch.ones_like(er)).clamp_min(1)
                    u = torch.unique(ar)
                    sb = bi[u].detach()
                    score_loss = (self.child_score[0](sb).squeeze(-1).float() - per[u]).square().mean()
                else:
                    sl = up8[torch.cat([s for _, s in err_at])]             # their 8 mm parent sub-block
                    sb = bi[ar].detach() + self.child_slot[0].detach()[sl]
                    score_loss = (self.child_score[1](sb).squeeze(-1).float() - er).square().mean()
            out[name] = (zp, zt, score_loss)
        return out

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


def min_key(origin: torch.Tensor, size: int, dims: torch.Tensor, seed: int) -> torch.Tensor:
    """Hilbert key of a block of size^3 cells at `origin` [M,3] on a grid `dims`: the smallest key of its cells (an
    aligned block of a power-of-two size is contiguous on the curve, so this orders mixed-size blocks along it)."""
    off = torch.stack(torch.meshgrid(*(torch.arange(size, device=origin.device),) * 3, indexing="ij"), -1).view(-1, 3)
    return keys("hilbert", (origin[:, None] + off[None]).view(-1, 3), dims, seed).view(len(origin), -1).amin(1)


def _ranges(start: torch.Tensor, n: torch.Tensor, total: int) -> torch.Tensor:
    """Indices of the ranges [start_i, start_i + n_i) back to back; total = sum(n), known on the host (no sync)."""
    r = torch.repeat_interleave(torch.arange(len(n), device=n.device), n, output_size=total)
    return start[r] + torch.arange(total, device=n.device) - (n.cumsum(0) - n)[r]
