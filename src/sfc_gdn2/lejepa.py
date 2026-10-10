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
  (symmetric, no stop-grad, as LeJEPA's invariance);
  Token targets are foreground only (as Vol-JEPA's head mask): background patches stay in the
  sequence (with no position embedding their count is spatial signal) but are never masked, paired or
  sampled for SIGReg, so near-identical air tokens neither give free pairs nor a degenerate mass.
  SIGReg on B's foreground token embeddings: `tokens_per_volume` per volume, so the sample count stays in
  LeJEPA's calibrated range (scaled by ~20k tokens it gave per-patch noise; by #volumes it let a
  per-volume code through).
- cells (`levels`, e.g. 2, 4, 8 patches = 32, 64, 128 mm per edge, aligned on the patch grid, so A and B
  share them whatever their boxes): mean foreground bi token per cell; every cell whose foreground is
  entirely masked in A is pulled to the same cell in B (`cell<c>` head); SIGReg on `tokens_per_volume` of
  B's cells per volume.
loss = (1 - lam) * mean(invariances) + lam * mean(SIGRegs), one of each per scale.
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
        statistic: lam=0.05 assumes M in LeJEPA's range (~hundreds to ~1000 samples)."""
        with torch.autocast(z.device.type, enabled=False):  # cos/sin of bf16 projections is too coarse
            z = z.float()
            a = torch.randn(z.shape[-1], self.slices, device=z.device, generator=generator)
            a = a / a.norm(dim=0)
            xt = (z @ a).unsqueeze(-1) * self.t                      # [..., M, S, T]
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
                 sigreg_knots: int = 17, levels: tuple = (), mask_cells: tuple = (1,)):
        super().__init__()
        self.encoder, self.gg, self.gl, self.k, self.lam = encoder, groups_global, groups_local, 2, lam
        self.curves, self.seed = list(view_curves), seed
        self.max_view_tokens = float("inf") if max_view_tokens is None else max_view_tokens
        self.embed_chunk_voxels, self.max_packed_tokens = embed_chunk_voxels, max_packed_tokens
        self.checkpoint_tokens = checkpoint_tokens
        self.thick_prob, self.thick_mm = thick_prob, tuple(float(t) for t in thick_mm)
        self.tokens_per_volume = tokens_per_volume
        self.global_frac, self.local_edge, self.jitter, self.mask_ratio = global_frac, local_edge, jitter, mask_ratio
        self.gamma, self.scale, self.shift, self.noise, self.fg_threshold = gamma, scale, shift, noise, fg_threshold
        self.levels = [int(c) for c in levels]
        self.register_buffer("mask_cells", torch.tensor(mask_cells), persistent=False)
        self.glob = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.tok = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.cell = nn.ModuleList(Projector(2 * encoder.dim, proj_hidden, proj_dim) for _ in self.levels)
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

    def serialize(self, xyz, corner, edge, gen):
        """Coords [V,N,3], boxes [V,3] -> patch indices [V,L] in the order of a random curve of
        `view_curves` under a random cube symmetry, traced within the box; valid [V,L]."""
        inside = ((xyz >= corner[:, None]) & (xyz < (corner + edge)[:, None])).all(-1)     # [V,N]
        dev, dims = corner.device, edge[:, None]
        sym = torch.randint(len(self.perms), (len(corner),), device=dev, generator=gen)[:, None]
        local, dims = transform((xyz - corner[:, None]).clamp_min(0).minimum(dims - 1), dims,
                                self.perms[sym], self.flips[sym])
        curve = torch.randint(len(self.curves), (len(corner), 1), device=dev, generator=gen)
        key = torch.zeros_like(inside, dtype=torch.long)
        for i, name in enumerate(self.curves):
            key = torch.where(curve == i, keys(name, local, dims, self.seed), key)
        order = key.masked_fill(~inside, 1 << 62).argsort(1)
        n = inside.sum(1)
        order = order[:, : int(n.max())]
        return order, torch.arange(order.shape[1], device=order.device) < n[:, None]

    @staticmethod
    def cell_ids(p: torch.Tensor, grid: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Patch coords p [V,L,3] in grids [V,3] -> id [V,L] of the aligned cube of c [V] patches per edge holding
        each patch (ids < the grid's patch count)."""
        gc = (grid + c[:, None] - 1) // c[:, None]
        q = p.clamp_min(0) // c[:, None, None]
        return (q[..., 0] * gc[:, 1, None] + q[..., 1]) * gc[:, 2, None] + q[..., 2]

    def thick(self, v: torch.Tensor, k, spacing, on, axis, mm) -> torch.Tensor:
        """Patches [R,P] of one scan (k voxels at `spacing` mm); per row with `on`: native slices along `axis`
        averaged in consecutive groups of round(mm / spacing) (a thick-slice acquisition at ~mm; nothing when
        the scan's slices are already that thick)."""
        for a in range(3):
            for t in self.thick_mm:
                if (f := round(t / spacing[a])) < 2:
                    continue
                g = torch.arange(k[a], device=v.device) // f
                m = (g[:, None] == g[None, :]).float()
                x = v.unflatten(-1, tuple(k)).movedim(1 + a, -1)
                with torch.autocast(v.device.type, enabled=False):              # voxel values stay fp32
                    y = (x @ (m / m.sum(0))).movedim(-1, 1 + a).flatten(-3)
                v = torch.where((on & (axis == a) & (mm == t))[:, None], y, v)   # no host sync (every row computed)
        return v

    def _embed(self, x, pid, vid, w, aug, k, spacing, seed):
        """Patches `pid` of one scan's [N,P] voxel values, each augmented with the parameters of its view `vid`
        (aug: per-view tensors; noise drawn from `seed`, so a recomputation in backward is identical) -> [R,d]."""
        gamma, scale, shift, noise, on, axis, mm = (a[vid] for a in aug)
        v = x[pid]
        v = v.sign() * v.abs() ** gamma[:, None] * scale[:, None] + shift[:, None]
        g = torch.Generator(v.device).manual_seed(seed)
        v = v + torch.randn(v.shape, device=v.device, generator=g) * noise[:, None]
        if self.thick_prob:
            v = self.thick(v, k, spacing, on, axis, mm)
        return self.encoder.patch_embed(v, w)

    def read(self, vols, ws, fg, xyz, vol, corner, edge, masked, gen):
        """Views [V] -> patch indices, valid, foreground, mask [V,L], bi tokens [V,L,2d]. Masking applies only to foreground patches of views with `masked`; the
        backward pass reads each view's valid tokens in reverse (padding stays at the end, zero in the outputs).
        Tokens are embedded per scan (its own voxels per patch and kernel weights `ws`)."""
        idx, valid = self.serialize(xyz[vol], corner, edge, gen)
        fgv = valid & fg[vol[:, None], idx]
        n, nv, dev = valid.sum(1), len(idx), idx.device
        u = lambda lo, hi: self._u((nv, 1, 1), lo, hi, gen)
        gamma, scale, shift, noise = u(-self.gamma, self.gamma).exp(), u(1 - self.scale, 1 + self.scale), \
            u(-self.shift, self.shift), u(0, self.noise)
        on = torch.rand(nv, device=dev, generator=gen) < self.thick_prob
        axis = torch.randint(3, (nv,), device=dev, generator=gen)
        mm = torch.tensor(self.thick_mm, device=dev)[torch.randint(len(self.thick_mm), (nv,), device=dev, generator=gen)]
        # masked in aligned cubes of one of mask_cells per view (<= half its shortest edge): big cubes are the
        # coarse levels' prediction targets
        cells = self.mask_cells[None]
        ok_c = cells <= (edge.amin(-1, keepdim=True) // 2).clamp_min(1)
        c = cells[0][torch.rand(ok_c.shape, device=dev, generator=gen).masked_fill(~ok_c, -1).argmax(1)]
        cid = self.cell_ids(xyz[vol[:, None], idx], vols["grid"][vol], c)
        draw = torch.rand(nv, xyz.shape[1], device=dev, generator=gen).gather(1, cid)
        mask = masked[:, None] & fgv & (draw < u(*self.mask_ratio)[..., 0])
        aug = (gamma.view(-1), scale.view(-1), shift.view(-1), noise.view(-1), on, axis, mm)
        seed = int(torch.randint(1 << 62, (1,), device=dev, generator=gen))
        t = torch.zeros(*idx.shape, self.encoder.dim, device=dev)
        rows_v, pos = valid.nonzero(as_tuple=True)                                          # every token of every view
        vol_r = vol[rows_v]
        for b in vol.unique().tolist():                    # one scan at a time: its voxels per patch and kernel weights
            r = (vol_r == b).nonzero()[:, 0]
            x, k, sp = vols["patches"][b], vols["k_list"][b], vols["spacing_list"][b]
            chunk = max(1, self.embed_chunk_voxels // x.shape[1])
            for c in range(0, len(r), chunk):              # recomputed in backward: memory ~ tokens, not voxels
                rc = r[c:c + chunk]
                t[rows_v[rc], pos[rc]] = checkpoint(self._embed, x, idx[rows_v[rc], pos[rc]], rows_v[rc], ws[b], aug, k,
                                                    sp, seed + b * 1_000_003 + c, use_reentrant=False).to(t.dtype)
        t = self.encoder.apply_mask(t, mask)
        j = torch.arange(idx.shape[1], device=dev)
        rev = torch.where(j < n[:, None], (n[:, None] - 1 - j).clamp_min(0), j)            # an involution
        seqs, ok = torch.cat([t, t.gather(1, rev[..., None].expand_as(t))]), torch.cat([valid, valid])
        lens, groups, cur, tot = ok.sum(1).tolist(), [], [], 0
        for r, ln in enumerate(lens):      # packed calls of <= max_packed_tokens: GDN-2's backward peak follows the call
            if cur and tot + ln > self.max_packed_tokens:
                groups.append(cur)
                cur, tot = [], 0
            cur.append(r)
            tot += ln
        groups.append(cur)
        ckpt = self.encoder.grad_checkpoint and sum(lens) > self.checkpoint_tokens   # recompute only for big scans
        outs = [(g, self.encoder.packed(seqs[g, : max(lens[r] for r in g)], ok[g, : max(lens[r] for r in g)], ckpt))
                for g in map(torch.tensor, groups)]
        h = seqs.new_zeros(*seqs.shape[:2], outs[0][1].shape[-1], dtype=outs[0][1].dtype)
        for g, o in outs:
            h[g, : o.shape[1]] = o
        fwd, bwd = h[: len(t)], h[len(t):].gather(1, rev[..., None].expand(-1, -1, h.shape[-1]))
        return idx, valid, fgv, mask, torch.cat([fwd, bwd], -1)

    def forward(self, vols: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """vols (`dataset.to_device`): patches = per scan [N_b, P_b] native voxel values, spacing [B,3] mm,
        k [B,3] voxels per patch axis, grid [B,3] patch grid."""
        grid, gen = vols["grid"], generator
        b, n_patch = len(grid), int(grid.prod(1).max())
        xyz = grid_coords(grid, n_patch)                                                  # [B,N,3]
        fg = per_patch(vols, n_patch, lambda x: x.mean(-1)) > self.fg_threshold            # [B,N]
        vols = {**vols, "k_list": vols["k"].tolist(), "spacing_list": vols["spacing"].tolist()}
        memo = {}                                          # one kernel weight matrix per distinct (spacing, k)
        for sp, k in zip(vols["spacing_list"], vols["k_list"]):
            if (key := (*sp, *k)) not in memo:
                memo[key] = self.encoder.patch_embed.weights(torch.tensor(sp, device=grid.device), k)
        ws = [memo[(*sp, *k)] for sp, k in zip(vols["spacing_list"], vols["k_list"])]
        corner, edge = self.jittered(*self.boxes(fg, xyz, grid, gen), grid, gen)           # [B,G,2,3]
        vol = torch.arange(b, device=grid.device)[:, None, None].expand(-1, corner.shape[1], 2)
        zg, pred, tgt, ztok, tvol, tfg, lens = [], [], [], [], [], [], []
        cpred, ctgt, cz, cvol = ([[] for _ in self.levels] for _ in range(4))
        for part in (slice(0, self.gg), slice(self.gg, None)):                             # globals / locals
            c, e, v = (t[:, part].reshape(-1, *t.shape[3:]) for t in (corner, edge, vol))  # views A,B interleaved
            masked = torch.arange(len(c), device=c.device) % 2 == 0
            idx, valid, fgv, mask, bi = self.read(vols, ws, fg, xyz, v, c, e, masked, gen)
            w = fgv | (valid & ~fgv.any(1, keepdim=True))                              # foreground (all if none)
            top = torch.bmm(w[:, None].to(bi.dtype), bi)[:, 0].float() / w.sum(1, keepdim=True)
            zg.append(self.glob(top).float().view(b, -1, 2, self.glob[-1].out_features))
            a, bb = masked.nonzero()[:, 0], (~masked).nonzero()[:, 0]
            pxyz, gv = xyz[v[:, None], idx], grid[v]
            for i, (cs, head) in enumerate(zip(self.levels, self.cell)):
                p, q, zb, vb = self.cells(bi, valid & fgv, mask, pxyz, gv, cs, n_patch, head)
                cpred[i].append(p)
                ctgt[i].append(q)
                cz[i].append(zb)
                cvol[i].append(v[1::2][vb])
            safe = torch.where(valid, idx, n_patch)                                        # padding -> dump column
            pos_b = torch.full((len(bb), n_patch + 1), -1, device=c.device)
            pos_b.scatter_(1, safe[bb], torch.arange(idx.shape[1], device=c.device).expand(len(bb), -1))
            pos_b[:, -1] = -1
            j_b = pos_b.gather(1, safe[a])                                                 # A's patch -> its position in B
            pair = mask[a] & (j_b >= 0)                                                    # mask is foreground only
            # token head recomputed in backward for big scans: whole-volume views give ~10^5 tokens x 1024 hidden
            # (deterministic: batch-statistics BatchNorm)
            head = (lambda x: checkpoint(self.tok, x, use_reentrant=False)) if len(bi) * bi.shape[1] > self.checkpoint_tokens \
                else self.tok
            z_b = head(bi[bb][valid[bb]]).float()                                          # B's valid tokens, flat
            flat = valid[bb].flatten().long().cumsum(0).view_as(valid[bb]) - 1             # (row, pos) -> flat index
            pred.append(head(bi[a][pair]).float())
            tgt.append(z_b[flat.gather(1, j_b.clamp_min(0))[pair]])
            ztok.append(z_b)
            tvol.append(v[bb][:, None].expand_as(valid[bb])[valid[bb]])
            tfg.append(fgv[bb][valid[bb]])
            lens.append(e.prod(-1).float().mean())
        zg = torch.cat(zg, 1)                                                              # [B,G,2,D]
        inv_g = (zg - zg.mean(2, keepdim=True)).square().mean()
        # symmetric, as LeJEPA: both tokens pulled to their mean, no stop-grad (with a stop-grad target and
        # no EMA teacher the target drifted and inv_token grew 0.19 -> 2.7); SIGReg prevents collapse
        inv_t = (torch.cat(pred) - torch.cat(tgt)).square().mean() / 4                     # = mean ||z - mean||^2
        ztok, tvol, tfg = torch.cat(ztok), torch.cat(tvol), torch.cat(tfg)
        # equal foreground tokens per volume (with replacement: fixed shape for the cross-GPU gather); a
        # volume whose B views hold no foreground (boxes are centred on it, so only after jitter) falls back to all
        keep = self._sample(tvol, tfg, b, gen)
        dist = torch.distributed.is_initialized()   # SIGReg sees every rank's samples (differentiable gather)
        sig = lambda z: self.sigreg(gather_rows(z) if dist else z, gen)
        sig_g, sig_t = sig(zg.permute(2, 0, 1, 3).reshape(2, -1, zg.shape[-1])), sig(ztok[keep][None])
        out = {}
        for cs, p, q, z, vb in zip(self.levels, cpred, ctgt, cz, cvol):
            p, q, z, vb = torch.cat(p), torch.cat(q), torch.cat(z), torch.cat(vb)
            out[f"inv_cell{cs}"] = (p - q).square().mean(-1).sum() / max(len(p), 1) / 4
            out[f"sigreg_cell{cs}"] = sig(z[self._sample(vb, torch.ones_like(vb, dtype=torch.bool), b, gen)][None])
            out[f"pairs_cell{cs}"] = torch.tensor(float(len(p)), device=z.device)
        with torch.no_grad():                                   # between-volume share of token-embedding variance
            zt, tv = ztok[keep], tvol[keep]
            means = torch.zeros(b, zt.shape[1], device=zt.device).index_add_(0, tv, zt)
            means /= torch.bincount(tv, minlength=b).clamp_min(1)[:, None]
            vol_share = 1 - (zt - means[tv]).var(0).sum() / zt.var(0).sum()
        inv = [inv_g, inv_t, *(out[f"inv_cell{c}"] for c in self.levels)]
        sigs = [sig_g, sig_t, *(out[f"sigreg_cell{c}"] for c in self.levels)]
        return {"loss": (1 - self.lam) * sum(inv) / len(inv) + self.lam * sum(sigs) / len(sigs),
                "inv_global": inv_g, "inv_token": inv_t, "sigreg_global": sig_g, "sigreg_token": sig_t, **out,
                "vol_share": vol_share, "len_global": lens[0], "len_local": lens[1]}

    def _sample(self, vol: torch.Tensor, fg: torch.Tensor, b: int, gen) -> torch.Tensor:
        """`tokens_per_volume` row indices per volume (foreground first; with replacement: a fixed shape for
        the cross-GPU gather); a volume with no foreground row falls back to all its rows, with none to any."""
        keep = []
        for v in range(b):
            i = ((vol == v) & fg).nonzero()[:, 0]
            i = i if len(i) else (vol == v).nonzero()[:, 0]
            i = i if len(i) else torch.arange(len(vol), device=vol.device)
            keep.append(i[torch.randint(len(i), (self.tokens_per_volume,), device=i.device, generator=gen)])
        return torch.cat(keep)

    def cells(self, bi, sel, mask, pxyz, grid, c, n_patch, head):
        """Views A (even rows) and B (odd), bi tokens [V,L,2d], `sel` = their foreground [V,L] -> per cube of c
        patches per edge: projected mean token of A at cubes whose foreground is entirely masked in A, of B at the
        same cubes, of B at every cube it holds, and the pair index of the latter."""
        cid = self.cell_ids(pxyz, grid, torch.full((len(bi),), c, device=bi.device))
        pair = torch.arange(len(bi), device=bi.device)[:, None].expand_as(cid) // 2 * n_patch + cid
        sa, sb = sel[0::2], sel[1::2]
        u, inv = torch.unique(torch.cat([pair[0::2][sa], pair[1::2][sb]]), return_inverse=True)
        ia, ib = inv[: int(sa.sum())], inv[int(sa.sum()):]
        n_a, n_b = torch.bincount(ia, minlength=len(u)), torch.bincount(ib, minlength=len(u))
        mean = lambda x, i, n: x.new_zeros(len(u), x.shape[-1]).index_add_(0, i, x) / n.clamp_min(1)[:, None]
        m_a, m_b = mean(bi[0::2][sa].float(), ia, n_a), mean(bi[1::2][sb].float(), ib, n_b)
        full = (torch.bincount(ia, weights=mask[0::2][sa].float(), minlength=len(u)) == n_a) & (n_a > 0)
        ok, has_b = full & (n_b > 0), n_b > 0
        z = head(torch.cat([m_a[ok], m_b[has_b]])).float()
        z_a, z_b = z[: int(ok.sum())], z[int(ok.sum()):]
        return z_a, z_b[(has_b.cumsum(0) - 1)[ok]], z_b, u[has_b] // n_patch
