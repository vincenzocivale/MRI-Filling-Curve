"""LeJEPA over groups of patches seen through different serializations, with a global and a
per-token term (the DINOv2 pattern: a global loss alone lets token features drift into a per-volume
code and dense quality decay).

Per volume, boxes of the patch grid ("groups": a few global, most local). Each group yields two
views read along different curves (random curve of `view_curves` x cube symmetry), each with its own
box jitter and intensity augmentation; view A also has `mask_ratio` of its patches replaced by the
mask token. Each view is read forwards and backwards by the same causal encoder.

- global: the last forward state (a summary of the whole box) -> `glob` head; invariance between
  A and B + SIGReg (per view, over the batch).
- token: for every masked patch of A that B also contains, the `bi` token (forward ++ backward
  output at that patch, as in the segmentation probe) of A and of B -> `tok` head, pulled together
  (symmetric, no stop-grad, as LeJEPA's invariance);
  SIGReg on B's token embeddings: `tokens_per_volume` per volume, so the sample count stays in
  LeJEPA's calibrated range (scaled by ~20k tokens it gave per-patch noise; by #volumes it let a
  per-volume code through).
loss = (1 - lam) * (inv_global + inv_token) / 2 + lam * (sig_global + sig_token) / 2
"""
from __future__ import annotations

import torch
import torch.distributed.nn
from torch import nn

from .curves import CurveViews, grid_coords
from .model import Encoder


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
                 jitter: float = 0.25, mask_ratio: tuple = (0.2, 0.5), gamma: float = 0.3, scale: float = 0.1,
                 shift: float = 0.05, noise: float = 0.02, fg_threshold: float = 0.05, proj_hidden: int = 1024,
                 proj_dim: int = 128, tokens_per_volume: int = 64, lam: float = 0.05, sigreg_slices: int = 256,
                 sigreg_knots: int = 17):
        super().__init__()
        self.encoder, self.gg, self.gl, self.k, self.lam = encoder, groups_global, groups_local, 2, lam
        self.tokens_per_volume = tokens_per_volume
        self.global_frac, self.local_edge, self.jitter, self.mask_ratio = global_frac, local_edge, jitter, mask_ratio
        self.gamma, self.scale, self.shift, self.noise, self.fg_threshold = gamma, scale, shift, noise, fg_threshold
        self.glob = Projector(encoder.dim, proj_hidden, proj_dim)
        self.tok = Projector(2 * encoder.dim, proj_hidden, proj_dim)
        self.sigreg = SIGReg(sigreg_slices, sigreg_knots)
        g = encoder.grid
        ranks = torch.cat([CurveViews(c, g, seed).ranks for c in view_curves])   # every curve x symmetry
        self.register_buffer("ranks", ranks, persistent=False)
        self.register_buffer("xyz", torch.from_numpy(grid_coords(g)).long(), persistent=False)

    @classmethod
    def from_config(cls, cfg: dict) -> LeJEPA:
        return cls(Encoder.from_config(cfg), seed=int(cfg["seed"]), **cfg["objective"])

    def _u(self, shape, lo, hi, gen):
        return torch.rand(shape, device=self.xyz.device, generator=gen) * (hi - lo) + lo

    def boxes(self, fg: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """fg [B,N] -> group boxes (corner, edge) [B,G,3], globals first. Centres on foreground
        patches; global edges span global_frac of the foreground bounding box (volume fraction),
        local edges are log-uniform in local_edge per axis."""
        b, g = len(fg), self.encoder.grid
        xyz = self.xyz.expand(b, -1, -1)
        big = torch.where(fg[..., None], xyz, g).amin(1), torch.where(fg[..., None], xyz, -1).amax(1)
        extent = (big[1] - big[0] + 1).clamp_min(1)                                       # [B,3]
        frac = self._u((b, self.gg, 1), *self.global_frac, gen) ** (1 / 3)
        e_glob = (extent[:, None] * frac).round()
        lo, hi = torch.tensor(self.local_edge, dtype=torch.float).log()
        # ponytail: independent per-axis edges (aspect in [3/8, 8/3]); add an explicit aspect prior if needed
        e_loc = self._u((b, self.gl, 3), lo, hi, gen).exp().round()
        edge = torch.cat([e_glob, e_loc], 1).clamp(1, g).long()                            # [B,G,3]
        centre = xyz.gather(1, torch.multinomial(fg.float() + 1e-6, self.gg + self.gl, replacement=True,
                                                 generator=gen)[..., None].expand(-1, -1, 3))
        return (centre - edge // 2).clamp(min=0).minimum(g - edge), edge

    def jittered(self, corner, edge, gen):
        """K views per group: each box rescaled and shifted by up to +-jitter of its edge -> [B,G,K,3]."""
        g, shape = self.encoder.grid, (*edge.shape[:2], self.k, 3)
        e = (edge[:, :, None] * self._u(shape, -self.jitter, self.jitter, gen).exp()).round().clamp(1, g).long()
        c = corner[:, :, None] + (self._u(shape, -self.jitter, self.jitter, gen) * edge[:, :, None]).round().long()
        return c.clamp(min=0).minimum(g - e), e

    def serialize(self, corner, edge, gen):
        """Boxes [V,3] -> canonical patch indices [V,L] in a random curve's order, valid [V,L]."""
        inside = ((self.xyz >= corner[:, None]) & (self.xyz < (corner + edge)[:, None])).all(-1)   # [V,N]
        rank = self.ranks[torch.randint(len(self.ranks), (len(corner),), device=corner.device, generator=gen)]
        order = torch.where(inside, rank, rank.shape[1] + rank).argsort(1)
        n = inside.sum(1)
        order = order[:, : int(n.max())]
        return order, torch.arange(order.shape[1], device=order.device) < n[:, None]

    def read(self, patches, vol, corner, edge, masked, gen):
        """Views [V] -> canonical indices, valid, mask [V,L], last forward state [V,d], bi tokens [V,L,2d].
        Masking applies only to views with `masked`; the backward pass reads each view's valid
        tokens in reverse (padding stays at the end, zero in the outputs)."""
        idx, valid = self.serialize(corner, edge, gen)
        n = valid.sum(1)
        v = patches[vol[:, None], idx]                                                     # [V,L,P]
        u = lambda lo, hi: self._u((len(v), 1, 1), lo, hi, gen)
        v = v.clamp_min(0) ** u(-self.gamma, self.gamma).exp() * u(1 - self.scale, 1 + self.scale)
        v = (v + u(-self.shift, self.shift) + torch.randn(v.shape, device=v.device, generator=gen)
             * u(0, self.noise)).clamp(0, 1)
        mask = masked[:, None] & valid & (torch.rand(valid.shape, device=v.device, generator=gen)
                                          < u(*self.mask_ratio)[..., 0])
        t = self.encoder.tokens(v, mask)
        j = torch.arange(idx.shape[1], device=idx.device)
        rev = torch.where(j < n[:, None], (n[:, None] - 1 - j).clamp_min(0), j)            # an involution
        x, keep = torch.cat([t, t.gather(1, rev[..., None].expand_as(t))]), torch.cat([valid, valid])
        # the encoder sees only valid tokens, packed (views differ ~10x in length: padding was ~3x the work).
        # FLA's short conv re-tunes (~20 s) for every new ceil(T/1024): a dummy last sequence rounds T
        # up to 8 lengths per octave (<= 12.5% extra)
        x, total = x[keep], int(keep.sum())
        pad = -total % (1 << max(total.bit_length() - 3, 10))
        cu = nn.functional.pad(keep.sum(1).cumsum(0), (1, 0))
        cu = torch.cat([cu, cu[-1:] + pad] if pad else [cu]).int()
        out = self.encoder.norm(self.encoder.run(nn.functional.pad(x, (0, 0, 0, pad))[None], cu)[0, :total])
        h = out.new_zeros(*keep.shape, out.shape[-1]).index_put((keep,), out)
        fwd, bwd = h[: len(t)], h[len(t):].gather(1, rev[..., None].expand(-1, -1, h.shape[-1]))
        return idx, valid, mask, fwd[torch.arange(len(t)), n - 1], torch.cat([fwd, bwd], -1)

    def forward(self, patches: torch.Tensor, generator: torch.Generator) -> dict[str, torch.Tensor]:
        b, n_patch, gen = len(patches), patches.shape[1], generator
        corner, edge = self.jittered(*self.boxes(patches.mean(-1) > self.fg_threshold, gen), gen)   # [B,G,2,3]
        vol = torch.arange(b, device=patches.device)[:, None, None].expand(-1, corner.shape[1], 2)
        zg, pred, tgt, ztok, tvol, lens = [], [], [], [], [], []
        for part in (slice(0, self.gg), slice(self.gg, None)):                             # globals / locals
            c, e, v = (t[:, part].reshape(-1, *t.shape[3:]) for t in (corner, edge, vol))  # views A,B interleaved
            masked = torch.arange(len(c), device=c.device) % 2 == 0
            idx, valid, mask, last, bi = self.read(patches, v, c, e, masked, gen)
            zg.append(self.glob(last).float().view(b, -1, 2, self.glob[-1].out_features))
            a, bb = masked.nonzero()[:, 0], (~masked).nonzero()[:, 0]
            safe = torch.where(valid, idx, n_patch)                                        # padding -> dump column
            pos_b = torch.full((len(bb), n_patch + 1), -1, device=c.device)
            pos_b.scatter_(1, safe[bb], torch.arange(idx.shape[1], device=c.device).expand(len(bb), -1))
            pos_b[:, -1] = -1
            j_b = pos_b.gather(1, safe[a])                                                 # A's patch -> its position in B
            pair = mask[a] & (j_b >= 0)
            z_b = self.tok(bi[bb][valid[bb]]).float()                                      # B's valid tokens, flat
            flat = valid[bb].flatten().long().cumsum(0).view_as(valid[bb]) - 1             # (row, pos) -> flat index
            pred.append(self.tok(bi[a][pair]).float())
            tgt.append(z_b[flat.gather(1, j_b.clamp_min(0))[pair]])
            ztok.append(z_b)
            tvol.append(v[bb][:, None].expand_as(valid[bb])[valid[bb]])
            lens.append(e.prod(-1).float().mean())
        zg = torch.cat(zg, 1)                                                              # [B,G,2,D]
        inv_g = (zg - zg.mean(2, keepdim=True)).square().mean()
        # symmetric, as LeJEPA: both tokens pulled to their mean, no stop-grad (with a stop-grad target and
        # no EMA teacher the target drifted and inv_token grew 0.19 -> 2.7); SIGReg prevents collapse
        inv_t = (torch.cat(pred) - torch.cat(tgt)).square().mean() / 4                     # = mean ||z - mean||^2
        ztok, tvol = torch.cat(ztok), torch.cat(tvol)
        # equal tokens per volume (with replacement: fixed shape for the cross-GPU gather)
        keep = torch.cat([(i := (tvol == v).nonzero()[:, 0])[torch.randint(len(i), (self.tokens_per_volume,),
                          device=i.device, generator=gen)] for v in range(b)])
        sg_in, st_in = zg.permute(2, 0, 1, 3).reshape(2, -1, zg.shape[-1]), ztok[keep][None]
        if torch.distributed.is_initialized():  # SIGReg sees every rank's samples (differentiable gather)
            sg_in, st_in = (torch.cat(torch.distributed.nn.functional.all_gather(x), 1) for x in (sg_in, st_in))
        sig_g, sig_t = self.sigreg(sg_in, gen), self.sigreg(st_in, gen)
        with torch.no_grad():                                   # between-volume share of token-embedding variance
            zt, tv = ztok[keep], tvol[keep]
            means = torch.zeros(b, zt.shape[1], device=zt.device).index_add_(0, tv, zt)
            means /= torch.bincount(tv, minlength=b).clamp_min(1)[:, None]
            vol_share = 1 - (zt - means[tv]).var(0).sum() / zt.var(0).sum()
        return {"loss": (1 - self.lam) * (inv_g + inv_t) / 2 + self.lam * (sig_g + sig_t) / 2,
                "inv_global": inv_g, "inv_token": inv_t, "sigreg_global": sig_g, "sigreg_token": sig_t,
                "vol_share": vol_share, "len_global": lens[0], "len_local": lens[1]}
