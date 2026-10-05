"""LeJEPA over groups of patches seen through different serializations.

Per volume, N contiguous boxes of the patch grid ("groups": a few global, most local). Each group
yields K views; a view is its box jittered (partial overlap), its voxels intensity-augmented, a
fraction of its patches replaced by the mask token, and the rest read in the order of a curve
drawn at random from `view_curves` (any of its 48 cube symmetries). The view embedding is the mean
encoder output over its tokens, then a projector:

    loss = (1 - lam) * invariance(views of a group) + lam * SIGReg(per-volume-centred embeddings)

Views must differ in content, not only in order: with no position embedding, an encoder that
ignores order (a bag of patches) would be invariant to serialization alone for free.
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
        """z: [..., M, D] -> scalar (mean over leading dims and slices)."""
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


def per_volume_centred(z: torch.Tensor, vol: torch.Tensor, n_vol: int) -> torch.Tensor:
    """z [K, M, D], vol [M] -> z minus the mean of its volume's samples over all K views."""
    per_vol = torch.zeros(n_vol, z.shape[-1], device=z.device, dtype=z.dtype).index_add_(0, vol, z.sum(0))
    return z - (per_vol / (z.shape[0] * torch.bincount(vol, minlength=n_vol).clamp_min(1))[:, None])[vol]


class LeJEPA(nn.Module):
    def __init__(self, encoder: Encoder, view_curves: list[str], seed: int = 17, groups_global: int = 2,
                 groups_local: int = 6, views: int = 2, global_frac: tuple = (0.3, 1.0), local_edge: tuple = (3, 8),
                 jitter: float = 0.25, mask_ratio: tuple = (0.1, 0.3), gamma: float = 0.3, scale: float = 0.1,
                 shift: float = 0.05, noise: float = 0.02, fg_threshold: float = 0.05, proj_hidden: int = 1024,
                 proj_dim: int = 128, lam: float = 0.05, sigreg_slices: int = 256, sigreg_knots: int = 17):
        super().__init__()
        self.encoder, self.gg, self.gl, self.k, self.lam = encoder, groups_global, groups_local, views, lam
        self.global_frac, self.local_edge, self.jitter, self.mask_ratio = global_frac, local_edge, jitter, mask_ratio
        self.gamma, self.scale, self.shift, self.noise, self.fg_threshold = gamma, scale, shift, noise, fg_threshold
        self.projector = Projector(encoder.dim, proj_hidden, proj_dim)
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
        # pad to a power of two: GDN-2's Triton kernels re-tune (up to ~20 s) for every new length
        order = order[:, : min(1 << (int(n.max()) - 1).bit_length(), order.shape[1])]
        return order, torch.arange(order.shape[1], device=order.device) < n[:, None]

    def embed(self, patches, vol, corner, edge, gen):
        """One batch of views (vol [V] = source volume) -> projected embeddings [V,D]."""
        idx, valid = self.serialize(corner, edge, gen)
        v = patches[vol[:, None], idx]                                                     # [V,L,P]
        u = lambda lo, hi: self._u((len(v), 1, 1), lo, hi, gen)
        v = v.clamp_min(0) ** u(-self.gamma, self.gamma).exp() * u(1 - self.scale, 1 + self.scale)
        v = (v + u(-self.shift, self.shift) + torch.randn(v.shape, device=v.device, generator=gen)
             * u(0, self.noise)).clamp(0, 1)
        mask = valid & (torch.rand(valid.shape, device=v.device, generator=gen) < u(*self.mask_ratio)[..., 0])
        h = self.encoder.norm(self.encoder.run(self.encoder.tokens(v, mask)))
        w = valid[..., None].to(h.dtype)
        return self.projector((h * w).sum(1) / w.sum(1))

    def forward(self, patches: torch.Tensor, generator: torch.Generator) -> dict[str, torch.Tensor]:
        b, k = len(patches), self.k
        corner, edge = self.jittered(*self.boxes(patches.mean(-1) > self.fg_threshold, generator), generator)
        vol = torch.arange(b, device=patches.device)[:, None, None].expand(-1, corner.shape[1], k)
        z, lens = [], []
        for part in (slice(0, self.gg), slice(self.gg, None)):                             # globals / locals
            c, e, v = (t[:, part].reshape(-1, *t.shape[3:]) for t in (corner, edge, vol))
            z.append(self.embed(patches, v, c, e, generator).float().view(b, -1, k, self.projector[-1].out_features))
            lens.append(e.prod(-1).float().mean())
        z = torch.cat(z, 1)                                                                # [B,G,K,D]
        inv = (z - z.mean(2, keepdim=True)).square().mean()
        zk = z.permute(2, 0, 1, 3).reshape(k, -1, z.shape[-1])                             # [K, B*G, D]
        zc = per_volume_centred(zk, torch.arange(b, device=z.device).repeat_interleave(z.shape[1]), b)
        if torch.distributed.is_initialized():  # SIGReg sees every rank's samples (differentiable gather)
            zc = torch.cat(torch.distributed.nn.functional.all_gather(zc), 1)
        sig = self.sigreg(zc, generator)
        return {"loss": (1 - self.lam) * inv + self.lam * sig, "inv": inv, "sigreg": sig,
                "len_global": lens[0], "len_local": lens[1]}
