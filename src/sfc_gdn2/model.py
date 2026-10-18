from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .curves import grid_coords, keys


def _gdn2():
    try:
        from lit_gpt.gdn2 import GatedDeltaNet2
    except Exception as e:
        raise RuntimeError("Official NVlabs/GatedDeltaNet-2 (and its FLA/Triton deps) is required; "
                           "see README 'Install'.") from e
    _fla_host_fixes()
    return GatedDeltaNet2


_FLA_FIXED = []


def _fla_host_fixes() -> None:
    """Host-side costs of FLA 0.4.2 on packed sequences of every length (profile of 11/10, notes/LOG.md):
    - `prepare_chunk_indices` remembers only its last call (`tensor_cache`), and the short conv and the GDN-2 kernels
      call it in turn with different arguments: recomputed several times per block, each time with a host sync and
      one Python `arange` per sequence (~1.6k syncs and ~150k aranges per step). Here: kept per (cu_seqlens, chunk
      size) while that tensor lives, built by a few kernels and one sync. Same values.
    - the short conv's, gated norm's and l2norm's kernels take NB = cdiv(T, ...), a constexpr their bodies never read, in
      the autotune key too: a compile and a 16-config benchmark for every new packed length. Pinned to 1 by the
      heuristics in front of each autotuner: compiled and tuned once per width."""
    if _FLA_FIXED:
        return
    import sys
    import weakref

    from fla.modules import fused_norm_gate, l2norm
    from fla.modules.conv.triton import kernels
    from fla.ops.utils import index
    from triton.runtime.autotuner import Autotuner, Heuristics

    old, cache = index.prepare_chunk_indices, {}

    def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int, cu_seqlens_cpu=None) -> torch.Tensor:
        key = (id(cu_seqlens), chunk_size)
        if (hit := cache.get(key)) is not None and hit[0]() is cu_seqlens:
            return hit[1]
        n = (cu_seqlens[1:] - cu_seqlens[:-1] + chunk_size - 1) // chunk_size    # chunks per sequence (all >= 1)
        tot, dev = int(n.sum()), n.device
        seq = torch.repeat_interleave(torch.arange(len(n), device=dev), n, output_size=tot)
        out = torch.stack([seq, torch.arange(tot, device=dev) - (n.cumsum(0) - n)[seq]], 1).to(cu_seqlens)
        if len(cache) > 4096:
            cache.clear()
        cache[key] = (weakref.ref(cu_seqlens), out)
        return out

    for m in list(sys.modules.values()):
        if getattr(m, "prepare_chunk_indices", None) is old:
            m.prepare_chunk_indices = prepare_chunk_indices
    for mod in (kernels, fused_norm_gate, l2norm):
        for name, h in list(vars(mod).items()):
            if isinstance(h, Autotuner) and "NB" in h.keys:            # l2norm's: no heuristics in front, add one
                h = Heuristics(h, h.arg_names, {})
                setattr(mod, name, h)
            if isinstance(h, Heuristics) and isinstance(h.fn, Autotuner) and "NB" in h.fn.keys:
                h.values["NB"] = lambda _: 1
    _FLA_FIXED.append(True)


class GDN2Block(nn.Module):
    """Pre-norm causal GDN-2 token mixer + MLP."""

    def __init__(self, d: int, head_dim: int, num_heads: int, use_short_conv: bool, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mix = _gdn2()(hidden_size=d, head_dim=head_dim, num_heads=num_heads, mode="chunk",
                           use_short_conv=use_short_conv, layer_idx=layer_idx)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None, rev: torch.Tensor | None = None,
                both: bool = False) -> torch.Tensor:
        """rev (index reversing each packed sequence): the mixer reads backwards; both: the mean of both ways."""
        y = self.norm(x)
        if torch.is_autocast_enabled(y.device.type):    # cast once, not once per projection (~6 per block)
            y = y.to(torch.get_autocast_dtype(y.device.type))
        m = self.mix(y, cu_seqlens=cu_seqlens)[0] if rev is None or both else 0
        if rev is not None:                             # index_select: its backward is an index_add, no sort
            m = (m + self.mix(y.index_select(1, rev), cu_seqlens=cu_seqlens)[0].index_select(1, rev)) / (2 if both else 1)
        x = x + m
        return x + self.ffn(self.ffn_norm(x))


class AttnBlock(nn.Module):
    """Pre-norm full (non-causal) softmax attention over each packed sequence + MLP, the GDN2Block interface
    (17/10 screens). Output-gated like GDN-2's mixer (sigmoid(W_g y) * attention: 5 d^2 against GDN-2's ~6.4 d^2).
    rope: rotary position on the index along the curve (a pure attention stack has no other order); without it the
    block is position-free and takes its order from the GDN-2 blocks around it (hybrid). `rev`/`both` do not apply."""

    def __init__(self, d: int, head_dim: int, num_heads: int, rope: bool = False):
        super().__init__()
        self.h, self.hd, self.rope = num_heads, head_dim, rope
        self.norm, self.ffn_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.g, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d), nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None, rev=None, both=False) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        y = self.norm(x[0])                                                               # [T, d] (x: [1, T, d])
        if torch.is_autocast_enabled(y.device.type):
            y = y.to(torch.get_autocast_dtype(y.device.type))
        T = len(y)
        cu = cu_seqlens if cu_seqlens is not None else torch.tensor([0, T], device=y.device, dtype=torch.int32)
        q, k, v = self.qkv(y).view(T, 3, self.h, self.hd).unbind(1)
        if self.rope:  # position along the curve within each sequence
            seq = torch.searchsorted(cu[1:].long(), torch.arange(T, device=y.device), right=True)
            pos = (torch.arange(T, device=y.device) - cu.long()[seq]).float()
            f = pos[:, None] * 10000 ** (-torch.arange(0, self.hd, 2, device=y.device) / self.hd)
            cos, sin = f.cos()[:, None].to(q.dtype), f.sin()[:, None].to(q.dtype)

            def rot(t):
                a, b = t[..., ::2], t[..., 1::2]
                return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)
            q, k = rot(q), rot(k)
        a = flash_attn_varlen_func(q, k, v, cu, cu, int((cu[1:] - cu[:-1]).max()), int((cu[1:] - cu[:-1]).max()))
        x = x + self.o(torch.sigmoid(self.g(y)) * a.reshape(T, -1))[None]
        return x + self.ffn(self.ffn_norm(x))


def long_memory(mix: nn.Module, channels: int, half_life: tuple[float, float]) -> None:
    """Slow decay for the first `channels` key channels of every head of a GDN-2 mixer (notes/loss_stato.md, 14/10).
    The decay of key channel i (row i of each head's state S) is exp(-A_h softplus(f_i(x) + dt_bias_i)); at GDN-2's init
    (A_h in [1, 16], softplus(dt_bias) log-uniform in [1e-3, 0.1]) its half-life ln 2 / (A_h dt_i) has median ~10
    tokens, and training does not move it (p4_d1024: 10 tokens at step 760, none above 1000) against views of 1-2k
    tokens. Here dt_i = ln 2 / (A_h n_i), half-lives n_i log-spaced over `half_life` tokens: those rows sum k u^T over
    the whole view (an order-free second-order summary every token can read). No random draw: the other weights'
    init is unchanged (paired screens)."""
    with torch.no_grad():
        n = torch.logspace(math.log10(half_life[0]), math.log10(half_life[1]), channels)
        dt = math.log(2) / (mix.A_log.float().exp()[:, None] * n)          # [heads, channels]
        mix.dt_bias.view(mix.num_heads, mix.head_k_dim)[:, :channels] = dt + torch.log(-torch.expm1(-dt))  # softplus^-1


class KernelEmbed(nn.Module):
    """Patch embedding for native voxels of any spacing: a learned continuous kernel K: mm^3 -> R^d,
    integrated over each voxel's physical extent,

        token = b + (1 / patch_mm^3) * sum_voxels v * integral_{voxel} K,

    so every voxel enters with its own weight and nothing is resampled (CKConv / CCNN-style continuous
    kernels: Romero et al. 2022, arXiv 2102.02611; Knigge et al. 2023, arXiv 2301.10540). K is separable
    in a learned basis, K(x, y, z) = G . (phi_x(x) (x) phi_y(y) (x) phi_z(z)), phi_a: mm -> R^rank an MLP on
    Fourier features, so the voxel integrals reduce to 1D integrals per axis (midpoint rule on `quad_mm`
    sub-intervals: exact up to quad_mm, whatever the spacing, 0.15 to 28 mm in the pool).
    `weights` builds one [P, d] matrix per scan (spacing, voxels per patch axis k).
    `freq_decay` a: the Fourier feature at f is scaled by min(1, f0 / f)^a, f0 = one cycle per patch (a = 1: the 1/f
    amplitude of natural images). A fixed gain, not an init, so AdamW's steps cannot re-roughen the curves. With a = 0
    (all features equal, the runs before 14/10) the curves are rough down to 0.25 mm and the tokens of one patch
    change with the voxel size (notes/kernel_trained.py, notes/kernel_init_check.py)."""

    def __init__(self, d: int, patch_mm: float = 16.0, rank: int = 16, freqs: int = 32, hidden: int = 128,
                 max_cycles_per_mm: float = 4.0, quad_mm: float = 0.02, freq_decay: float = 0.0):
        super().__init__()
        self.patch_mm, self.rank, self.quad_mm = patch_mm, rank, quad_mm
        f = torch.logspace(math.log10(1 / (4 * patch_mm)), math.log10(max_cycles_per_mm), freqs)
        self.register_buffer("freqs", 2 * math.pi * f, persistent=False)
        self.register_buffer("gain", ((1 / (patch_mm * f)).clamp(max=1) ** freq_decay).repeat(2), persistent=False)
        self.phi = nn.Sequential(nn.Linear(2 * freqs, hidden), nn.GELU(), nn.Linear(hidden, 3 * rank))
        # init: a constant patch of 1s gives tokens of std ~0.5 at every spacing. The 1000 is a constant in
        # `weights`, not in G: AdamW steps are ~lr in absolute size, so G at std ~16 learned ~250x slower than the
        # other matrices (renamed from G: an old checkpoint fails to load instead of embedding 1000x off)
        self.G_unit = nn.Parameter(torch.randn(d, rank, rank, rank) / rank ** 1.5)
        self.bias = nn.Parameter(torch.zeros(d))

    def axis_integrals(self, spacing: torch.Tensor, k) -> list[torch.Tensor]:
        """Per axis a: [k_a, rank] = integral of phi_a over each voxel (mm), voxel 0 starting at 0 mm."""
        out = []
        for a, (s, n) in enumerate(zip(spacing.tolist(), k)):
            q = max(1, math.ceil(s / self.quad_mm))
            t = (torch.arange(int(n) * q, device=self.freqs.device, dtype=torch.float32) + 0.5) * (s / q)
            ft = t[:, None] * self.freqs
            phi = self.phi(torch.cat([ft.sin(), ft.cos()], -1) * self.gain)[:, a * self.rank:(a + 1) * self.rank]
            out.append(phi.view(int(n), q, self.rank).mean(1) * s)
        return out

    def weights(self, spacing: torch.Tensor, k) -> torch.Tensor:
        """spacing [3] mm, k [3] voxels per patch axis -> W [k0*k1*k2, d] (patchify's voxel order)."""
        with torch.autocast(spacing.device.type, enabled=False):  # bf16 picked an sm75 bmm, ~15% of the step
            px, py, pz = self.axis_integrals(spacing, k)
            w = torch.einsum("dabc,xa,yb,zc->xyzd", self.G_unit, px, py, pz)
        return w.reshape(-1, self.G_unit.shape[0]) * (1000 / self.patch_mm ** 3)

    def forward(self, patches: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """[..., P] native voxel values (fp32: the embedding sees them unrounded) -> [..., d]."""
        with torch.autocast(patches.device.type, enabled=False):
            return patches.float() @ w.float() + self.bias


class ViewPos(nn.Module):
    """View position: a patch's coordinates inside the box its view reads (after the view's cube symmetry; patch
    units), Fourier features (periods 2..256 patches) -> linear, zero-initialised (the encoder starts as without it).
    A and B of a box differ in box (jitter) and symmetry, so a patch has different coordinates in the two views:
    position alone cannot pair them, as an absolute embedding could (the reason the encoder had none)."""

    def __init__(self, d: int, freqs: int = 8):
        super().__init__()
        self.register_buffer("w", 2 * math.pi / torch.logspace(1, 8, freqs, base=2), persistent=False)
        self.proj = nn.Linear(6 * freqs, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """[..., 3] coordinates -> [..., d]."""
        a = c.float()[..., None] * self.w
        return self.proj(torch.cat([a.sin(), a.cos()], -1).flatten(-2))


class Encoder(nn.Module):
    """Causal GDN-2 over curve-ordered patch tokens.

    Tokens are built in raster space (continuous-kernel embedding of each patch's native voxels,
    `KernelEmbed`; masked patches replaced by a learned mask token) and only then gathered into curve order. There is no coordinate embedding: the curve
    order is the only spatial signal, and it differs between views, so an embedding cannot be made
    view-invariant through absolute position. Any grid size: sequences are packed, not padded.
    `view_pos`: adds `ViewPos` (coordinates relative to the view's own box: no absolute position).
    `bidir`: "late" (default) the stack reads each sequence forwards and, separately, backwards, bi = fwd ++ bwd;
    "alternate" every odd block reads backwards (Vision-LSTM, Alkin et al. 2024): every token sees the whole sequence
    from block 2 on, at half the late encoder's cost; "block" every block reads both ways with the same mixer, outputs
    averaged (Vision Mamba / Hydra pattern), the late encoder's mixer cost and half its MLP. One stream: bi = h ++ h
    (same interface for the heads, probes and decoder).
    """

    def __init__(self, d_model: int, depth: int, head_dim: int, num_heads: int, use_short_conv: bool = True,
                 grad_checkpoint: bool = False, patch_mm: float = 16.0, view_pos: bool = False, bidir: str = "late",
                 long_mem: dict | None = None, attn_every: int = 0, attn_rope: bool = False, **embed):
        super().__init__()
        assert bidir in ("late", "alternate", "block"), bidir
        self.dim, self.grad_checkpoint, self.bidir = d_model, grad_checkpoint, bidir
        self.taps: tuple = ()  # block indices whose outputs `flat` also returns (segmentation decoder input, fm/sfc.py)
        self.patch_embed = KernelEmbed(d_model, patch_mm, **embed)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos = ViewPos(d_model) if view_pos else None
        # attn_every k > 0 (17/10 screens): every k-th block is an AttnBlock (1: all of them; attn_rope for those)
        self.blocks = nn.ModuleList(AttnBlock(d_model, head_dim, num_heads, attn_rope) if attn_every and (i + 1) % attn_every == 0
                                    else GDN2Block(d_model, head_dim, num_heads, use_short_conv, i) for i in range(depth))
        if long_mem:
            for b in self.blocks:
                long_memory(b.mix, long_mem["channels"], long_mem["half_life"])
        self.norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, cfg: dict) -> Encoder:
        m = cfg["model"]
        return cls(m["d_model"], m["depth"], m["head_dim"], m["num_heads"], m.get("use_short_conv", True),
                   m.get("grad_checkpoint", False), cfg["data"]["patch_mm"], m.get("view_pos", False),
                   m.get("bidir", "late"), m.get("long_mem"), m.get("attn_every", 0), m.get("attn_rope", False),
                   **m.get("embed", {}))

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """[...,N,d] tokens, masked ones replaced by the learned mask token."""
        return x if mask is None else torch.where(mask[..., None], self.mask_token.to(x.dtype), x)

    def run(self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None, ckpt: bool | None = None,
            rev: torch.Tensor | None = None) -> torch.Tensor:
        """[B,S,d] curve-ordered tokens -> [B,S,d] hidden states (pre final norm). With `cu_seqlens`
        (int32 [n+1] offsets), x is [1,T,d]: n sequences packed back to back, no padding. rev: see `bidir`."""
        ckpt = self.grad_checkpoint if ckpt is None else ckpt
        taps = [x] * self.taps.count(-1)  # -1: the input tokens (patch embedding)
        for i, block in enumerate(self.blocks):
            r = rev if self.bidir == "block" or i % 2 else None
            x = (checkpoint(block, x, cu_seqlens, r, self.bidir == "block", use_reentrant=False) if ckpt and self.training
                 else block(x, cu_seqlens, r, self.bidir == "block"))
            taps += [x] * self.taps.count(i)  # a repeated index: same input width, no new block (control)
        return torch.cat([x, *taps], -1) if taps else x

    def packed(self, x: torch.Tensor, valid: torch.Tensor, ckpt: bool | None = None) -> torch.Tensor:
        """[V,L,d] curve-ordered tokens, [V,L] valid (a prefix of each row) -> [V,L,d] normed hidden
        states, zero at padding. Only valid tokens are run, packed into one sequence (views differ ~10x
        in length: padding was ~3x the work)."""
        cu = nn.functional.pad(valid.sum(1).cumsum(0), (1, 0)).int()
        out = self.flat(x[valid], cu, ckpt)
        return out.new_zeros(*valid.shape, out.shape[-1]).index_put((valid,), out)

    def flat(self, x: torch.Tensor, cu_seqlens: torch.Tensor, ckpt: bool | None = None) -> torch.Tensor:
        """[T,d] curve-ordered tokens of n sequences back to back (int32 offsets cu_seqlens [n+1]) -> [T,d] normed
        hidden states. A dummy sequence pads T <= 64: GDN-2 switches to its recurrent kernel there in eval."""
        total = len(x)
        if pad := max(65 - total, 0):
            x, cu_seqlens = nn.functional.pad(x, (0, 0, 0, pad)), torch.cat([cu_seqlens, cu_seqlens[-1:] + pad])
        rev = None
        if self.bidir != "late":                           # index reversing each sequence in place
            k, cu = torch.arange(len(x), device=x.device), cu_seqlens.long()
            s = torch.searchsorted(cu[1:], k, right=True)
            rev = cu[s] + cu[s + 1] - 1 - k
        h = self.run(x[None], cu_seqlens, ckpt, rev)[0, :total]
        if not self.taps:
            return self.norm(h)
        h = h.split(self.dim, -1)  # the last block's output (final norm) ++ the taps' (plain layer norm)
        return torch.cat([self.norm(h[0]), *(nn.functional.layer_norm(t, t.shape[-1:]) for t in h[1:])], -1)


# scans on the device (`dataset.to_device`) -> {"bi": [B,N,F] (patch level, N = the batch's largest grid),
# "mean": [B,F] (volume level)}
Extractor = Callable[[dict], dict[str, torch.Tensor]]


def encoder_extractor(encoder: Encoder, curves: list[str], level: str, seed: int, fg_threshold: float,
                      crop: bool = False, embed_only: bool = False, reverse: bool = False) -> Extractor:
    """Each volume read along each of `curves` over its whole patch grid (untransformed view), forwards and
    backwards; `bi` = forward ++ backward output per patch, averaged over the curves (test-time augmentation).
    patch: `bi` per patch (raster order). both: `bi` and `mean`, bi averaged over the foreground patches (the top of
    the pretraining pyramid; the benchmark wrapper, fm/sfc.py).
    crop: only the foreground's bounding box is read, as the training's global views (patches outside: bi 0);
    embed_only: no encoder, bi = the patch embedding twice (what the encoder adds to its input). The view position
    (`Encoder.pos`) is the coordinate inside the box read.
    both also gives `std` (per channel over the same patches) and `fg` [B,N], and with several curves each curve's own
    `mean@<curve>`, `std@<curve>` (the probe menu can read one curve instead of their average). reverse (one-stream encoders): each curve
    is also read from its other end (test-time augmentation, 2x the readings): patch -> `bi` over all of them; both ->
    `bi`, `mean`, `std` as without it plus `bi_tta`, `mean_tta`, `std_tta` over all of them."""
    from .data.dataset import per_patch

    def extract(vols):
        grid = vols["grid"]
        xyz = grid_coords(grid, int(grid.prod(1).max()))
        real = xyz[..., 0] >= 0
        fg = real & (per_patch(vols, xyz.shape[1], lambda x: x.mean(-1)) > fg_threshold)
        lo, hi = torch.zeros_like(grid), grid - 1
        if crop:  # the foreground's bounding box (the whole grid when there is no foreground)
            has = fg.any(1)[:, None]
            lo = torch.where(has, torch.where(fg[..., None], xyz, 1 << 30).amin(1), lo)
            hi = torch.where(has, torch.where(fg[..., None], xyz, -1).amax(1), hi)
        inside = real & ((xyz >= lo[:, None]) & (xyz <= hi[:, None])).all(-1)
        local, dims = (xyz - lo[:, None]).clamp_min(0), hi - lo + 1
        n = inside.sum(1)
        j = torch.arange(xyz.shape[1], device=grid.device)
        valid = j < n[:, None]
        rev = torch.where(valid, (n[:, None] - 1 - j).clamp_min(0), j)
        t = torch.zeros(len(grid), xyz.shape[1], encoder.dim, device=grid.device)
        for i, x in enumerate(vols["patches"]):                             # per scan: its own kernel weights
            kw = encoder.patch_embed.weights(vols["spacing"][i], vols["k"][i].tolist())
            t[i, : len(x)] = encoder.patch_embed(x, kw)
        if encoder.pos is not None:
            t = t + encoder.pos(local) * inside[..., None]
        bi = torch.cat([t, t], -1) * inside[..., None] if embed_only else 0
        rv = reverse and not embed_only and encoder.bidir != "late"  # late already reads both ways
        bir = 0
        w = (fg | (inside & ~fg.any(1, keepdim=True))).float()[..., None]

        def stats(b):  # foreground mean and std per channel
            m = (w * b).sum(1) / w.sum(1)
            return m, ((w * (b - m[:, None]) ** 2).sum(1) / w.sum(1)).sqrt()

        per_curve = {}
        for curve in [] if embed_only else curves:
            order = torch.stack([keys(curve, c, d, seed).masked_fill(~m, 1 << 62).argsort()
                                 for c, d, m in zip(local, dims, inside)])  # per volume: no batch-dependent curve
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = t.gather(1, order[..., None].expand(-1, -1, encoder.dim))
                if encoder.bidir == "late":
                    h = encoder.packed(torch.cat([s, s.gather(1, rev[..., None].expand_as(s))]), valid.repeat(2, 1)).float()
                    fb = torch.cat([h[: len(grid)], h[len(grid):].gather(1, rev[..., None].expand(-1, -1, h.shape[-1]))], -1)
                else:                                                          # one stream: bi = h ++ h
                    fb = encoder.packed(s, valid).float().repeat(1, 1, 2)
                    if rv:                                                     # the curve from its other end
                        hr = encoder.packed(s.gather(1, rev[..., None].expand_as(s)), valid).float()
                        fr = hr.gather(1, rev[..., None].expand(-1, -1, hr.shape[-1])).repeat(1, 1, 2)
            back = order.argsort(1)[..., None]                                 # curve position -> raster
            fb = fb.gather(1, back.expand(-1, -1, fb.shape[-1]))
            bi = bi + fb / len(curves)
            if level != "patch" and len(curves) > 1:                           # each curve's own reading (probe menu)
                per_curve |= dict(zip((f"mean@{curve}", f"std@{curve}"), stats(fb)))
            if rv:
                bir = bir + fr.gather(1, back.expand(-1, -1, fr.shape[-1])) / len(curves)
        tta = (bi + bir) / 2 if rv else bi
        if level == "patch":
            return {"bi": tta if reverse else bi}
        (mean, std), out = stats(bi), per_curve
        if reverse:
            out = dict(zip(("bi_tta", "mean_tta", "std_tta"), (tta, *stats(tta))))
        return {"bi": bi, "mean": mean, "std": std, "fg": fg} | out
    return extract
