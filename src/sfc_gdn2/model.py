from __future__ import annotations

import torch
from torch import nn

from .curves import order


def _gdn2():
    try:
        from lit_gpt.gdn2 import GatedDeltaNet2
        return GatedDeltaNet2
    except Exception as e:
        raise RuntimeError(
            "Official GatedDeltaNet-2 is required. Install NVlabs/GatedDeltaNet-2 and its FLA/Triton dependencies."
        ) from e


class GDN2Block(nn.Module):
    def __init__(self, d_model: int, head_dim: int, num_heads: int, use_short_conv: bool, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        GDN2 = _gdn2()
        self.norm = nn.LayerNorm(d_model)
        self.mix = GDN2(hidden_size=d_model, head_dim=head_dim, num_heads=num_heads,
                        mode="chunk", use_short_conv=use_short_conv)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))

    def forward(self, x):
        z = self.norm(x)
        f = self.mix(z)[0]
        if self.bidirectional:
            b = torch.flip(self.mix(torch.flip(z, dims=[1]))[0], dims=[1])
            f = 0.5 * (f + b)
        x = x + f
        return x + self.ffn(self.ffn_norm(x))


class MRIProbe(nn.Module):
    def __init__(self, patch_voxels: int, grid: int, curve: str, seed: int, cfg: dict,
                 objective: str = "masked", predict_k: int = 1):
        super().__init__()
        if objective not in {"masked", "next_patch"}:
            raise ValueError(f"Unknown objective: {objective}")
        if objective == "next_patch":
            if predict_k < 1:
                raise ValueError("predict_k must be >= 1.")
            if grid**3 <= predict_k:
                raise ValueError("next_patch requires more patches than predict_k.")
        self.objective = objective
        self.predict_k = predict_k
        d = cfg["d_model"]
        self.grid = grid
        self.register_buffer("perm", torch.from_numpy(order(curve, grid, seed)), persistent=False)
        self.patch_embed = nn.Linear(patch_voxels, d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.coord = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            GDN2Block(d, cfg["head_dim"], cfg["num_heads"], cfg.get("use_short_conv", True),
                      bidirectional=objective == "masked")
            for _ in range(cfg["depth"])
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch_voxels)
        nn.init.normal_(self.mask_token, std=0.02)

        c = torch.stack(torch.meshgrid(
            torch.linspace(-1,1,grid), torch.linspace(-1,1,grid), torch.linspace(-1,1,grid), indexing="ij"
        ), -1).reshape(-1, 3)
        self.register_buffer("coords", c, persistent=False)

    def forward(self, patches: torch.Tensor, mask3d: torch.Tensor | None = None):
        # patches [B,N,V], mask3d [B,N] in canonical raster coordinates.
        x = self.patch_embed(patches)
        if self.objective == "masked":
            if mask3d is None:
                raise ValueError("masked reconstruction requires mask3d.")
            x = torch.where(mask3d[..., None], self.mask_token.view(1,1,-1), x)
        x = x + self.coord(self.coords)[None]
        x = x[:, self.perm]
        if self.objective == "next_patch":
            x = x[:, :-self.predict_k]
        for block in self.blocks: x = block(x)
        pred = self.head(self.norm(x))
        if self.objective == "next_patch":
            return pred
        inv = torch.argsort(self.perm)
        return pred[:, inv]
