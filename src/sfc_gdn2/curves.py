"""3D space-filling curves as sort keys over integer patch coordinates, and the cube's symmetries.

A curve is a key per patch coordinate; sorting a box's patches by key gives its traversal. Boxes
need not be cubic nor powers of two: Hilbert and Morton keys are those of the enclosing 2^bits cube,
so the traversal of a box is the cube's curve restricted to it (it may jump where the curve leaves
and re-enters the box). A *view* applies one of the 48 symmetries of the cube (axis permutation x
reflection, within the box) before keying, so views of one curve share its locality statistics but
differ in traversal.
"""
from __future__ import annotations

import itertools

import torch

CURVES = ("raster", "snake", "morton", "hilbert", "random")


def grid_coords(grid: torch.Tensor, n_patch: int) -> torch.Tensor:
    """Patch grids [B,3] -> raster coordinates [B,N,3] along a patch axis padded to N (-1 = padding)."""
    j = torch.arange(n_patch, device=grid.device)
    gy, gz = grid[:, 1:2], grid[:, 2:3]
    xyz = torch.stack([j // (gy * gz), j // gz % gy, j % gz], -1)
    return torch.where((j < grid.prod(1, keepdim=True))[..., None], xyz, -1)


def symmetries() -> tuple[torch.Tensor, torch.Tensor]:
    """The 48 (axis permutation [48,3], per-axis flip [48,3]) pairs; identity first."""
    perms, flips = zip(*itertools.product(itertools.permutations(range(3)),
                                          itertools.product((False, True), repeat=3)))
    return torch.tensor(perms), torch.tensor(flips)


def transform(c: torch.Tensor, dims: torch.Tensor, perm: torch.Tensor, flip: torch.Tensor):
    """Box-local coords c [...,3] in a box of size dims [...,3] -> (c', dims') under a symmetry."""
    c = torch.where(flip, dims - 1 - c, c)
    return c.gather(-1, perm.expand_as(c)), dims.gather(-1, perm.expand_as(dims))


def _hilbert(c: torch.Tensor, bits: int) -> torch.Tensor:
    """Skilling's AxesToTranspose + bit interleave, vectorised."""
    x = [c[..., i] for i in range(3)]
    q = 1 << (bits - 1)
    while q > 1:
        m = q - 1
        for i in range(3):
            hit = (x[i] & q) != 0
            t = (x[0] ^ x[i]) & m
            x0 = torch.where(hit, x[0] ^ m, x[0] ^ t)
            if i:
                x[i] = torch.where(hit, x[i], x[i] ^ t)
            x[0] = x0
        q >>= 1
    for i in (1, 2):
        x[i] = x[i] ^ x[i - 1]
    t = torch.zeros_like(x[0])
    q = 1 << (bits - 1)
    while q > 1:
        t = torch.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x = [v ^ t for v in x]
    key = torch.zeros_like(x[0])
    for b in range(bits - 1, -1, -1):
        for v in x:
            key = (key << 1) | ((v >> b) & 1)
    return key


def _morton(c: torch.Tensor, bits: int) -> torch.Tensor:
    key = torch.zeros_like(c[..., 0])
    for b in range(bits):
        for axis in range(3):
            key |= ((c[..., axis] >> b) & 1) << (3 * b + axis)
    return key


def keys(name: str, c: torch.Tensor, dims: torch.Tensor, seed: int = 17) -> torch.Tensor:
    """Curve sort key of box-local coords c [...,3] (int64, 0 <= c < dims) in boxes of size dims [...,3].
    Distinct within a box."""
    bits = max(int(dims.max()) - 1, 1).bit_length()
    x, y, z = c.unbind(-1)
    ny, nz = dims[..., 1], dims[..., 2]
    if name == "raster":
        return (x * ny + y) * nz + z
    if name == "snake":
        y = torch.where(x % 2 == 0, y, ny - 1 - y)
        z = torch.where((x + c[..., 1]) % 2 == 0, z, nz - 1 - z)
        return (x * ny + y) * nz + z
    if name == "morton":
        return _morton(c, bits)
    if name == "hilbert":
        return _hilbert(c, bits)
    if name == "random":  # a fixed random permutation of the enclosing cube
        table = torch.randperm(1 << 3 * bits, generator=torch.Generator().manual_seed(seed)).to(c.device)
        return table[_morton(c, bits)]
    raise ValueError(f"Unknown curve {name!r}; have {CURVES}")
