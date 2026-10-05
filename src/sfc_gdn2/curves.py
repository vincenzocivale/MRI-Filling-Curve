"""3D space-filling curves over an n^3 patch grid, and their views under the cube's symmetries.

Patches live in canonical raster index space: index = x*n^2 + y*n + z. A curve is a
permutation `order` of those indices (sequence position -> patch index). A *view* is the same
curve traced on the grid after one of the 48 symmetries of the cube (axis permutation x
reflection), so every view has identical locality statistics but a different traversal.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

CURVES = ("raster", "snake", "morton", "hilbert", "random")


def grid_coords(n: int) -> np.ndarray:
    """[n^3, 3] integer coordinates in canonical raster order."""
    return np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)


def _hilbert_key(c: np.ndarray, bits: int) -> np.ndarray:
    """Skilling's AxesToTranspose + bit interleave, vectorised over points."""
    x = [c[:, i].astype(np.int64) for i in range(3)]
    q = 1 << (bits - 1)
    while q > 1:
        m = q - 1
        for i in range(3):
            hit = (x[i] & q) != 0
            t = (x[0] ^ x[i]) & m
            x0 = np.where(hit, x[0] ^ m, x[0] ^ t)
            if i:
                x[i] = np.where(hit, x[i], x[i] ^ t)
            x[0] = x0
        q >>= 1
    for i in (1, 2):
        x[i] = x[i] ^ x[i - 1]
    t = np.zeros_like(x[0])
    q = 1 << (bits - 1)
    while q > 1:
        t = np.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x = [v ^ t for v in x]
    key = np.zeros_like(x[0])
    for b in range(bits - 1, -1, -1):
        for v in x:
            key = (key << 1) | ((v >> b) & 1)
    return key


def _morton_key(c: np.ndarray, bits: int) -> np.ndarray:
    key = np.zeros(len(c), dtype=np.int64)
    for b in range(bits):
        for axis in range(3):
            key |= ((c[:, axis].astype(np.int64) >> b) & 1) << (3 * b + axis)
    return key


def _snake_key(c: np.ndarray, n: int) -> np.ndarray:
    x, y, z = c.T
    y = np.where(x % 2 == 0, y, n - 1 - y)
    z = np.where((x + c[:, 1]) % 2 == 0, z, n - 1 - z)
    return (x * n + y) * n + z


def order(name: str, n: int, seed: int = 17) -> np.ndarray:
    """Sequence position -> canonical patch index."""
    c = grid_coords(n)
    bits = max(int(n - 1).bit_length(), 1)
    if name == "raster":
        return np.arange(n ** 3)
    if name == "random":
        return np.random.default_rng(seed).permutation(n ** 3)
    if name == "snake":
        key = _snake_key(c, n)
    elif name == "morton":
        key = _morton_key(c, bits)
    elif name == "hilbert":
        if n & (n - 1):
            raise ValueError("Hilbert ordering requires a power-of-two grid.")
        key = _hilbert_key(c, bits)
    else:
        raise ValueError(f"Unknown curve {name!r}; have {CURVES}")
    return np.argsort(key, kind="stable")


def cube_symmetries() -> list[tuple[tuple[int, int, int], tuple[bool, bool, bool]]]:
    """The 48 (axis permutation, per-axis flip) pairs; identity first."""
    return [(p, f) for p in itertools.permutations(range(3))
            for f in itertools.product((False, True), repeat=3)]


class CurveViews:
    """Distinct traversals of one curve under the cube symmetries.

    `perms[v]`: sequence position -> canonical patch index for view v.
    `ranks[v]`: canonical patch index -> sequence position (inverse of perms[v]).
    View 0 is the untransformed curve (used for downstream feature extraction).
    """

    def __init__(self, name: str, n: int, seed: int = 17):
        self.name, self.n = name, n
        ordered = grid_coords(n)[order(name, n, seed)]
        perms, seen = [], set()
        for axes, flips in cube_symmetries():
            c = ordered[:, axes]
            c = np.where(np.array(flips), n - 1 - c, c)
            p = (c[:, 0] * n + c[:, 1]) * n + c[:, 2]
            if (key := p.tobytes()) not in seen:
                seen.add(key)
                perms.append(p)
        self.perms = torch.from_numpy(np.stack(perms))
        self.ranks = torch.argsort(self.perms, dim=1)

    def __len__(self) -> int:
        return len(self.perms)
