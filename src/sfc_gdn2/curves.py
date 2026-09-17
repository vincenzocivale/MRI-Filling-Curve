from __future__ import annotations

import math
import numpy as np


def _coords(n: int) -> np.ndarray:
    return np.array([(x, y, z) for x in range(n) for y in range(n) for z in range(n)], dtype=np.int32)



def _hilbert_distance(point: list[int], bits: int, dims: int) -> int:
    """Skilling-style transpose mapping: integer point -> Hilbert distance."""
    x = list(point)
    q = 1 << (bits - 1)
    while q > 1:
        mask = q - 1
        for i in range(dims):
            if x[i] & q:
                x[0] ^= mask
            else:
                t = (x[0] ^ x[i]) & mask
                x[0] ^= t; x[i] ^= t
        q >>= 1
    for i in range(1, dims): x[i] ^= x[i - 1]
    t = 0; q = 1 << (bits - 1)
    while q > 1:
        if x[dims - 1] & q: t ^= q - 1
        q >>= 1
    for i in range(dims): x[i] ^= t
    out = 0
    for b in range(bits - 1, -1, -1):
        for i in range(dims): out = (out << 1) | ((x[i] >> b) & 1)
    return out

def _morton_code(x: int, y: int, z: int) -> int:
    out = 0
    x, y, z = int(x), int(y), int(z)
    bits = max(x.bit_length(), y.bit_length(), z.bit_length(), 1)
    for b in range(bits):
        out |= ((x >> b) & 1) << (3*b)
        out |= ((y >> b) & 1) << (3*b + 1)
        out |= ((z >> b) & 1) << (3*b + 2)
    return out


def order(name: str, n: int, seed: int = 17) -> np.ndarray:
    c = _coords(n)
    if name == "raster":
        idx = np.arange(len(c))
    elif name == "snake":
        seq = []
        for x in range(n):
            ys = range(n) if x % 2 == 0 else range(n-1, -1, -1)
            for y in ys:
                zs = range(n) if (x + y) % 2 == 0 else range(n-1, -1, -1)
                seq.extend((x, y, z) for z in zs)
        pos = {tuple(v): i for i, v in enumerate(c)}
        idx = np.array([pos[p] for p in seq])
    elif name == "morton":
        idx = np.argsort([_morton_code(*p) for p in c], kind="stable")
    elif name == "hilbert":
        p = int(round(math.log2(n)))
        if 2**p != n:
            raise ValueError("Hilbert ordering requires a power-of-two grid per axis.")
        idx = np.argsort([_hilbert_distance(v.tolist(), p, 3) for v in c], kind="stable")
    elif name == "random":
        idx = np.random.default_rng(seed).permutation(len(c))
    else:
        raise ValueError(f"Unknown curve: {name}")
    return idx.astype(np.int64)


def ordered_coords(name: str, n: int, seed: int = 17) -> np.ndarray:
    c = _coords(n)
    return c[order(name, n, seed)]
