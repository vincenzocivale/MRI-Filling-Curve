"""Checks of src/sfc_gdn2/apt.py. Run by hand (no test suite in this repo):

    python tests/apt_check.py                 # CPU: the leaves tile the volume exactly once, arms behave
    python tests/apt_check.py <run dir>       # + GPU: parity with the uniform reading of fm/sfc.py (needs GDN-2)
        [--volume <nifti> | auto]            # default: a synthetic volume; auto: an IXI scan of the benchmark

Parity: an arm that splits everything (tau -1) must equal the existing reading at base_mm / 2, one that splits nothing
(tau 1e9) the existing reading at base_mm; curves raster and morton only (the Hilbert key of a coarse grid embedded in
the finest one has another orientation, still a Hilbert curve)."""
from __future__ import annotations

import argparse

import torch

from sfc_gdn2.apt import base_dims, partition, scores


def tiles(vol_shape, k0, levels, leaves) -> bool:
    """Every voxel of the padded volume belongs to exactly one leaf."""
    cover = torch.zeros(vol_shape, dtype=torch.int32)
    for l, c in enumerate(leaves):
        k = [x // 2 ** l for x in k0]
        for x, y, z in c.tolist():
            cover[x * k[0]:(x + 1) * k[0], y * k[1]:(y + 1) * k[1], z * k[2]:(z + 1) * k[2]] += 1
    return bool((cover == 1).all())


def cpu_checks() -> None:
    g = torch.Generator().manual_seed(0)
    for spacing, shape, levels in [((1.0, 1.0, 1.0), (40, 33, 50), 2), ((1.2, 1.2, 1.2), (50, 61, 47), 3),
                                   ((0.5, 0.5, 4.0), (70, 66, 20), 3)]:
        k0 = base_dims(spacing, 16, levels)
        assert all(q % 2 ** (levels - 1) == 0 for q in k0), k0
        vol = torch.rand(*[-(-n // q) * q for n, q in zip(shape, k0)], generator=g) * (torch.rand(1, generator=g) + .3)
        vol[: vol.shape[0] // 2] = 0   # half the volume is flat: low entropy
        g0 = [s // q for s, q in zip(vol.shape, k0)]
        sc = scores(vol, k0, levels, 16)
        ent = {"score": "entropy", "tau": [1.5] * (levels - 1)}
        leaves, split = partition(sc, g0, levels, ent)
        assert tiles(vol.shape, k0, levels, leaves), (spacing, "entropy arm does not tile")
        n_ent = sum(len(c) for c in leaves)
        rnd, rsplit = partition(sc, g0, levels, {"score": "random"}, counts=split, gen=g)
        assert tiles(vol.shape, k0, levels, rnd), (spacing, "random arm does not tile")
        assert rsplit[0] == split[0], "random arm must split as many base cells as the entropy arm"
        nf = partition(sc, g0, levels, {"tau": [-1.0] * (levels - 1)})[0]
        assert len(nf[-1]) == torch.tensor(g0).prod() * 8 ** (levels - 1) and all(len(c) == 0 for c in nf[:-1])
        nb = partition(sc, g0, levels, {"tau": [1e9] * (levels - 1)})[0]
        assert len(nb[0]) == torch.tensor(g0).prod() and all(len(c) == 0 for c in nb[1:])
        fr = partition(sc, g0, levels, {"frac": [0.25] * (levels - 1)})[1]
        assert fr[0] == round(0.25 * int(torch.tensor(g0).prod()))
        assert n_ent < len(nf[-1]), "entropy arm should have fewer tokens than the finest grid"
        print(f"ok  spacing {spacing} grid {g0} levels {levels}: entropy arm {n_ent} leaves (finest {len(nf[-1])}), "
              f"split per level {split}, random copy {rsplit}")


def gpu_checks(run: str, volume: str | None) -> None:
    import numpy as np

    from sfc_gdn2.data.volume import decode, open_image, patch_dims, patch_grid
    from sfc_gdn2.fm.sfc import SFC

    if volume == "auto":   # the first IXI scan of the benchmark
        import pandas as pd
        volume = pd.read_csv("/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench/volumes.csv",
                             low_memory=False).query("dataset == 'IXI'").sort_values("id")["input"].iloc[0]
    if volume:
        arr, _, (a, c) = decode(open_image(volume))
        arr = arr[:arr.shape[0] // 16 * 16, :arr.shape[1] // 16 * 16, :arr.shape[2] // 16 * 16]
    else:
        arr = (torch.rand(96, 96, 96, generator=torch.Generator().manual_seed(1)) * 800).numpy().astype("float32")
        arr[:48] = 0
        a, c = 1 / 800, 0.0
    sp = (1.0, 1.0, 1.0)

    def prep(patch_mm):
        k = patch_dims(sp, patch_mm)
        return {"volume": torch.from_numpy(np.ascontiguousarray(arr)), "value": (a, c), "k": torch.tensor(k),
                "grid": patch_grid(arr.shape, k), "spacing": torch.tensor(sp), "meta": {}}

    curves = ["raster", "morton"]
    apt = SFC({"checkpoint": run, "args": {"curves": curves, "apt": {"base_mm": 16, "levels": 2, "arms": {
        "fine": {"tau": [-1]}, "base": {"tau": [1e9]}, "ent": {"score": "entropy", "tau": [1.0]},
        "rnd": {"score": "random", "match": "ent"}}}}}, "cuda")
    out = apt.features_apt(prep(16))["features"]
    for mm, arm in ((8, "fine"), (16, "base")):
        ref = SFC({"checkpoint": run, "args": {"curves": curves, "patch_mm": mm}}, "cuda").features(prep(mm))["features"]
        for stat in ("mean", "std", "mean_b4", "mean_b6"):
            x, y = ref[stat].float().cpu(), out[f"{stat}__{arm}"].float().cpu()
            cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
            rel = ((x - y).norm() / x.norm()).item()
            print(f"{'ok ' if cos > 0.999 and rel < 0.03 else 'FAIL'} parity {arm} vs uniform {mm} mm  {stat}: "
                  f"cos {cos:.5f} rel diff {rel:.4f}")
    for arm in ("fine", "base", "ent", "rnd"):
        print(f"arm {arm}: leaves per level {out[f'ntok__{arm}'].tolist()}  encoder {out[f'time__{arm}'].item():.3f}s")
    assert out["ntok__rnd"].tolist() == out["ntok__ent"].tolist(), "matched random arm has another token count"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--volume")
    a = ap.parse_args()
    cpu_checks()
    if a.run:
        gpu_checks(a.run, a.volume)
