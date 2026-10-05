# MRI-Filling-Curve

Self-supervised pretraining of a causal Gated DeltaNet-2 on 3D MRI with LeJEPA, where the views of a
region are different space-filling-curve serializations of it, evaluated with frozen linear probes.

## Pretraining (`configs/pretrain_lejepa.yaml`)

A volume is a `16³` grid of `8³`-voxel patches (linear embedding, no position embedding). Per batch of
B volumes:

1. **Groups.** N boxes of the patch grid per volume, centred on foreground patches: `groups_global`
   spanning `global_frac` of the foreground bounding box, `groups_local` with `local_edge` patches per edge.
2. **Views.** K per group. A view is the group box rescaled/shifted by up to `jitter` (partial
   overlap), intensity-augmented (gamma, scale, shift, noise), with `mask_ratio` of its patches set to
   the mask token, read along a curve drawn from `view_curves` in a random cube symmetry.
3. **Loss.** View embedding = mean encoder output over its tokens → projector;
   `(1-λ)·invariance(views of a group) + λ·SIGReg(per-volume-centred embeddings)`.

Views differ in content as well as order: with no position embedding, serialization alone would be
satisfied by an encoder that ignores order.

## Install

```bash
conda env create -f environment.yml && conda activate sfc-gdn2
pip install -e . --no-deps
```

The official [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) is required and not
vendored (NVIDIA Source Code License-NC); put a clone on the env's path so
`from lit_gpt.gdn2 import GatedDeltaNet2` resolves:

```bash
echo /path/to/GatedDeltaNet-2 > "$CONDA_PREFIX"/lib/python3.11/site-packages/gdn2.pth
```

`torch==2.8.0` and `flash-linear-attention==v0.4.2` are pinned in `environment.yml` on purpose
(prebuilt flash-attn wheels; later FLA drops the `use_exp2` argument GDN-2 passes).

## Usage

```bash
sfc prepare  configs/datasets/fomo300k.yaml          # scan a local dataset root -> manifest CSV
sfc split    configs/splits/fomo300k_sex.yaml        # exact subject counts: pretrain/train/val/test
sfc split    configs/splits/totalseg_mri.yaml        # TotalSegmentator MRI: pretrain + seg probe splits
sfc pretrain configs/pretrain_lejepa.yaml            # -> <output_root>/lejepa-<hash>/
sfc probe    configs/probe_{sex,age,totalseg}.yaml --run-dir <run> [...] [--raw configs/pretrain_lejepa.yaml]
sfc summarize <output_root>                           # -> probe_summary.csv
```

Pretraining uses every `pretrain` row of `data.manifests` (FOMO300K brain T1w + TotalSegmentator MRI)
and saves `encoder_step*.pt` every `save_every` steps, starting with step 0 (the run's own init).

| probe | task | features | head | selected on |
|---|---|---|---|---|
| `probe_totalseg.yaml` | per-patch majority class, 50 classes + background | patch token: `token` (causal) / `bi` (++ backward pass) | logistic | val macro AP (fg) |
| `probe_age.yaml` | age − cohort train mean, IXI/NKI/OASIS1 | pooled tokens (`mean`, `late`, `last`, `fg_mean`, `fg_meanstd`) | ridge | 5-fold CV R² |
| `probe_sex.yaml` | sex (sanity check: trivial cues give AUC ~0.81) | pooled tokens | logistic | 5-fold CV AUC |

Each probe reports, per inference curve in the run's `view_curves`, the pretrained encoder (best
checkpoint × features × L2) and its step-0 `init`, scores test once with a volume-bootstrap 95% CI and
saves `test_preds.pt` for paired comparisons. Segmentation also reports `@token`/`@bi` separately and
`chance_ap`. `--raw` adds encoder-free baselines (patch intensities; for segmentation also `position`,
random Fourier features of patch coordinates). A probe refuses to run if a probe subject was seen in
pretraining.

Preprocessing (canonical reorientation, foreground 1–99th percentile, aspect-preserving isotropic
resample with the longest side spanning `target_shape`, zero-padded to the cube) runs once per scan and is cached as a float16 `.npy` cube under `cache_dir/cubes/`;
later epochs memory-map it.

**FOMO300K is a superset of OpenMind:** never pool them as independent cohorts.

No dataset or GDN-2 code is redistributed here (`THIRD_PARTY.md`).
