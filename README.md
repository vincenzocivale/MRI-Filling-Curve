# MRI-Filling-Curve

Self-supervised pretraining of a causal Gated DeltaNet-2 on 3D MRI with LeJEPA, where the views of a
region are different space-filling-curve serializations of it, evaluated with frozen linear probes.

## Pretraining (`configs/pretrain_lejepa.yaml`)

Nothing is resampled. A patch is a block of the scan's own voxels, `round(patch_mm / spacing)` per axis
(~16 mm: 13³ voxels of a 1.2 mm brain, 102×102×7 of a 0.16×0.16×2.2 mm body scan, one 28 mm slice), so the
patch grid follows the field of view (a brain ~12×15×15, a whole body ~34×9×123). The patch embedding is a
learned continuous kernel in mm, integrated over each voxel's physical extent (`model.KernelEmbed`): every
voxel enters with its own weight, the same kernel serves every spacing, and a scan and the same scan with
each voxel copied into finer voxels give the same token. No position embedding. Per batch of B volumes:

1. **Groups.** N boxes of the patch grid per volume, centred on foreground patches: `groups_global`
   spanning `global_frac` of the foreground bounding box, `groups_local` with `local_edge` patches per edge.
2. **Views.** K per group. A view is the group box rescaled/shifted by up to `jitter` (partial
   overlap) and capped at `max_view_tokens` patches, intensity-augmented (gamma, scale, shift, noise), with
   probability `thick_prob` given simulated thick slices (native slices averaged into ~`thick_mm` slabs along
   a random axis, where the scan's slices are thinner), view A with `mask_ratio` of its aligned cubes (edge drawn from `mask_cells` patches) set to the mask token on their foreground patches, read along a curve drawn
   from `view_curves` in a random cube symmetry, traced within the view's box (any shape: Hilbert/Morton
   are the enclosing power-of-two cube's curve restricted to the box).
3. **Loss.** One masked-prediction task per spatial scale (the H-JEPA pattern, arXiv 2610.06805, with space in
   place of time), each with its own projector and SIGReg: A's masked patch tokens → B's same patches; mean
   token of each cube of `levels` patches fully masked in A → B's same cube; the view's mean foreground token
   (top) → B's. `(1-λ)·mean(invariances) + λ·mean(SIGRegs)`.

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
sfc pretrain-pool configs/pretrain_lejepa.yaml       # catalog -> pretraining manifest (header pass: a CPU job)
sfc pretrain      configs/pretrain_lejepa.yaml       # -> <output_root>/lejepa-<hash>/encoder_step*.pt
sfc bench-split   configs/bench/bench.yaml           # downstream volumes + fixed subject splits
sfc bench-extract configs/bench/bench.yaml --group <g>   # features of a model group (ours: `sfc`)
sfc bench-probe   configs/bench/bench.yaml           # volume tasks, every model
sfc bench-segdec  configs/bench/bench.yaml --stage prep|train ...   # segmentation, full resolution
```

Pretraining pool (`data.pool`, DECISIONS.md Version 1): every scan of MR-RATE (phase-1 brain series), FOMO300K
after dedup, SALD, CoRR and Petfrog; no subject of the downstream benchmark. Checkpoints `encoder_step*.pt` every
`save_every` steps, starting with step 0 (the run's own init). Evaluation is the benchmark (`src/sfc_gdn2/bench/`):
ours is one more model there (`fm/sfc.py`), with the same volumes, splits, probes and segmentation protocol.

Storage: each scan's voxels exactly as stored (dtype kept, canonical reorientation = axis permutation and
flips only) with the value map `a·raw + c` (header slope/intercept over the 99th percentile of the foreground; no
clipping). With `cache_dir` set, cached once as `.npy` under `cache_dir/native/`; the pretraining pool runs with
`cache_dir: null` (decoded from the sources, zip members in memory). Values become
fp32 on the GPU, where the volume is cut into patches; the embedding runs in fp32.

**FOMO300K is a superset of OpenMind:** never pool them as independent cohorts.

No dataset or GDN-2 code is redistributed here (`THIRD_PARTY.md`).

## SLURM (CINECA Leonardo)

`slurm/submit.py` turns a YAML preset into a batch script (repo root as cwd, conda env active) and
submits it; any `sfc` command or script goes after `--`:

```bash
cp slurm/local.yaml.example slurm/local.yaml      # once: mail, conda base / env path
python slurm/submit.py slurm/presets/dbg.yaml -- sfc fm configs/fm/brainiac.yaml --images a.nii.gz --out o
python slurm/submit.py slurm/presets/cpu.yaml -- sfc pretrain-pool configs/pretrain_lejepa.yaml --workers 8
python slurm/submit.py slurm/presets/4gpu.yaml -- -m sfc_gdn2.cli pretrain configs/pretrain_lejepa.yaml
```

Presets: `dbg` (1 GPU, debug QOS, 30 min), `1gpu`, `4gpu` and `8gpu` (torchrun DDP, 1 and 2 nodes),
`cpu` (`lrd_all_serial`). Under torchrun the command is `-m <module>` or a script, never `python`.
Config: `slurm/defaults.yaml` <- `slurm/local.yaml` <- preset <- `--set key=value`; logs in `outputs/slurm/`.
Configs carry Leonardo paths (runs and raw data on `$SCRATCH`, metadata on `$WORK`). `configs/leonardo/` keeps
aliases of the benchmark configs for jobs queued before the move to `configs/bench/`.
flash-attn's upstream wheels need GLIBC 2.32 (Leonardo: 2.28); build it with `slurm/build_flash_attn.sh`.

## External foundation models

One entrypoint for every model: given image files, each model returns **its own** features, computed by
the original repo's code (preprocessing, network, inference procedure, feature output) in a dedicated
conda env. Nothing of ours sits between the NIfTI and the model.

```bash
source configs/fm/leonardo.env        # SFC_FM_ENVS / SFC_FM_REPOS / SFC_FM_MODELS (see configs/fm/README.md)
conda env create -p $SFC_FM_ENVS/fm-<family> -f envs/fm/fm-<family>.yml     # once per family
sfc fm configs/fm/<name>.yaml --images a.nii.gz b.nii.gz --out feats/<name> [--device cpu]
sfc fm configs/fm/mome_plus.yaml --csv cases.csv --out feats/mome_plus   # multi-modal: id,<modality>,...
```

Each image gives `<out>/<id>.pt` = `{features: {name: tensor}, canonical, derived, meta, provenance}`:
`canonical` names the embedding the repo itself uses, `derived` lists anything we added (e.g. a pooling
the repo does not define), `meta` maps the feature grid back to the image (for voxel/patch probes).
`tests/fm/<family>_parity.py` (run in the model's env) checks a wrapper against the untouched original
pipeline. Linear probing on these features is not wired yet.
