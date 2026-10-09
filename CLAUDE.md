# CLAUDE.md

Guidance for Claude Code in this repository. User-facing docs: `README.md`.

## What this is

LeJEPA pretraining of a **causal** Gated DeltaNet-2 on 3D MRI: per volume, groups (boxes of the patch
grid); per group, two views that differ in serialization (space-filling curve × cube symmetry), box
jitter and intensity; view A is masked in aligned cubes. One masked-prediction task per spatial scale (H-JEPA
pattern, arXiv 2610.06805, space for time), each with its own projector + SIGReg: patch level, bi tokens of A's masked
(foreground-only) patches and B's same patches pulled together (symmetric, no stop-grad: a stop-grad target without EMA
drifted); cube levels (`levels` patches per edge), mean token of A's fully masked cubes -> B's; top, the view's mean
foreground token. Frozen linear probes (TotalSeg segmentation, brain age, sex): every checkpoint, fixed L2, features
averaged over the run's curves.

## Environment

CINECA Leonardo (project IscrC_SFMRI), env `sfc-gdn2` at `/leonardo_work/IscrC_SFMRI/fcorrent/envs/sfc-gdn2`:

```bash
source ~/mri-sfc/env.sh      # activates the env, PYTHONNOUSERSITE=1, caches on $WORK, cd repo
python -m pytest -q && ruff check .
```

GDN-2 is at `/leonardo_work/IscrC_SFMRI/fcorrent/repos/GatedDeltaNet-2` (on the env's path via `gdn2.pth`).
flash-attn is built from source (`slurm/build_flash_attn.sh`): the upstream wheels need GLIBC 2.32, Leonardo has 2.28.
Configs with Leonardo paths are in `configs/leonardo/` (runs on $SCRATCH, native cache on $FAST, raw data on $SCRATCH).
GPU work goes through SLURM (`slurm/submit.py`, 4x A100 64GB per node; short tests on `slurm/presets/dbg.yaml`);
compute nodes are offline, so data, weights and sdists are fetched on a login node.
`tests/test_model_gpu.py` needs CUDA + GDN-2 and is skipped otherwise; the first run spends minutes compiling
Triton kernels (`TRITON_CACHE_DIR` on $WORK keeps them across jobs).

## Layout (`src/sfc_gdn2/`)

- `curves.py`: curve sort `keys` over box-local patch coords (any box shape), the 48 cube `symmetries` + `transform`, `grid_coords`.
- `model.py`: `KernelEmbed` (continuous kernel in mm, separable learned basis, integrated per voxel: one [P,d] weight per scan),
  `Encoder` (mask token, causal GDN-2 stack, no coordinate embedding; `packed` runs variable-length sequences via cu_seqlens).
- `lejepa.py`: `SIGReg`, `Projector`, `LeJEPA` (`boxes` → `jittered` → `serialize` → `read` (fwd + reversed pass, cube masks) → glob/tok/`cells` heads).
- `pretrain.py`: `Pretrainer` (bf16, fused AdamW, decay on matrices only, warmup+cosine, grad clip,
  non-finite steps skipped, `encoder_step*.pt` incl. step 0).
- `probe.py`: tasks, `LinearProbe` (fixed L2 from the probe config, fit on train+val, one test pass per candidate,
  bootstrap CI), `probe_run` (every checkpoint incl. step 0, curve-averaged bi features), `probe_raw` (raw + position).
- `data/`: manifests (`zip_bids`, `totalseg`), labels, subject-level splits, `VolumeStore` native cache, loader (volumes as a list;
  `to_device` makes values + per-scan [N,P] patches; `grid`, `k`, `spacing` [B,3]).
- `fm/`: external foundation models, image -> the model's own features. `api.extract` (`sfc fm`) runs `worker.py` under the model's conda env (`envs/fm/`, path in `configs/fm/*.yaml`); each wrapper (`base.Wrapper`) imports the ORIGINAL repo code for preprocessing, network, inference and outputs. Parity scripts in `tests/fm/`.
- `bench.py`: downstream benchmark volume list (one per subject/session/sequence, `dense` = stride-8 maps kept),
  input staging (OASIS Analyze -> NIfTI, hard links for paths with spaces) and fixed 70/10/20 subject splits;
  `sfc bench-split|bench-extract configs/leonardo/bench.yaml` (FM groups sharing one preprocessing, CPU prefetch).
- `bench_probe.py` (`sfc bench-probe`): one probe for every model on cached globals (linear + MLP heads, grid and
  selection on val); `bench_seg.py` + `bench_geom.py`: stride-8 segmentation probe (diagnostic) and map geometry.
- `bench_segdec.py` + `fm/segrun.py` (`sfc bench-segdec`): segmentation through each model's OFFICIAL downstream
  decoder (wrapper hooks `seg_input` / `seg_net` / optional `seg_preprocess`) on its frozen pretrained encoder,
  same recipe for all, `init: random` baseline; inputs cached per preprocessing group, no feature cache.
- `cli.py`: `sfc prepare|split|pretrain|probe|fm|summarize|bench-split|bench-extract|bench-probe|bench-segdec`.
- `slurm/`: `submit.py` + YAML presets to sbatch any command on Leonardo (`python slurm/submit.py slurm/presets/<p>.yaml -- <cmd>`).

## Invariants (do not break)

- No model selection on test. Probes have none at all: L2 per feature kind is fixed in the probe configs (v7's val/CV
  choices); every checkpoint is reported.
- `grad_clip: 1.0`: GDN-2 gives rare single-token gradient spikes (near-zero recurrent outputs × output RMSNorm); unclipped, one NaN'd a run.
- Weight decay on matrices only (GDN-2 `A_log`/`dt_bias` are `_no_weight_decay`).
- Projector BatchNorm without running stats.
- No separate global objective: v7's last-state global term competed with the token term (age below raw). The volume
  summary is the top of the token pyramid (mean foreground bi token), and that is what volume probes read.
- A global-only objective makes token features a per-volume code (v5: 51% of token variance between
  volumes, dense probes decaying; SIGReg on centred or raw projector outputs did not stop it, the
  projector absorbs it). The token term is what targets dense quality (DINOv2/iBOT pattern).
- No coordinate embedding: the serialization is the only spatial signal (an embedding once gave every anchor one volume code).
- Views must differ in content, not only order (bag-of-patches shortcut). Never ask two causal traversals to agree at a position: their states summarise different parts of the volume (tried: invariance stuck at correlation ~0.07).
- Report probes next to `init`, `raw`, `position`, `chance_ap`; sex alone cannot rank encoders.
- All randomness derives from `seed`.
- No voxel is resampled, averaged or clipped before the encoder: patches are blocks of native voxels
  (~`patch_mm`), embedded by a continuous kernel integrated per voxel; values fp32 from the stored ints.
  (A common 2 mm grid threw away everything finer: all FOMO, 684 TotalSeg scans below 1 mm.) Pool spacing is
  0.16-28 mm; TotalSeg MRI is 86% anisotropic (>2x); FOMO brains ~1-1.2 mm. The only compression is learned
  (token = d_model numbers; kernel rank per axis).
- numpy's hugepage madvise is off (`sfc_gdn2/__init__.py`): on this host it made volume loads 40x slower.
- FOMO300K ⊃ OpenMind. No dataset or GDN-2 code in this repo.
- External FMs: no re-implemented preprocessing / network / pooling. Use the original repo's code in its own env; anything we add is listed in `derived`.
