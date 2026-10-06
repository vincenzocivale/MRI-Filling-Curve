# CLAUDE.md

Guidance for Claude Code in this repository. User-facing docs: `README.md`.

## What this is

LeJEPA pretraining of a **causal** Gated DeltaNet-2 on 3D MRI: per volume, groups (boxes of the patch
grid); per group, two views that differ in serialization (space-filling curve × cube symmetry), box
jitter and intensity; view A is masked. Global term: last forward state, invariance + SIGReg. Token
term: bi tokens of A's masked patches and B's same patches pulled together (symmetric, no stop-grad:
a stop-grad target without EMA drifted) + SIGReg on 64 token embeddings per volume (~1000 samples, LeJEPA's calibrated range). Frozen linear probes
(TotalSeg segmentation, brain age, sex) per inference curve.

## Environment

```bash
/home/fcorrentio/data/MRI-US/miniconda3/envs/sfc-gdn2/bin/python -m pytest -q   # or: conda activate sfc-gdn2
ruff check .
```

GDN-2 is at `/raid/DATASETS/_external_deps/GatedDeltaNet-2` (on the env's path). `tests/test_model_gpu.py`
needs CUDA + GDN-2 and is skipped otherwise; the first run spends minutes compiling Triton kernels.
Up to 4 GPUs at a time; pick ones with free memory (`nvidia-smi`), set `CUDA_VISIBLE_DEVICES`.

## Layout (`src/sfc_gdn2/`)

- `curves.py`: curve orders; `CurveViews(name, n)`: `perms`/`ranks` of the 48 cube-symmetry views (view 0 = identity).
- `model.py`: `Encoder` (linear patch embed, mask token, causal GDN-2 stack, no coordinate embedding).
- `lejepa.py`: `SIGReg`, `Projector`, `LeJEPA` (`boxes` → `jittered` → `serialize` → `read` (fwd + reversed pass) → glob/tok heads).
- `pretrain.py`: `Pretrainer` (bf16, fused AdamW, decay on matrices only, warmup+cosine, grad clip,
  non-finite steps skipped, `encoder_step*.pt` incl. step 0).
- `probe.py`: tasks, `LinearProbe` (val or subject-grouped CV selection, one test pass, bootstrap CI),
  `probe_run` (per inference curve: pretrained vs init), `probe_raw` (raw + position).
- `data/`: manifests (`zip_bids`, `totalseg`), labels, subject-level splits, `VolumeStore` cube cache, loader.
- `fm/`: external foundation models, image -> the model's own features. `api.extract` (`sfc fm`) runs `worker.py` under the model's conda env (`envs/fm/`, path in `configs/fm/*.yaml`); each wrapper (`base.Wrapper`) imports the ORIGINAL repo code for preprocessing, network, inference and outputs. Parity scripts in `tests/fm/`.
- `cli.py`: `sfc prepare|split|pretrain|probe|fm|summarize`.
- `slurm/`: `submit.py` + YAML presets to sbatch any command on Leonardo (`python slurm/submit.py slurm/presets/<p>.yaml -- <cmd>`).

## Invariants (do not break)

- Model selection only on probe val / train+val CV, never on test.
- `grad_clip: 1.0`: GDN-2 gives rare single-token gradient spikes (near-zero recurrent outputs × output RMSNorm); unclipped, one NaN'd a run.
- Weight decay on matrices only (GDN-2 `A_log`/`dt_bias` are `_no_weight_decay`).
- Projector BatchNorm without running stats.
- A global-only objective makes token features a per-volume code (v5: 51% of token variance between
  volumes, dense probes decaying; SIGReg on centred or raw projector outputs did not stop it, the
  projector absorbs it). The token term is what targets dense quality (DINOv2/iBOT pattern).
- No coordinate embedding: the serialization is the only spatial signal (an embedding once gave every anchor one volume code).
- Views must differ in content, not only order (bag-of-patches shortcut). Never ask two causal traversals to agree at a position: their states summarise different parts of the volume (tried: invariance stuck at correlation ~0.07).
- Report probes next to `init`, `raw`, `position`, `chance_ap`; sex alone cannot rank encoders.
- All randomness derives from `seed`. Volumes resampled isotropically, aspect preserved.
- FOMO300K ⊃ OpenMind. No dataset or GDN-2 code in this repo.
- External FMs: no re-implemented preprocessing / network / pooling. Use the original repo's code in its own env; anything we add is listed in `derived`.
