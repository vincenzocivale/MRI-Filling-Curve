# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A controlled benchmark testing whether locality-preserving 3D space-filling-curve serialization
(`raster`, `snake`, `morton`, `hilbert`, `random`) improves Gated DeltaNet-2 (GDN-2) performance on
MRI volumes. Everything except token ordering (data, preprocessing, architecture, masking, optimizer,
update budget) is held fixed — the point is an ablation, not a model to train to convergence.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

The official GDN-2 implementation (`NVlabs/GatedDeltaNet-2`, NVIDIA Source Code License-NC) must be
installed separately so that `from lit_gpt.gdn2 import GatedDeltaNet2` resolves — it is **not**
vendored here (see `THIRD_PARTY.md`). `src/sfc_gdn2/model.py` raises a clear `RuntimeError` if it's
missing; without it, only geometry-only runs and curve/metrics tests work (no actual training).

## Commands

```bash
# 1. Build a manifest from an already-downloaded local dataset root
python scripts/prepare.py configs/datasets/mrrate_atlas.yaml   # or fomo300k.yaml / openmind.yaml

# 2. Run the benchmark (all curves in configs/benchmark.yaml, one set of manifests)
python scripts/run.py configs/benchmark.yaml
python scripts/run.py configs/benchmark.yaml --geometry-only   # fast, no training/GDN-2 needed

# 3. Aggregate metrics.json across run directories
python scripts/summarize.py outputs

# Tests
pytest                       # pythonpath=src is set via pyproject.toml
pytest tests/test_curves.py -k hilbert

# Lint
ruff check .
```

There is no top-level `sfc_gdn2` CLI entrypoint installed by the package; `scripts/{prepare,run,summarize}.py`
are the thin argparse wrappers around `sfc_gdn2.cli.{prepare,run,summarize}`.

## Architecture

**Curve ordering is the single experimental variable.** `src/sfc_gdn2/curves.py::order(name, n, seed)`
returns a permutation of a canonical raster-ordered `n³` coordinate grid. `MRIProbe` (in `model.py`)
always embeds patches and applies masking in canonical raster order, then permutes into the curve
order (`x[:, self.perm]`) before running GDN-2 blocks, and un-permutes (`argsort(perm)`) before the
reconstruction loss — so masking/coords/loss are curve-invariant and only the sequence GDN-2 actually
sees changes.

**Two independent diagnostics, both keyed by curve:**
- *Geometry* (`metrics.py::geometry_metrics`, no training, no GDN-2 dependency): spatial-neighbor
  recall within sequence windows, consecutive-step 3D distance stats. Driven by `--geometry-only`.
- *Micro-training* (`engine.py::microtrain`): a small bidirectional GDN-2 encoder (`BiGDN2Block` runs
  GDN-2 forward and on the time-reversed sequence, averages) does masked-patch reconstruction for a
  fixed step budget (`microtrain.steps`, default 500) — never full pretraining. Reports val MSE/MAE
  and convergence AUC (`metrics.py::auc`) per dataset and per curve.

**Data pipeline** (`src/sfc_gdn2/data/`):
- `manifest.py::build` — dataset-specific scanners producing a canonical CSV manifest
  (`dataset,sample_id,subject,session,modality,path,mask_path,split`). `mrrate_atlas` reads NIfTI
  members directly out of `*_atlas.zip` archives (`zip://archive::member` paths, extracted lazily);
  other dataset kinds fall back to a generic BIDS-like directory walk (`bids_like`).
- `volume.py` — `materialize` extracts zip-backed volumes into `cache_dir` on first access;
  `load_volume` reorients to canonical, clips to the 1st/99th percentile, normalizes to [0,1], and
  trilinearly resamples to `data.target_shape` (must be cubic, divisible by `patch_size`);
  `patchify` unfolds into `[N, patch³]` non-overlapping patches.
- `dataset.py::balanced_split` keeps all scans from one subject on the same side of the train/val
  split when subject IDs are available, capped by `max_train_per_dataset`/`max_val_per_dataset`.

**Run identity**: every run/config is hashed via `io.py::stable_id` (sha1 of sorted-key JSON) into the
output dir name (`outputs/<curve>-<hash>/` or `outputs/geometry-<hash>/`), so identical configs collide
predictably and `summarize.py` can glob `outputs/*/metrics.json` across runs.

## Fairness rules (do not violate when editing the pipeline)

- Same manifest rows and split across every curve in a run.
- Same random mask in **3D patch coordinates** (canonical raster order), only reordered per curve —
  masking must happen before the curve permutation, not after.
- Same init seed, optimizer, and number of updates across curves.
- Same 3D coordinate encoding regardless of curve.
- **Do not pool FOMO300K and OpenMind** as independent cohorts — FOMO300K is a superset of OpenMind.
  Run them separately or explicitly de-duplicate, and always report per-source results before pooled
  metrics.
- `random` curve is the negative control; `snake` isolates "removing raster discontinuities" from
  genuine multiscale locality preservation (Hilbert/Morton) — don't drop either when comparing curves.

## Licensing constraints

No dataset or GDN-2 code is redistributed in this repo (`THIRD_PARTY.md`). Point dataset configs
(`configs/datasets/*.yaml`) at already-downloaded local roots — don't add download/fetch logic for the
gated datasets (MR-RATE-atlas, FOMO300K) into this codebase.
