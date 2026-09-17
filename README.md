# MRI-Filling-Curve

SFC × Gated DeltaNet-2 for 3D MRI

A small, controlled benchmark for one question:

> Does locality-preserving 3D serialization make a measurable difference for Gated DeltaNet-2 before committing to full MRI foundation-model pretraining?

The benchmark keeps data, preprocessing, architecture, masking, optimizer, and update budget fixed. Only the token ordering changes.

**Curves:** `raster`, `snake`, `morton`, `hilbert`, `random`.

**Datasets:**
- `Forithmus/MR-RATE-atlas` — atlas-registered MRI, including zipped study archives. The official dataset is gated and non-commercial. 
- `FOMO-MRI/FOMO300K` — modified-BIDS NIfTI collection. The official dataset is gated and includes constituent-specific DUAs.
- `MIC-DKFZ/OpenMind` — modified-BIDS/OpenNeuro-style NIfTI collection.

The repository does **not** redistribute datasets or NVIDIA Gated DeltaNet-2 code. External resources retain their own licenses.

## What is measured

1. **No-training diagnostics**
   - spatial-neighbor recall within sequence windows;
   - consecutive-step 3D distance;
   - MRI sequence total variation after patch pooling.
2. **Micro-training diagnostic**
   - masked patch reconstruction with a small bidirectional GDN-2 encoder;
   - fixed number of optimizer updates (`500` by default), not full pretraining;
   - validation MSE/MAE and convergence AUC per dataset and curve.

The primary evidence is whether geometric locality and micro-task efficiency improve consistently across independent MRI sources.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Install the **official** Gated DeltaNet-2 implementation separately, following NVIDIA's repository instructions, so this import works:

```python
from lit_gpt.gdn2 import GatedDeltaNet2
```

The official implementation currently uses custom FLA/Triton kernels and its own CUDA/PyTorch requirements. Do not vendor it into this repository.

## 1. Build canonical manifests

Point each dataset config at an already-downloaded local root.

```bash
python scripts/prepare.py configs/datasets/mrrate_atlas.yaml
python scripts/prepare.py configs/datasets/fomo300k.yaml
python scripts/prepare.py configs/datasets/openmind.yaml
```

Outputs are CSV manifests with the same schema:

```text
dataset,sample_id,subject,session,modality,path,mask_path,split
```

MR-RATE archives can remain zipped; members are represented as `zip://...::...nii.gz` and extracted lazily into a local cache.

## 2. Run the controlled benchmark

Edit `configs/benchmark.yaml` with the manifest(s) for one benchmark cohort, then:

```bash
python scripts/run.py configs/benchmark.yaml
```

**Do not treat FOMO300K and OpenMind as independent pooled cohorts:** FOMO300K is a superset of OpenMind. Run them separately (or explicitly de-duplicate them) and report per-source results.

For a fast geometry-only check:

```bash
python scripts/run.py configs/benchmark.yaml --geometry-only
```

## 3. Aggregate

```bash
python scripts/summarize.py outputs
```

Each run is self-contained:

```text
outputs/<run_id>/
  config.yaml
  environment.json
  geometry.csv
  history.csv
  metrics.json
  per_dataset.csv
  checkpoint.pt
```

## Fairness rules

- same manifest rows and split for every curve;
- same random mask in **3D patch coordinates**, then reordered by the selected curve;
- same initialization seed, optimizer and number of updates;
- same 3D coordinate encoding;
- curve-specific ordering is the only experimental variable;
- report each dataset separately before pooled metrics.

`random` is a negative control. `snake` is necessary to distinguish simple removal of raster discontinuities from multiscale locality preservation.
