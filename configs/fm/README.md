# External foundation-model configs

One file per model (checkpoint): `sfc fm configs/fm/<name>.yaml --images a.nii.gz ... --out <dir>`.

```yaml
model: brainiac                    # key of sfc_gdn2.fm.MODELS
env: ${SFC_FM_ENVS}/fm-brainiac    # conda env built from envs/fm/<env>.yml
repo: ${SFC_FM_REPOS}/BrainIAC     # clone of the ORIGINAL repo (imported, never vendored); pin in `commit`
commit: ba60f45                    # commit the wrapper was verified against
checkpoint: ${SFC_FM_MODELS}/BrainIAC/BrainIAC.ckpt
args: {}                           # wrapper options, documented in the wrapper module
save: {features: [embedding], dtype: float16}   # optional: what to STORE (computation unchanged); default all, as computed
```

Paths use three environment variables, so the same configs work on every machine:
`SFC_FM_ENVS` (conda envs), `SFC_FM_REPOS` (original repo clones), `SFC_FM_MODELS` (checkpoints).
On Leonardo: `source configs/fm/leonardo.env`.
