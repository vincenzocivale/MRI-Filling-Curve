"""Downstream benchmark of every model (the external FMs and ours, fm/), with fixed volumes and splits.

- `split`: volume list, input staging, subject-level 70/10/20 splits (`sfc bench-split`).
- `probe`: frozen-feature probes of the volume tasks on the stored features (`sfc bench-probe`).
- `segdec`: segmentation through each model's own network on its frozen encoder, full resolution (`sfc bench-segdec`).
- `geom`: source voxel -> model input grid, GT reader (used by the wrappers and fm/segrun.py in the model envs:
  numpy / torch / nibabel only).
Config: configs/bench/bench.yaml, tasks in configs/bench/probe_tasks.yaml.
"""
