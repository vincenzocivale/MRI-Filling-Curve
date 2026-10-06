"""External foundation models: image (NIfTI) -> the model's own features, each in its original pipeline.

One entrypoint for every model, `extract(config, images, out)` (CLI: `sfc fm configs/fm/<name>.yaml
--images a.nii.gz ... --out dir`). Each model runs in its own conda env (`env:` in the config, spec in
`envs/fm/`) with the original repo's code (`repo:`) for preprocessing, network and feature output;
`base.py` holds the wrapper contract, `worker.py` the in-env driver. Kept import-light: the worker
imports this module inside every model env.
"""
from __future__ import annotations

# config `model:` -> module in this package exposing `build(cfg, device) -> base.Wrapper`
MODELS = {
    "brainiac": "brainiac",
    "brainsegfounder": "brainsegfounder",
    "medicalnet": "medicalnet",
    "brainmvp": "brainmvp",
    "mome": "mome", "mome_plus": "mome",
    "openmind": "nnssl", "nnfoundation": "nnssl",
    "fomo26": "asparagus", "amaes_fomo260k": "asparagus",
    "brainfm": "brainfm",
    "nnunet": "nnunet",
}


def extract(*args, **kwargs):
    from .api import extract as _extract
    return _extract(*args, **kwargs)
