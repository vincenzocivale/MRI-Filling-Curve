"""External foundation models behind one interface: `build(cfg)` from a `configs/fm/*.yaml`."""
from __future__ import annotations

import importlib

from .base import FoundationEncoder

# model name (config `model:`) -> module in this package exposing `build(name, model_args) -> FoundationEncoder`
MODELS = {
    "brainiac": "brainiac", "medicalnet": "medicalnet",
    "brainsegfounder": "brainsegfounder", "brainmvp": "brainmvp", "mome": "mome", "mome_plus": "mome",
    "openmind": "nnssl", "nnfoundation": "nnssl", "amaes": "amaes", "fomo26": "amaes",
    "brainfm": "brainfm", "nnunet": "nnunet",
}


def build(cfg: dict) -> FoundationEncoder:
    """cfg: {model: <name>, model_args: {...}}. Weights are loaded separately via `load_checkpoint`."""
    name = cfg["model"]
    if name not in MODELS:
        raise KeyError(f"Unknown foundation model {name!r}; available: {sorted(MODELS)}")
    return importlib.import_module(f"{__name__}.{MODELS[name]}").build(name, cfg.get("model_args") or {})
