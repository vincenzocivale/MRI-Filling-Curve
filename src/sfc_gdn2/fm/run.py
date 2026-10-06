"""`sfc fm`: the one evaluation path for every external foundation model.

For each (model config x probe config): build the encoder from the registry, load the checkpoint in
the model's original format, and run the *same* `LinearProbe` we use for our own encoder (same
manifests/splits, same pooling candidates, same L2 grid and selection rule, same bootstrap CI).
Groups reported: `pretrained` (the checkpoint) and `init` (same architecture, seeded random init),
next to the `raw`/`position` baselines from `sfc probe --raw`. Results land in
`<output_root>/fm-<model>-<hash>/probe-<label>-<hash>/`, where `sfc summarize <output_root>` finds them.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..io import RunDir, load_yaml, seed_all
from ..probe import LinearProbe
from . import build
from .base import FoundationEncoder
from .extract import fm_extractor


def make_encoder(model_cfg: dict, pretrained: bool, seed: int, device: torch.device) -> FoundationEncoder:
    seed_all(seed)  # same seed -> `init` is reproducible and identical across probes
    enc = build(model_cfg)
    if pretrained:
        report = enc.load_checkpoint(model_cfg["checkpoint"])
        print(f"[fm] {model_cfg['model']}: loaded {report['loaded']} tensors "
              f"({len(report['ignored'])} ignored) from {model_cfg['checkpoint']}", flush=True)
    return enc.to(device).eval().requires_grad_(False)


def probe_fm(eval_cfg: dict, model_cfg: dict, probe_cfg: dict, run: RunDir) -> dict:
    device = torch.device("cuda")
    seed = int(probe_cfg["seed"])
    probe = LinearProbe(probe_cfg, eval_cfg["data"], seen_subjects=set())  # nothing of ours was pretrained on
    level = probe.task.level
    groups = {"init": {"init": fm_extractor(make_encoder(model_cfg, False, seed, device), level)}}
    if model_cfg.get("checkpoint"):
        groups["pretrained"] = {"checkpoint": fm_extractor(make_encoder(model_cfg, True, seed, device), level)}
    else:
        print(f"[fm] {model_cfg['model']}: no checkpoint configured, reporting the random init only.", flush=True)
    out = RunDir.create(run.path, probe_cfg, prefix=f"probe-{probe_cfg['label']}")
    result = probe.run(groups, out) | {"curve": "none", "objective": f"fm:{model_cfg['model']}"}
    out.save_json(result, "metrics.json")
    return result


def evaluate(eval_cfg: dict, model_paths: list[str], probe_paths: list[str] | None = None) -> None:
    probes = probe_paths or eval_cfg["probes"]
    for mp in model_paths:
        model_cfg = load_yaml(mp)
        run = RunDir.create(eval_cfg["output_root"], {"model": model_cfg, "data": eval_cfg["data"]},
                            prefix=f"fm-{model_cfg['model']}")
        for pp in probes:
            probe_fm(eval_cfg, model_cfg, load_yaml(pp), run)
        print(Path(run.path))
