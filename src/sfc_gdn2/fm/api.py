"""The single entrypoint: `extract(config, images, out)` -> one `<out>/<id>.pt` per image.

The config (`configs/fm/<name>.yaml`) names the model's conda env; the work is done by `worker.py`
running under that env's python, with this package on PYTHONPATH (nothing of ours is installed in
the model envs). `${VAR}` in config paths are expanded from the environment (see configs/fm/README.md).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[2]  # .../src


def expand(x):
    if isinstance(x, str):
        out = os.path.expandvars(x)
        if re.search(r"\$\{?\w+", out):
            raise KeyError(f"unset environment variable in config value {x!r}")
        return out
    if isinstance(x, dict):
        return {k: expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [expand(v) for v in x]
    return x


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = expand(yaml.safe_load(f))
    for key in ("model", "env", "repo"):
        if not cfg.get(key):
            raise KeyError(f"{path}: `{key}` is required")
    return cfg


def image_id(image) -> str:
    return Path(image).name.removesuffix(".gz").removesuffix(".nii")


def run_in_env(env: str, module: str, job: dict) -> None:
    """`<env>/bin/python -m <module> <job.json>`, with this package on PYTHONPATH."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
    pythonpath = os.pathsep.join([str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])])
    try:
        subprocess.run([str(Path(env) / "bin" / "python"), "-m", module, f.name], check=True,
                       env=os.environ | {"PYTHONPATH": pythonpath, "PYTHONNOUSERSITE": "1"})
    finally:
        os.unlink(f.name)


def extract(config, images: list, out, device: str = "cuda", overwrite: bool = False,
            ids: list[str] | None = None, workers: int = 0) -> list[Path]:
    """images: NIfTI paths. Returns the written files.

    `config` / `out` may be lists (same length): a group of models sharing the first one's preprocessing
    (same env required). See worker.py."""
    cfgs = [c if isinstance(c, dict) else load_config(c) for c in (config if isinstance(config, list) else [config])]
    outs = [Path(o) for o in (out if isinstance(out, list) else [out])]
    if len(outs) != len(cfgs) or len({c["env"] for c in cfgs}) != 1:
        raise ValueError("a model group needs one `out` per config and a single env")
    ids = ids or [image_id(im) for im in images]
    if len(set(ids)) != len(ids):
        raise ValueError("image ids are not unique; pass `ids`.")
    items = [{"id": i, "image": str(im)} for i, im in zip(ids, images)]
    run_in_env(cfgs[0]["env"], "sfc_gdn2.fm.worker",
               {"cfgs": cfgs, "items": items, "outs": [str(o) for o in outs], "device": device,
                "overwrite": overwrite, "workers": workers})
    return [o / f"{i}.pt" for o in outs for i in ids]
