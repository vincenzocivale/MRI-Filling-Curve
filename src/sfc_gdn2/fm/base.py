"""Contract of an external foundation-model (FM) wrapper: image file(s) -> the model's own features.

A wrapper runs INSIDE the model's conda env (see `worker.py`) and imports the ORIGINAL repo code
(`cfg["repo"]`, put on sys.path, never vendored) for all three steps:

- `preprocess(image)`: the repo's own preprocessing, from the NIfTI file(s) on disk, exactly as the repo
  runs it before its encoder (reorientation, resampling, registration / skull stripping when the repo
  ships them, cropping, intensity normalisation). No step of ours in between.
- `features(prepared)`: the repo's own network, checkpoint loader, inference procedure (single crop,
  sliding window, TTA, precision) and the repo's own feature output.
- the result dict: every tensor the repo exposes as a representation (`features`), the name of the one
  the repo itself uses as its embedding (`canonical`), and the geometry needed to map voxel labels onto
  the feature grid (`meta`). Anything derived by us (e.g. a pooling the repo does not define) is listed in
  `derived` so it is never mistaken for the repo's output.

An image is a path (single-modality models) or a {modality: path} dict (multi-modal models; the
accepted modality names are the wrapper's `modalities`).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Union

Image = Union[str, dict[str, str]]  # noqa: UP007 -- runtime alias, some model envs run python 3.9


def add_to_path(*dirs: str | Path) -> None:
    """Make an original repo importable (front of sys.path, once)."""
    for d in reversed([str(Path(d)) for d in dirs]):
        if d not in sys.path:
            sys.path.insert(0, d)


def git_commit(repo: str | Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Wrapper:
    """Subclasses: set `name`, `modalities`, implement `preprocess` and `features`.

    cfg = {model, env, repo, checkpoint, args: {...}}; device = "cuda" | "cpu".
    """

    name = "fm"
    modalities: tuple[str, ...] = ("image",)   # single-modality wrappers take a bare path

    def __init__(self, cfg: dict, device: str):
        self.cfg, self.args, self.device = cfg, dict(cfg.get("args") or {}), device

    def preprocess(self, image: Image) -> Any:
        raise NotImplementedError

    def features(self, prepared: Any) -> dict:
        """-> {"features": {name: tensor}, "canonical": name, "meta": {...}, "derived": [names]}."""
        raise NotImplementedError

    def provenance(self) -> dict:
        return {"repo": self.cfg.get("repo"), "repo_commit": git_commit(self.cfg["repo"]) if self.cfg.get("repo") else None,
                "checkpoint": self.cfg.get("checkpoint"), "python": sys.version.split()[0]}

    def __call__(self, image: Image) -> dict:
        if isinstance(image, dict):
            unknown = set(image) - set(self.modalities)
            if unknown:
                raise ValueError(f"{self.name}: unknown modalities {sorted(unknown)}; accepted {self.modalities}.")
        out = self.features(self.preprocess(image))
        missing = {"features", "canonical", "meta"} - set(out)
        if missing or out["canonical"] not in out["features"]:
            raise RuntimeError(f"{self.name}: wrapper output violates the contract ({sorted(missing)} missing "
                               f"or canonical {out.get('canonical')!r} not in features).")
        out.setdefault("derived", [])
        return out | {"model": self.cfg.get("model"), "provenance": self.provenance()}
