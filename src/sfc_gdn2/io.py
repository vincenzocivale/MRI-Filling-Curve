from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stable_id(obj: dict, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "host": platform.node(), "pid": os.getpid()}


@dataclass
class RunDir:
    """`<root>/<prefix>-<sha1(cfg)>/`: identical configs land in the same directory."""

    path: Path

    @classmethod
    def create(cls, root: str | Path, cfg: dict, prefix: str) -> RunDir:
        run = cls(Path(root) / f"{prefix}-{stable_id(cfg)}")
        run.path.mkdir(parents=True, exist_ok=True)
        with open(run.path / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        run.save_json(environment(), "environment.json")
        return run

    @property
    def config(self) -> dict:
        return load_yaml(self.path / "config.yaml")

    def save_json(self, obj, name: str) -> None:
        with open(self.path / name, "w") as f:
            json.dump(obj, f, indent=2)

    def save_table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.path / name, index=False)
