"""Shared by the parity scripts (their directory is on sys.path when they run as scripts)."""
from __future__ import annotations

from pathlib import Path

import torch

SAMPLES = Path("/leonardo_scratch/large/userexternal/fcorrent/fm_check/samples")


def check(name: str, a, b) -> None:
    """Prints the max abs difference and asserts bitwise equality (as float32)."""
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    print(f"  {name:<48} {tuple(a.shape)} max|diff|={(a - b).abs().max().item():.3e}", flush=True)
    assert torch.equal(a, b), f"{name}: not bitwise equal"
