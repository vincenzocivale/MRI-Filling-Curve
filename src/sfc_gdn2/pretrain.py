from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .data.dataset import loader
from .io import RunDir, seed_all
from .lejepa import LeJEPA


def pretrain_rows(cfg: dict) -> pd.DataFrame:
    """Rows of every manifest in `data.manifests` whose split is in `data.splits`."""
    d = cfg["data"]
    df = pd.concat([pd.read_csv(m) for m in d["manifests"]], ignore_index=True)
    df = df[df["split"].isin(d["splits"])]
    if df.empty:
        raise RuntimeError(f"No rows in {d['manifests']} with split in {d['splits']}.")
    return df.reset_index(drop=True)


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices only. Biases, norms, the mask token and GDN-2's decay-gate parameters
    (A_log, dt_bias: flagged `_no_weight_decay` upstream) are left undecayed."""
    decay, keep = [], []
    for p in model.parameters():
        (keep if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": keep, "weight_decay": 0.0}]


class Pretrainer:
    """LeJEPA pretraining on every row of the pretraining split: no held-out set; checkpoints are
    selected downstream on the probe's val split. All randomness derives from cfg['seed'].
    Under torchrun: one process per GPU, each on a disjoint shard of the rows with its own
    `batch_size` (effective batch = world size x batch_size); only rank 0 writes."""

    def __init__(self, cfg: dict, run: RunDir):
        self.cfg, self.run, t = cfg, run, cfg["train"]
        self.seed = int(cfg["seed"])
        seed_all(self.seed)
        self.rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(self.device)
        rows = pretrain_rows(cfg)
        if self.rank == 0:
            run.save_table(rows[["dataset", "subject", "sample_id"]], "split.csv")
        self.loader = loader(rows.iloc[self.rank::world], cfg["data"], t["batch_size"], train=True,
                             seed=self.seed + self.rank)
        self.net = LeJEPA.from_config(cfg).to(self.device)
        self.model = DistributedDataParallel(self.net, device_ids=[self.device.index]) if world > 1 else self.net
        self.opt = torch.optim.AdamW(param_groups(self.net, t["weight_decay"]), lr=t["lr"], fused=True)
        warmup, steps = t.get("warmup_steps", 0), t["steps"]
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / max(warmup, 1))
                                                       * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
        self.log = print if self.rank == 0 else (lambda *a, **k: None)
        self.log(f"[lejepa] world={world} curves={cfg['objective']['view_curves']} volumes={len(rows)} "
              f"params={sum(p.numel() for p in self.net.parameters()) / 1e6:.1f}M", flush=True)

    def _batches(self):
        while True:
            yield from self.loader

    def fit(self) -> None:
        t = self.cfg["train"]
        gen = torch.Generator(self.device).manual_seed(self.seed + 123 + self.rank)
        save = self.rank == 0
        clip = float(t.get("grad_clip", float("inf")))
        hist, skipped, t0 = [], 0, time.time()
        if save:
            torch.save(self.net.encoder.state_dict(), self.run.path / "encoder_step000000.pt")  # init baseline
        self.model.train()
        for step, batch in zip(range(1, t["steps"] + 1), self._batches()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(batch["patches"].to(self.device, non_blocking=True).float(), gen)
            self.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            if torch.isfinite(gnorm):  # a non-finite step would poison the weights and Adam's moments
                self.opt.step()
            else:
                skipped += 1
            self.sched.step()
            if step % t.get("log_every", 50) == 0 or step == t["steps"]:
                hist.append({"step": step, **{k: v.item() for k, v in out.items()}, "grad_norm": gnorm.item(),
                             "skipped": skipped, "seconds": time.time() - t0})
                self.log("[lejepa] " + " ".join(f"{k}={v:.4g}" for k, v in hist[-1].items()), flush=True)
            if save and (step % t["save_every"] == 0 or step == t["steps"]):
                torch.save(self.net.encoder.state_dict(), self.run.path / f"encoder_step{step:06d}.pt")
        if not save:
            return
        self.run.save_table(pd.DataFrame(hist), "history.csv")
        self.run.save_json({**hist[-1],
                            "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2 ** 20},
                           "metrics.json")
        torch.save(self.net.state_dict(), self.run.path / "model.pt")  # encoder + objective heads


def pretrain(cfg: dict) -> Path | None:
    """Returns the run directory on rank 0 (None on other ranks)."""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
    run = RunDir.create(cfg["output_root"], cfg, prefix="lejepa")
    trainer = Pretrainer(cfg, run)
    trainer.fit()
    if dist.is_initialized():
        dist.destroy_process_group()
    return run.path if trainer.rank == 0 else None
