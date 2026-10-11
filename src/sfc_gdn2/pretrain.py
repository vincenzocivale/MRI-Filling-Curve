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

from .data.dataset import loader, to_device
from .io import RunDir, seed_all
from .lejepa import LeJEPA

# grids differ ~30x in size between a brain and a whole body: with the default allocator the cache
# fragmented to the card's limit and steps slowed 10x (4 shared GPUs, 0.6 -> 6 s/step)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def pretrain_rows(cfg: dict) -> pd.DataFrame:
    """Rows of every manifest in `data.manifests` whose split is in `data.splits`."""
    d = cfg["data"]
    df = pd.concat([pd.read_csv(m) for m in d["manifests"]], ignore_index=True)
    df = df[df["split"].isin(d["splits"])]
    if df.empty:
        raise RuntimeError(f"No rows in {d['manifests']} with split in {d['splits']}.")
    return df.reset_index(drop=True)


def lr_factor(step: int, t: dict) -> float:
    """Linear warmup, then `schedule`: cosine to 0 at `steps`, or wsd (warmup-stable-decay): constant until the
    last `decay_steps`, then 1 - sqrt to 0 (Hägele et al. 2024). WSD's decay can start from any checkpoint of
    the plateau: resume it with train.steps = its step + decay_steps."""
    warm = min(1.0, (step + 1) / max(t.get("warmup_steps", 0), 1))
    if t.get("schedule", "cosine") == "cosine":
        return warm * 0.5 * (1 + math.cos(math.pi * min(step / t["steps"], 1.0)))
    left = (t["steps"] - step) / max(t["decay_steps"], 1)   # 1 at the decay's start, 0 at the end
    return warm * (1.0 if left >= 1 else 1 - math.sqrt(max(0.0, 1 - left)))


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
    `batch_size` (effective batch = world size x batch_size); only rank 0 writes.
    Every `ckpt_every` steps `ckpt.pt` holds the whole state (model with heads, optimizer, scheduler, every rank's
    RNG; the sampler is placed by the step) and `history.csv` is rewritten: a run whose directory has a ckpt.pt
    resumes from it (same config = same directory, so a chain of identical jobs continues one run), else from
    `train.resume_from` (another run's ckpt.pt, e.g. a WSD decay branch)."""

    def __init__(self, cfg: dict, run: RunDir):
        self.cfg, self.run, t = cfg, run, cfg["train"]
        self.seed = int(cfg["seed"])
        seed_all(self.seed)
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        world = self.world
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
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, t))
        self.gen = torch.Generator(self.device).manual_seed(self.seed + 123 + self.rank)
        self.log = print if self.rank == 0 else (lambda *a, **k: None)
        self.log(f"[lejepa] world={world} curves={cfg['objective']['view_curves']} volumes={len(rows)} "
              f"params={sum(p.numel() for p in self.net.parameters()) / 1e6:.1f}M", flush=True)
        self.step, self.skipped, self.seconds, self.hist = 0, 0, 0.0, []
        ckpt = run.path / "ckpt.pt"
        src = ckpt if ckpt.exists() else t.get("resume_from")
        if src:
            self._restore(Path(src))

    def _restore(self, path: Path) -> None:
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c["world"] != self.world:
            raise RuntimeError(f"{path}: saved with {c['world']} GPUs, running on {self.world} (rank shards differ)")
        self.net.load_state_dict(c["model"])
        self.opt.load_state_dict(c["opt"])
        self.sched.load_state_dict(c["sched"])
        r = c["rng"][self.rank]
        torch.set_rng_state(r["cpu"])
        torch.cuda.set_rng_state(r["cuda"], self.device)
        self.gen.set_state(r["gen"])
        self.step, self.skipped, self.seconds = c["step"], c["skipped"], c["seconds"]
        self.loader.batch_sampler.seek(self.step)
        if (path.parent / "history.csv").exists():
            h = pd.read_csv(path.parent / "history.csv")
            self.hist = h[h["step"] <= self.step].to_dict("records")
        self.log(f"[lejepa] resumed {path} at step {self.step}", flush=True)

    def _checkpoint(self) -> None:
        """Collective (every rank's RNG gathered on rank 0); written to a temporary file, then renamed."""
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device),
               "gen": self.gen.get_state()}
        rngs = [None] * self.world
        if dist.is_initialized():
            dist.all_gather_object(rngs, rng)
        else:
            rngs = [rng]
        if self.rank != 0:
            return
        tmp = self.run.path / "ckpt.pt.tmp"
        torch.save({"step": self.step, "world": self.world, "skipped": self.skipped,
                    "seconds": self.seconds + time.time() - self.t0, "model": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "rng": rngs}, tmp)
        os.replace(tmp, self.run.path / "ckpt.pt")
        self.run.save_table(pd.DataFrame(self.hist), "history.csv")

    def _batches(self):
        while True:
            yield from self.loader

    def fit(self) -> None:
        t = self.cfg["train"]
        gen, hist = self.gen, self.hist
        save = self.rank == 0
        clip = float(t.get("grad_clip", float("inf")))
        self.t0 = t0 = time.time() - self.seconds
        batches, wait, patches, t_log = self._batches(), 0.0, 0, time.time()  # wait share, patches/s
        if save and self.step == 0:
            torch.save(self.net.encoder.state_dict(), self.run.path / "encoder_step000000.pt")  # init baseline
        if self.step >= t["steps"]:
            self.log(f"[lejepa] already at step {self.step} of {t['steps']}", flush=True)
            return
        self.model.train()
        for step in range(self.step + 1, t["steps"] + 1):
            t_w = time.time()
            batch = next(batches)   # the previous step already synced (isfinite(gnorm)): this wait is a GPU stall
            wait += time.time() - t_w
            patches += int(batch["grid"].prod(1).sum())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(to_device(batch, self.device), gen)
            self.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            if torch.isfinite(gnorm):  # a non-finite step would poison the weights and Adam's moments
                self.opt.step()
            else:
                self.skipped += 1
            self.sched.step()
            self.step = step
            if step % t.get("log_every", 50) == 0 or step == t["steps"]:
                dt = time.time() - t_log
                hist.append({"step": step, **{k: v.item() for k, v in out.items()}, "grad_norm": gnorm.item(),
                             "skipped": self.skipped, "lr": self.sched.get_last_lr()[0],
                             "seconds": time.time() - t0, "data_wait": wait / dt, "patches_per_s": patches / dt,
                             "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2 ** 20})
                wait, patches, t_log = 0.0, 0, time.time()
                self.log("[lejepa] " + " ".join(f"{k}={v:.4g}" for k, v in hist[-1].items()), flush=True)
            if save and (step % t["save_every"] == 0 or step == t["steps"]):
                torch.save(self.net.encoder.state_dict(), self.run.path / f"encoder_step{step:06d}.pt")
            if step % t.get("ckpt_every", 500) == 0 or step == t["steps"]:
                self._checkpoint()
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
