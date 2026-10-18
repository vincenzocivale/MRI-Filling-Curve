from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .data.dataset import loader, session_ids, to_device
from .io import run_dir, seed_all
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
    the plateau: resume it with train.steps = its step + decay_steps. `warmup_from`: the step the warmup starts at
    (a run continued from a decayed checkpoint warms up again from there)."""
    warm = min(1.0, max(0.0, step + 1 - t.get("warmup_from", 0)) / max(t.get("warmup_steps", 0), 1))
    if t.get("schedule", "cosine") == "cosine":
        return warm * 0.5 * (1 + math.cos(math.pi * min(step / t["steps"], 1.0)))
    left = (t["steps"] - step) / max(t["decay_steps"], 1)   # 1 at the decay's start, 0 at the end
    return warm * (1.0 if left >= 1 else 1 - math.sqrt(max(0.0, 1 - left)))


def param_groups(model: nn.Module, weight_decay: float, muon: bool = False) -> list[dict]:
    """Decay matrices only. Biases, norms, the mask token and GDN-2's decay-gate parameters
    (A_log, dt_bias: flagged `_no_weight_decay` upstream) are left undecayed. muon: the encoder blocks' weight
    matrices in a group of their own (`MuonAdamW`)."""
    groups = {"muon": [], "decay": [], "keep": []}
    for name, p in model.named_parameters():
        g = "keep" if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else "decay"
        groups["muon" if muon and g == "decay" and p.ndim == 2 and ".blocks." in f".{name}" else g].append(p)
    return [{"params": groups["muon"], "weight_decay": weight_decay, "muon": True},
            {"params": groups["decay"], "weight_decay": weight_decay}, {"params": groups["keep"], "weight_decay": 0.0}]


def _orthogonalize(g: torch.Tensor, steps: int) -> torch.Tensor:
    """~U V^T of g's SVD by the quintic Newton-Schulz iteration of Muon (Jordan et al. 2024), in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    if tall := x.shape[0] > x.shape[1]:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return x.T if tall else x


class MuonAdamW(torch.optim.Optimizer):
    """Muon on the groups flagged `muon`, AdamW on the rest. Muon: Nesterov momentum orthogonalised by Newton-Schulz,
    scaled by 0.2 sqrt(max(fan_out, fan_in)) so its RMS matches AdamW's (Moonlight, Liu et al. 2025, arXiv
    2502.16982): AdamW's lr, schedule and decoupled weight decay carry over. Same rank-identical update under DDP."""

    def __init__(self, groups: list[dict], lr: float, betas=(0.9, 0.999), eps: float = 1e-8, momentum: float = 0.95,
                 ns_steps: int = 5):
        super().__init__(groups, {"lr": lr, "betas": betas, "eps": eps, "momentum": momentum, "ns_steps": ns_steps,
                                  "muon": False, "weight_decay": 0.0})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["muon"]:
                    m = st.setdefault("m", torch.zeros_like(p))
                    m.mul_(g["momentum"]).add_(p.grad)
                    u = _orthogonalize(p.grad.add(m, alpha=g["momentum"]), g["ns_steps"])
                    p.mul_(1 - lr * wd).add_(u.to(p.dtype), alpha=-lr * 0.2 * max(p.shape) ** 0.5)
                    continue
                b1, b2 = g["betas"]
                st["t"] = t = st.get("t", 0) + 1
                m, v = st.setdefault("m", torch.zeros_like(p)), st.setdefault("v", torch.zeros_like(p))
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - lr * wd).addcdiv_(m, (v / (1 - b2 ** t)).sqrt_().add_(g["eps"]), value=-lr / (1 - b1 ** t))


class Pretrainer:
    """LeJEPA pretraining on every row of the pretraining split: no held-out set; checkpoints are
    selected downstream on the probe's val split. All randomness derives from cfg['seed'].
    Under torchrun: one process per GPU, each taking its share of one batch stream with its own
    `batch_size` (effective batch = world size x batch_size); only rank 0 writes.
    Every `ckpt_every` steps `ckpt.pt` holds the whole state (model with heads, optimizer, scheduler, batches of
    the stream consumed; the objective's RNG is reseeded from (seed, step, rank) every step) and `history.csv` is
    rewritten. A run whose directory has a ckpt.pt resumes from it (same config = same directory, so a chain of
    identical jobs continues one run), else from `train.resume_from` (another run's ckpt.pt, e.g. a WSD decay
    branch), on any number of GPUs: the stream continues where it stopped (exactly as before on the same number;
    the global batch, and so the effective lr per scan, follows the GPUs)."""

    def __init__(self, cfg: dict, run: Path):
        self.cfg, self.run, t = cfg, run, cfg["train"]
        self.seed = int(cfg["seed"])
        seed_all(self.seed)
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        world = self.world
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(self.device)
        # fp32 matmuls on tensor cores (TF32, 10-bit mantissa): mostly the patch embedding of native voxels and its
        # gradient, ~30% of the step on the SIMT cores. Its rounding (~5e-4 relative) is ~40x below the training
        # noise augmentation (objective.noise); SIGReg keeps full fp32; extraction (benchmark) stays fp32.
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(t.get("tf32", True))
        rows = pretrain_rows(cfg)
        if self.rank == 0:
            rows[["dataset", "subject", "sample_id"]].to_csv(run / "split.csv", index=False)
        ses = session_ids(rows)  # the objective's session level (scans of one session batched together)
        self.loader = loader(rows, cfg["data"], t["batch_size"], ses, seed=self.seed, rank=self.rank, world=world)
        self.net = LeJEPA.from_config(cfg).to(self.device)
        self.session = torch.tensor(ses)
        self.model = DistributedDataParallel(self.net, device_ids=[self.device.index]) if world > 1 else self.net
        muon = t.get("optimizer", "adamw") == "muon"
        groups = param_groups(self.net, t["weight_decay"], muon)
        self.opt = MuonAdamW(groups, lr=t["lr"]) if muon else torch.optim.AdamW(
            [{k: v for k, v in g.items() if k != "muon"} for g in groups], lr=t["lr"], fused=True)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, t))
        self.gen = torch.Generator(self.device).manual_seed(self.seed + 123 + self.rank)
        self.log = print if self.rank == 0 else (lambda *a, **k: None)
        self.log(f"[lejepa] world={world} curves={cfg['objective']['view_curves']} volumes={len(rows)} "
              f"params={sum(p.numel() for p in self.net.parameters()) / 1e6:.1f}M", flush=True)
        self.step, self.consumed, self.skipped, self.seconds, self.hist = 0, 0, 0, 0.0, []
        ckpt = run / "ckpt.pt"
        src = ckpt if ckpt.exists() else t.get("resume_from")
        if src:
            self._restore(Path(src))
        elif t.get("init_from"):  # weights only (new modules keep their init); fresh optimizer and schedule from step 0
            r = self.net.load_state_dict(torch.load(t["init_from"], map_location="cpu"), strict=False)
            self.log(f"[lejepa] init from {t['init_from']}: new {r.missing_keys}, dropped {r.unexpected_keys}", flush=True)

    def _restore(self, path: Path) -> None:
        c = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(c["model"])
        self.opt.load_state_dict(c["opt"])
        self.sched.load_state_dict(c["sched"])
        t = self.cfg["train"]  # the lr of this config's schedule, not the source's (a run that ended its decay saved 0)
        self.sched.base_lrs = [t["lr"]] * len(self.opt.param_groups)
        for g in self.opt.param_groups:
            g["lr"] = t["lr"] * lr_factor(self.step, t)
        self.step, self.consumed, self.skipped, self.seconds = c["step"], c["consumed"], c["skipped"], c["seconds"]
        self.loader.batch_sampler.start = self.consumed
        try:   # a job killed before its first log leaves no (or an empty) history
            h = pd.read_csv(path.parent / "history.csv")
            self.hist = h[h["step"] <= self.step].to_dict("records")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        self.log(f"[lejepa] resumed {path} at step {self.step} ({c['world']} -> {self.world} GPUs)", flush=True)

    def _checkpoint(self) -> None:
        """Rank 0; written to a temporary file, then renamed."""
        tmp = self.run / "ckpt.pt.tmp"
        torch.save({"step": self.step, "consumed": self.consumed, "world": self.world, "skipped": self.skipped,
                    "seconds": time.time() - self.t0, "model": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "sched": self.sched.state_dict()}, tmp)
        os.replace(tmp, self.run / "ckpt.pt")
        pd.DataFrame(self.hist).to_csv(self.run / "history.csv", index=False)

    def fit(self) -> None:
        t = self.cfg["train"]
        gen, hist = self.gen, self.hist
        save = self.rank == 0
        clip = float(t.get("grad_clip", float("inf")))
        self.t0 = t0 = time.time() - self.seconds
        batches, wait, patches, t_log = iter(self.loader), 0.0, 0, time.time()  # endless; wait share, patches/s
        if save and self.step == 0:
            torch.save(self.net.encoder.state_dict(), self.run / "encoder_step000000.pt")  # init baseline
        if self.step >= t["steps"]:
            self.log(f"[lejepa] already at step {self.step} of {t['steps']}", flush=True)
            return
        self.model.train()
        prof, first = os.environ.get("SFC_PROFILE"), self.step + 1  # a dir: steps first+5..first+7 profiled, then stop
        for step in range(self.step + 1, t["steps"] + 1):
            if prof and step == first + 5 and save:
                torch.cuda.memory._record_memory_history(max_entries=500_000)
                p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
                p.__enter__()
            t_w = time.time()
            batch = next(batches)   # the previous step already synced (isfinite(gnorm)): this wait is a GPU stall
            wait += time.time() - t_w
            patches += int(batch["grid"].prod(1).sum())
            gen.manual_seed((self.seed * 1_000_003 + step) * 4096 + self.rank)  # resumable on any number of GPUs
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vols = to_device(batch, self.device) | {"session": self.session[batch["index"]].to(self.device)}
                out = self.model(vols, gen)
            self.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            if torch.isfinite(gnorm):  # a non-finite step would poison the weights and Adam's moments
                self.opt.step()
            else:
                self.skipped += 1
            self.sched.step()
            self.step, self.consumed = step, self.consumed + self.world
            if step % t.get("log_every", 50) == 0 or step == t["steps"]:
                dt = time.time() - t_log
                hist.append({"step": step, **{k: v.item() for k, v in out.items()}, "grad_norm": gnorm.item(),
                             "skipped": self.skipped, "lr": self.sched.get_last_lr()[0],
                             "seconds": time.time() - t0, "data_wait": wait / dt, "patches_per_s": patches / dt,
                             "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / 2 ** 20})
                wait, patches, t_log = 0.0, 0, time.time()
                self.log("[lejepa] " + " ".join(f"{k}={v:.4g}" for k, v in hist[-1].items()), flush=True)
            if prof and step == first + 7:
                if save:
                    torch.cuda.synchronize()
                    p.__exit__(None, None, None)
                    Path(prof).mkdir(parents=True, exist_ok=True)
                    p.export_chrome_trace(f"{prof}/trace.json")
                    ka = p.key_averages()
                    (Path(prof) / "ops.txt").write_text(ka.table(sort_by="cuda_time_total", row_limit=80) + "\n"
                                                       + ka.table(sort_by="self_cpu_time_total", row_limit=40))
                    torch.cuda.memory._dump_snapshot(f"{prof}/memory.pickle")
                return
            if save and (step % t["save_every"] == 0 or step == t["steps"]):
                torch.save(self.net.encoder.state_dict(), self.run / f"encoder_step{step:06d}.pt")
            if save and (step % t.get("ckpt_every", 500) == 0 or step == t["steps"]):
                self._checkpoint()
        if not save:
            return
        with open(self.run / "metrics.json", "w") as f:  # the last step: logged (hist[-1]), history.csv written
            json.dump(hist[-1], f, indent=2)
        torch.save(self.net.state_dict(), self.run / "model.pt")  # encoder + objective heads


def pretrain(cfg: dict) -> Path | None:
    """Returns the run directory on rank 0 (None on other ranks)."""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
    run = run_dir(cfg["output_root"], cfg, prefix="lejepa")
    trainer = Pretrainer(cfg, run)
    trainer.fit()
    if dist.is_initialized():
        dist.destroy_process_group()
    return run if trainer.rank == 0 else None
