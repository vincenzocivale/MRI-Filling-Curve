from __future__ import annotations

import json
import math
import os
import re
import shutil
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


def repatch(batch: dict, mm: float, max_patches: int | None = None) -> dict:
    """The batch's scans cut into patches of `mm` (data.patch_mm_choices, 18/10: one model at every resolution).
    The objective's geometry stays in tokens (views, masks, cell levels), so its physical size scales with the patch,
    as FlexiViT's; the kernel takes the patch size (`KernelEmbed.patch_mm`) and, with basis bspline, the cell's own
    coordinate."""
    from .data.volume import patch_dims, patch_grid
    k, grid, mms = [], [], []
    for s, v in zip(batch["spacing"].tolist(), batch["volumes"]):
        m = mm   # max_patches (data.max_patches_per_scan, 22/10): a scan whose grid would pass it (whole-body at 4 mm:
        while True:   # ~6e6 patches; the objective's [views x patches] tables of the batch's largest grid: OOM) gets
            kk = patch_dims(s, m)   # the smallest larger cells that fit
            g = patch_grid(tuple(v.shape[:3]), kk)
            if max_patches is None or int(g.prod()) <= max_patches:
                break
            m *= 1.25
        k.append(kk), grid.append(g), mms.append(m)
    return batch | {"k": torch.tensor(k), "grid": torch.stack(grid), "patch_mm": torch.tensor(mms)}


def fp32_forward(f):
    def g(x):
        with torch.autocast("cuda", enabled=False):
            return f(x.float())
    return g


def lr_factor(step: int, t: dict) -> float:
    """Linear warmup, then `schedule`: cosine to 0 at `steps`, or wsd (warmup-stable-decay): constant until the
    last `decay_steps`, then 1 - sqrt to 0 (Hägele et al. 2024). WSD's decay can start from any checkpoint of
    the plateau: resume it with train.steps = its step + decay_steps. `warmup_from`: the step the warmup starts at
    (a run continued from a decayed checkpoint warms up again from there)."""
    warm = min(1.0, max(0.0, step + 1 - t.get("warmup_from", 0)) / max(t.get("warmup_steps", 0), 1))
    if t.get("schedule") == "plateau":   # train.grain_plateau sets the lr (GrainPlateau.scale)
        return warm
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


class GrainPlateau:
    """train.grain_plateau (26/10, user: GrainMixScheduler, then ReduceLROnGrainPlateau): the grains of each step
    (data.patch_mm_choices) grow by stages, each stage running until the loss of its finest grain stops improving;
    after the last stage the lr is cut by `lr_factor` on each plateau of the whole mix, `lr_cuts` times, then the run
    ends. Plateau: every `every` steps, the mean loss of the watched steps over the last `window` steps against the
    same mean `gap` steps before improves by less than `rate` per 1000 steps, `patience` checks in a row (both windows
    inside the stage: >= window + gap steps per stage). The loss is the step's mean over the GPUs (the same decision on
    every rank); a run resumed from a checkpoint without this state replays its history.csv (the logged steps).
    A stage is a list of grains, or (28/10, sub-token targets in cascade) a dict {grains, set: {LeJEPA attribute: value},
    watch: a loss term of out}: the sets of every stage up to the current one hold (rec_contrast 0.2, rec_cells 8, ...)
    and the plateau is read on the watched term over every step (watch "loss", the default: the finest grain's steps).
    fresh: true (a new cascade from a checkpoint of another schedule): no state or history taken over, the first
    stage starts at the resumed step."""

    def __init__(self, g: dict, hist: list[dict], patch_mm: float, start: int = 0):
        self.stages = [{**st, "grains": [float(m) for m in st["grains"]]} if isinstance(st, dict)
                       else {"grains": [float(m) for m in st]} for st in g["stages"]]
        self.watched = sorted({st.get("watch", "loss") for st in self.stages} | {"loss"})
        self.window, self.gap, self.every = int(g.get("window", 250)), int(g.get("gap", 500)), int(g.get("every", 250))
        self.rate, self.patience = float(g.get("rate", 0.03)), int(g.get("patience", 2))
        self.min_samples = int(g.get("min_samples", 5))   # per window (a smoke's toy windows: 1)
        self.factor, self.max_cuts = float(g.get("lr_factor", 0.3)), int(g.get("lr_cuts", 3))
        self.stage, self.cuts, self.start, self.hits, self.done = int(g.get("start_stage", 0)), 0, start, 0, False
        self.rec, self.events = [], []
        for h in hist:   # replayed: the checks of the logged steps give the state the rule had on them
            self.update(int(h["step"]), float(h.get("mm", patch_mm)),
                        {k: float(h[k]) for k in self.watched if k in h and h[k] == h[k]})

    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("stage", "cuts", "start", "hits", "done", "rec", "events")}

    def load_state_dict(self, d: dict) -> None:
        self.__dict__.update(d)

    @property
    def choices(self) -> list[float]:
        return self.stages[self.stage]["grains"]

    def sets(self) -> dict:
        """The LeJEPA attributes the stages so far set (later stages win)."""
        return {k: v for st in self.stages[: self.stage + 1] for k, v in (st.get("set") or {}).items()}

    @property
    def scale(self) -> float:
        return self.factor ** self.cuts

    def update(self, step: int, mm: float, terms) -> str | None:
        """Records the step (terms: {loss term: the step's value}, or the loss alone); returns the event of a plateau
        (next stage, lr cut, done) or None."""
        terms = terms if isinstance(terms, dict) else {"loss": float(terms)}
        self.rec.append((step, mm, terms))
        if self.done or step % self.every or step - self.start < self.window + self.gap:
            return None
        key = self.stages[self.stage].get("watch", "loss") if self.cuts == 0 else "loss"
        # grain stages: the finest grain's steps (a watched term: every step); lr cuts: the whole loss, every step
        watch = min(self.choices) if self.cuts == 0 and key == "loss" else None

        def mean(a, b):
            v = [t[key] for s, m, t in self.rec if a < s <= b and key in t and (watch is None or m == watch)]
            return sum(v) / len(v) if len(v) >= self.min_samples else None
        now, before = mean(step - self.window, step), mean(step - self.gap - self.window, step - self.gap)
        if now is None or before is None:
            return None
        rate = (before - now) / before * 1000 / self.gap
        self.hits = self.hits + 1 if rate < self.rate else 0
        if self.hits < self.patience:
            return None
        self.hits, self.start = 0, step
        if self.cuts == 0 and self.stage < len(self.stages) - 1:
            self.stage += 1
            st = self.stages[self.stage]
            ev = f"stage {self.stage}: grains {self.choices}" + (f", set {st['set']}, watch {st.get('watch', 'loss')}"
                                                                    if st.get("set") else "")
        elif self.cuts < self.max_cuts:
            self.cuts += 1
            ev = f"lr cut {self.cuts}: x{self.scale:.3g}"
        else:
            self.done, ev = True, "done"
        self.events.append({"step": step, "event": ev, "rate": rate, "loss": now})
        return f"{ev} (loss {now:.4f}, {rate * 100:.2f}% per 1000 steps)"


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
        if gb := t.get("global_batch"):   # 26/10: the same run on 2 / 4 / 8 GPUs (scans and patches per step fixed)
            assert gb % world == 0, f"train.global_batch {gb} on {world} GPUs"
            t["batch_size"] = gb // world
            cfg["data"]["patches_per_batch"] = int(cfg["data"]["global_patches"]) // world
        rows = pretrain_rows(cfg)
        if self.rank == 0:
            rows[["dataset", "subject", "sample_id"]].to_csv(run / "split.csv", index=False)
        ses = session_ids(rows)  # the objective's session level (scans of one session batched together)
        self.loader = loader(rows, cfg["data"], t["batch_size"], ses, seed=self.seed, rank=self.rank, world=world)
        self.net = LeJEPA.from_config(cfg).to(self.device)
        if t.get("sync_bn") and world > 1:   # the projector's BatchNorm on the global batch (as SIGReg), whatever the
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)   # GPUs: same parameters, same state dict
            for m in self.net.modules():   # in fp32: under bf16 autocast SyncBatchNorm's kernel failed ("Expected
                if isinstance(m, nn.SyncBatchNorm):   # counts to have type BFloat16 but got Float", 21/10)
                    m.forward = fp32_forward(m.forward)
        self.session = torch.tensor(ses)
        self.model = DistributedDataParallel(self.net, device_ids=[self.device.index]) if world > 1 else self.net
        muon = t.get("optimizer", "adamw") == "muon"
        groups = param_groups(self.net, t["weight_decay"], muon)
        self.opt = MuonAdamW(groups, lr=t["lr"]) if muon else torch.optim.AdamW(
            [{k: v for k, v in g.items() if k != "muon"} for g in groups], lr=t["lr"], fused=True)
        self.gp, self._gp_state = None, None
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: lr_factor(s, t) * (self.gp.scale if self.gp else 1.0))
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
            sd = torch.load(t["init_from"], map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)  # a model.pt, or a ckpt.pt's weights (a plateau checkpoint)
            if self.cfg["model"].get("hc"):  # a plain stack under mHC: block j is blocks.j.block
                sd = {re.sub(r"(encoder\.blocks\.\d+\.)", r"\1block.", n): v for n, v in sd.items()}
            if k := self.cfg["model"].get("attn_insert", 0):  # a pure stack grown: GDN-2 block j sits at j + j // k
                sd = {re.sub(r"(encoder\.blocks\.)(\d+)\.", lambda m: f"{m[1]}{int(m[2]) + int(m[2]) // k}.", n): v
                      for n, v in sd.items()}
            r = self.net.load_state_dict(sd, strict=False)
            self.log(f"[lejepa] init from {t['init_from']}: new {r.missing_keys}, dropped {r.unexpected_keys}", flush=True)
        if g := t.get("grain_plateau"):
            fresh = bool(g.get("fresh")) and not ckpt.exists()   # a new cascade: the source run's state and history ignored
            self.gp = GrainPlateau(g, [] if fresh else self.hist, float(cfg["data"]["patch_mm"]), self.step if fresh else 0)
            if self._gp_state and not fresh:
                self.gp.load_state_dict(self._gp_state)
            self._gp_apply()
            for pg in self.opt.param_groups:   # the restored lr times the scheduler's cuts
                pg["lr"] = t["lr"] * lr_factor(self.step, t) * self.gp.scale
            self.log(f"[lejepa] grain_plateau: stage {self.gp.stage} {self.gp.choices}, lr cuts {self.gp.cuts}, "
                     f"{len(self.gp.rec)} steps of record", flush=True)

    def _gp_apply(self) -> None:
        """The grain_plateau stages' sets (rec_contrast, rec_edge, rec_cells ...) on the model, at start and on each event."""
        for k, v in self.gp.sets().items():
            assert hasattr(self.net, k), f"grain_plateau set: LeJEPA has no attribute {k}"
            setattr(self.net, k, v)
        if self.gp.sets():
            self.log(f"[lejepa] grain_plateau sets {self.gp.sets()}", flush=True)

    def _restore(self, path: Path) -> None:
        c = torch.load(path, map_location="cpu", weights_only=False)
        if self.cfg["train"].get("resume_partial") and not (self.run / "ckpt.pt").exists():
            self._restore_partial(c)   # a checkpoint of a model without the new heads (28/10): see there
        else:
            self.net.load_state_dict(c["model"])
            self.opt.load_state_dict(c["opt"])
        self.sched.load_state_dict(c["sched"])
        self._gp_state = c.get("grain")
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

    def _restore_partial(self, c: dict) -> None:
        """train.resume_partial (28/10): the model of another objective config. Its rec heads grown to
        objective.rec_head_cells (LeJEPA.expand_rec_heads: the same pooled prediction), new modules keep their init;
        Adam's moments carried over by parameter name for every parameter of the same shape (the new heads and a grown
        head start with none)."""
        from .lejepa import LeJEPA
        sd = LeJEPA.expand_rec_heads(dict(c["model"]), self.net.rec_head_cells)
        r = self.net.load_state_dict(sd, strict=False)
        self.log(f"[lejepa] partial resume: new {r.missing_keys}, dropped {r.unexpected_keys}", flush=True)
        old = c["opt"]
        names = {id(p): n for n, p in self.net.named_parameters()}
        new = self.opt.state_dict()
        for gi, g in enumerate(self.opt.param_groups):   # the old run's group gi held the same parameters minus the new ones
            mine = [p for p in g["params"] if names[id(p)] in c["model"]]
            for p, oi in zip(mine, old["param_groups"][gi]["params"]):
                st = old["state"].get(oi)
                if st is not None and st["exp_avg"].shape == p.shape:
                    new["state"][new["param_groups"][gi]["params"][g["params"].index(p)]] = st
        self.opt.load_state_dict(new)

    def _checkpoint(self) -> None:
        """Rank 0; written to a temporary file, then renamed."""
        tmp = self.run / "ckpt.pt.tmp"
        torch.save({"step": self.step, "consumed": self.consumed, "world": self.world, "skipped": self.skipped,
                    "seconds": time.time() - self.t0, "model": self.net.state_dict(),
                    "opt": self.opt.state_dict(), "sched": self.sched.state_dict()}
                   | ({"grain": self.gp.state_dict()} if self.gp else {}), tmp)
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
        if self.gp and self.gp.done:
            self.log(f"[lejepa] grain_plateau already done at step {self.step}", flush=True)
            return
        if self.step >= t["steps"]:
            self.log(f"[lejepa] already at step {self.step} of {t['steps']}", flush=True)
            return
        self.model.train()
        handoff = os.environ.get("SFC_HANDOFF")
        prof, first = os.environ.get("SFC_PROFILE"), self.step + 1  # a dir: steps first+5..first+7 profiled, then stop
        for step in range(self.step + 1, t["steps"] + 1):
            if handoff:   # $SFC_HANDOFF exists (a larger job takes the run over): rank 0's view, on every rank
                flag = torch.tensor([float(self.rank == 0 and os.path.exists(handoff))], device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(flag)
                if flag.item():
                    if save:
                        self._checkpoint()
                    self.log(f"[lejepa] handoff at step {self.step}: checkpoint written, exiting", flush=True)
                    return
            if prof and step == first + 5 and save:
                torch.cuda.memory._record_memory_history(max_entries=500_000)
                p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
                p.__enter__()
            t_w = time.time()
            batch = next(batches)   # the previous step already synced (isfinite(gnorm)): this wait is a GPU stall
            wait += time.time() - t_w
            mm = float(self.cfg["data"]["patch_mm"])
            if choices := self.gp.choices if self.gp else self.cfg["data"].get("patch_mm_choices"):  # one patch size per step, the same on every rank
                mm = float(choices[(self.seed * 1_000_003 + step) % len(choices)])
                full = batch
                batch = repatch(batch, mm, self.cfg["data"].get("max_patches_per_scan"))
                # data.global_mm (25/10, user): finer steps read their global views on a second cut at that cell
                # (the whole volume, as the trunk), the local ones at mm (lejepa.forward: cross-grain invariance)
                if (gmm := self.cfg["data"].get("global_mm")) and mm < gmm:
                    batch["global"] = repatch(full, float(gmm), self.cfg["data"].get("max_patches_per_scan"))
                self.net.encoder.patch_embed.patch_mm = mm
                # data.view_tokens_by_mm {mm: cap} (22/10 screens): fewer tokens per view at small cells, so a 4 mm step
                # holds about as many tokens as a 16 mm one (the same views in mm would hold up to 64x more: OOM)
                caps = self.cfg["data"].get("view_tokens_by_mm") or {}
                # an override without spaces ({4:768}) is one YAML scalar "4:768" -> None: no cap at all (22/10, OOM)
                if any(not isinstance(a, (int, float)) or not isinstance(b, int) for a, b in caps.items()):
                    raise ValueError(f"data.view_tokens_by_mm {caps}: write it as {{4: 768, 8: 1024}} (spaces)")
                base = self.cfg["objective"].get("max_view_tokens")
                cap = caps.get(mm, caps.get(int(mm), base))
                self.net.max_view_tokens = float("inf") if cap is None else cap
            patches += int(batch["grid"].prod(1).sum())
            gen.manual_seed((self.seed * 1_000_003 + step) * 4096 + self.rank)  # resumable on any number of GPUs
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vols = to_device(batch, self.device) | {"session": self.session[batch["index"]].to(self.device)}
                if "global" in batch:
                    vols["global"] = to_device(batch["global"], self.device)
                out = self.model(vols, gen)
            self.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            if torch.isfinite(gnorm):  # a non-finite step would poison the weights and Adam's moments
                self.opt.step()
            else:
                self.skipped += 1
            event = None
            if self.gp:   # the step's watched terms averaged over the GPUs: every rank takes the same decision
                keys = [k for k in self.gp.watched if k in out]
                lo = torch.stack([out[k].detach().float().reshape(()) for k in keys])
                if dist.is_initialized():
                    dist.all_reduce(lo)
                    lo /= self.world
                event = self.gp.update(step, mm, dict(zip(keys, lo.tolist())))   # a cut reaches the lr at sched.step()
            self.sched.step()
            self.step, self.consumed = step, self.consumed + self.world
            if step % t.get("log_every", 50) == 0 or step == t["steps"]:
                dt = time.time() - t_log
                hist.append({"step": step, "mm": mm, **{k: v.item() for k, v in out.items()}, "grad_norm": gnorm.item(),
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
            if event:   # a branch point: encoder, full checkpoint kept, the event in events.json
                self.log(f"[lejepa] grain_plateau at step {step}: {event}", flush=True)
                self._gp_apply()
                if save:
                    torch.save(self.net.encoder.state_dict(), self.run / f"encoder_step{step:06d}.pt")
                    self._checkpoint()
                    shutil.copy(self.run / "ckpt.pt", self.run / f"ckpt_step{step:06d}.pt")
                    (self.run / "events.json").write_text(json.dumps(self.gp.events, indent=1))
                if self.gp.done:
                    break
            if save and (step % t["save_every"] == 0 or step == t["steps"]):
                torch.save(self.net.encoder.state_dict(), self.run / f"encoder_step{step:06d}.pt")
            if save and (step % t.get("ckpt_every", 500) == 0 or step == t["steps"]):
                self._checkpoint()
                if step in t.get("keep_ckpt_at", ()):   # branch points (final run, 25/10): ckpt.pt is overwritten
                    shutil.copy(self.run / "ckpt.pt", self.run / f"ckpt_step{step:06d}.pt")
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
