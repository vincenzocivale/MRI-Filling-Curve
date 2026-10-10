"""Frozen-feature linear probes, no model selection: a fixed L2 per feature kind (`l2` in the probe config:
the values v7's val/CV selection chose, 2026-10-09), fitted on train+val, scored once on test with a bootstrap
CI over volumes, for every checkpoint. Encoder features are averaged over the inference curves (`curves`, test-
time augmentation): each volume read along each curve over its whole patch grid.

Tasks (cfg['task']):
- classification: volume-level label (e.g. sex), L2 logistic regression; ROC AUC (binary) or balanced accuracy.
- regression: volume-level target (e.g. age) centred on each cohort's fit mean, restricted to `cohorts`,
  ridge regression; R^2. Cohort centring removes "which scanner" as a cue.
- segmentation: per-patch majority class (TotalSegmentator), multinomial logistic regression on each patch's
  bi token; macro average precision over foreground classes.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn

from .curves import grid_coords, keys
from .data.dataset import loader, per_patch, to_device
from .data.totalseg import PatchLabels
from .data.totalseg import classes as seg_classes
from .data.volume import VolumeStore
from .io import RunDir, seed_all
from .model import Encoder

# scans on the device (`dataset.to_device`) -> {name: [B,F] (volume level) or [B,N,F] (patch level, N = the
# batch's largest grid)}
Extractor = Callable[[dict], dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------------------- metrics
def roc_auc(scores: torch.Tensor, y: torch.Tensor) -> float:
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    ranks = scores.argsort().argsort().double() + 1
    return (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)


def classification_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(-1)
    out = {"accuracy": (pred == y).float().mean().item(),
           "balanced_accuracy": float(np.mean([(pred[y == c] == c).float().mean().item() for c in y.unique()]))}
    if logits.shape[-1] == 2:
        out["roc_auc"] = roc_auc(logits.softmax(-1)[:, 1], y)
    return out


def regression_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred, y = pred.double().flatten(), y.double()
    res = pred - y
    return {"r2": 1 - (res.square().sum() / (y - y.mean()).square().sum()).item(),
            "pearson_r": torch.corrcoef(torch.stack([pred, y]))[0, 1].item(), "mae": res.abs().mean().item()}


def segmentation_metrics(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Macro average precision over foreground classes present in y (class 0 = background),
    one-vs-rest on softmax scores. Threshold-free: ~90% of patches are background, so an argmax
    metric (F1) scores every linear probe ~0."""
    p = logits.float().softmax(-1)
    ranks = torch.arange(1, len(y) + 1, device=y.device, dtype=torch.float64)
    aps, freq = [], []
    for c in y.unique().tolist():
        if c == 0:
            continue
        hit = (y[p[:, c].argsort(descending=True)] == c).double()
        aps.append(((hit.cumsum(0) / ranks) * hit).sum() / hit.sum())
        freq.append(hit.mean())
    return {"macro_ap": torch.stack(aps).mean().item(), "chance_ap": torch.stack(freq).mean().item(),
            "accuracy": (p.argmax(-1) == y).float().mean().item(), "classes": len(aps)}


def bootstrap_ci(out: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, metrics, key: str,
                 n: int = 1000, seed: int = 0) -> list[float]:
    """Resample whole volumes (`groups`) with replacement; percentile 95% CI of `key`."""
    g = torch.Generator(device=y.device).manual_seed(seed)
    uniq, inverse = groups.unique(return_inverse=True)
    order = inverse.argsort()
    counts = torch.bincount(inverse, minlength=len(uniq))
    starts = counts.cumsum(0) - counts
    vals = []
    for _ in range(n):
        pick = torch.randint(len(uniq), (len(uniq),), device=y.device, generator=g)
        lens = counts[pick]
        rep = torch.repeat_interleave(pick, lens)
        pos = torch.arange(int(lens.sum()), device=y.device) - torch.repeat_interleave(lens.cumsum(0) - lens, lens)
        i = order[starts[rep] + pos]
        if len(y[i].unique()) > 1:
            vals.append(metrics(out[i], y[i])[key])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ------------------------------------------------------------------------------------------ heads
def fit_logistic(x: torch.Tensor, y: torch.Tensor, n_classes: int, l2: float) -> nn.Linear:
    """L2-regularised multinomial logistic regression, solved to convergence with L-BFGS."""
    with torch.enable_grad():
        head = nn.Linear(x.shape[1], n_classes).to(x.device)
        opt = torch.optim.LBFGS(head.parameters(), max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(x), y) + l2 * head.weight.square().sum()
            loss.backward()
            return loss
        opt.step(closure)
    return head.requires_grad_(False)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, l2: float) -> nn.Linear:
    """argmin ||Xw + b - y||^2 / n + l2 ||w||^2, closed form (dual when features outnumber samples)."""
    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean()
    xc, yc, n = x - mx, y - my, len(x)
    if x.shape[1] > n:
        w = xc.T @ torch.linalg.solve(xc @ xc.T / n + l2 * torch.eye(n, device=x.device, dtype=x.dtype), yc / n)
    else:
        w = torch.linalg.solve(xc.T @ xc / n + l2 * torch.eye(x.shape[1], device=x.device, dtype=x.dtype),
                               xc.T @ yc / n)
    head = nn.Linear(x.shape[1], 1).to(x.device)
    head.weight.data, head.bias.data = w.float()[None], (my - mx @ w).float()[None]
    return head.requires_grad_(False)


# ------------------------------------------------------------------------------------------ tasks
class Task:
    """What a probe predicts from which rows, with which head, scored how."""

    level = "volume"
    key = ""
    worker_target = None  # optional per-row target computed in DataLoader workers

    def __init__(self, cfg: dict, df: pd.DataFrame):
        self.cfg, self.label = cfg, cfg["label"]
        self.rows = {"fit": df[df["split"].isin(["train", "val"])].reset_index(drop=True),
                     "test": df[df["split"] == "test"].reset_index(drop=True)}

    def targets(self, split: str, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, x: torch.Tensor, y: torch.Tensor, l2: float) -> nn.Module:
        raise NotImplementedError

    def metrics(self, out: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError


class Classification(Task):
    def __init__(self, cfg, df):
        df = df[df[cfg["label"]].notna()]
        super().__init__(cfg, df)
        self.classes = sorted(df[self.label].unique())
        self.key = "roc_auc" if len(self.classes) == 2 else "balanced_accuracy"
        codes = {c: i for i, c in enumerate(self.classes)}
        self.y = {s: torch.tensor([codes[v] for v in r[self.label]]) for s, r in self.rows.items()}

    def targets(self, split, batch):
        return self.y[split][batch["index"]]

    def fit(self, x, y, l2):
        return fit_logistic(x, y, len(self.classes), l2)

    def metrics(self, out, y):
        return classification_metrics(out, y)


class Regression(Task):
    key = "r2"

    def __init__(self, cfg, df):
        df = df[df[cfg["label"]].notna() & df["cohort"].isin(cfg["cohorts"])]
        super().__init__(cfg, df)
        means = self.rows["fit"].groupby("cohort")[self.label].mean()
        self.y = {s: torch.tensor((r[self.label] - r["cohort"].map(means)).to_numpy(), dtype=torch.float32)
                  for s, r in self.rows.items()}

    def targets(self, split, batch):
        return self.y[split][batch["index"]]

    def fit(self, x, y, l2):
        return fit_ridge(x, y, l2)

    def metrics(self, out, y):
        return regression_metrics(out, y)


class Segmentation(Task):
    level = "patch"
    key = "macro_ap"

    def __init__(self, cfg, df, data_cfg):
        super().__init__(cfg, df[df[cfg["label"]].notna()])
        names = seg_classes(self.rows["fit"]["mask_path"].iloc[0])
        self.n_classes = len(names) + 1
        self.worker_target = PatchLabels(VolumeStore(data_cfg["cache_dir"]), data_cfg["patch_mm"], names)

    def targets(self, split, batch):
        return batch["target"]

    def fit(self, x, y, l2):
        return fit_logistic(x, y, self.n_classes, l2)

    def metrics(self, out, y):
        return segmentation_metrics(out, y)


def make_task(cfg: dict, data_cfg: dict) -> Task:
    df = pd.read_csv(cfg["manifest"])
    kind = cfg["task"]
    if kind == "segmentation":
        return Segmentation(cfg, df, data_cfg)
    return {"classification": Classification, "regression": Regression}[kind](cfg, df)


# ------------------------------------------------------------------------------------- extractors
def encoder_extractor(encoder: Encoder, curves: list[str], level: str, seed: int, fg_threshold: float) -> Extractor:
    """Each volume read along each of `curves` over its whole patch grid (untransformed view), forwards and
    backwards; `bi` = forward ++ backward output per patch, averaged over the curves (test-time augmentation).
    patch: `bi` per patch (raster order). volume: `mean`, bi averaged over the foreground patches (the top of the
    pretraining pyramid)."""
    def extract(vols):
        grid = vols["grid"]
        n = grid.prod(1)
        xyz = grid_coords(grid, int(n.max()))
        j = torch.arange(xyz.shape[1], device=grid.device)
        valid = j < n[:, None]
        rev = torch.where(valid, (n[:, None] - 1 - j).clamp_min(0), j)
        t = torch.zeros(len(grid), xyz.shape[1], encoder.dim, device=grid.device)
        for i, x in enumerate(vols["patches"]):                             # per scan: its own kernel weights
            t[i, : len(x)] = encoder.tokens(x, encoder.patch_embed.weights(vols["spacing"][i], vols["k"][i].tolist()))
        bi = 0
        for curve in curves:
            order = torch.stack([keys(curve, c.clamp_min(0), g, seed).masked_fill(c[:, 0] < 0, 1 << 62).argsort()
                                 for c, g in zip(xyz, grid)])              # per volume: no batch-dependent curve
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = t.gather(1, order[..., None].expand(-1, -1, encoder.dim))
                h = encoder.packed(torch.cat([s, s.gather(1, rev[..., None].expand_as(s))]), valid.repeat(2, 1)).float()
            fwd, bwd = h[: len(grid)], h[len(grid):].gather(1, rev[..., None].expand(-1, -1, h.shape[-1]))
            back = order.argsort(1)[..., None]                                 # curve position -> raster
            bi = bi + torch.cat([fwd, bwd], -1).gather(1, back.expand(-1, -1, 2 * h.shape[-1])) / len(curves)
        if level == "patch":
            return {"bi": bi}
        fg = valid & (per_patch(vols, xyz.shape[1], lambda x: x.mean(-1)) > fg_threshold)
        w = (fg | (valid & ~fg.any(1, keepdim=True))).float()
        return {"mean": (w[..., None] * bi).sum(1) / w.sum(1, keepdim=True)}
    return extract


def raw_extractor(level: str, pooled: int = 16, patch_pooled: int = 8) -> Extractor:
    """What a linear model gets with no encoder. Patch level: the patch's voxels box-averaged to
    patch_pooled^3 (a fixed-length baseline input; patches hold different voxel counts per scan).
    Volume level: per-patch mean intensities box-averaged to a fixed pooled^3 grid."""
    def extract(vols):
        grid = vols["grid"]
        if level == "patch":
            out = torch.zeros(len(grid), int(grid.prod(1).max()), patch_pooled ** 3, device=grid.device)
            for i, x in enumerate(vols["patches"]):
                cubes = x.view(len(x), 1, *vols["k"][i].tolist())
                out[i, : len(x)] = nn.functional.adaptive_avg_pool3d(cubes, patch_pooled).flatten(1)
            return {"voxels": out}
        means = [x.mean(-1).view(*g.tolist()) for x, g in zip(vols["patches"], grid)]
        return {"patch_mean": torch.stack([nn.functional.adaptive_avg_pool3d(m[None], pooled).flatten()
                                           for m in means])}
    return extract


def position_extractor(n_features: int = 256, scale: float = 4.0, seed: int = 0) -> Extractor:
    """Patch level, no image content: random Fourier features (cos, sin) of the patch's grid
    coordinates scaled to [0, 1]^3 per volume. A linear probe on them is a smooth map from location to
    class, i.e. how much of the segmentation is "where", not "what"."""
    w = torch.randn(3, n_features, generator=torch.Generator().manual_seed(seed)) * 2 * torch.pi * scale

    def extract(vols):
        grid = vols["grid"]
        c = grid_coords(grid, int(grid.prod(1).max())).float()
        proj = ((c + 0.5) / grid[:, None]) @ w.to(grid.device)
        return {"rff": torch.cat([proj.cos(), proj.sin()], -1)}
    return extract


# ------------------------------------------------------------------------------------------ probe
class LinearProbe:
    """`candidates` maps a reported name to (extractor, l2). Under torchrun the candidates are split across ranks
    (each extracts, fits and scores its own); rank 0 gathers and writes."""

    def __init__(self, cfg: dict, data_cfg: dict, seen_subjects: set[str]):
        self.cfg, self.data_cfg = cfg, data_cfg
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
        self.task = make_task(cfg, data_cfg)
        for split, rows in self.task.rows.items():
            if leaked := seen_subjects & set(rows["subject"].astype(str)):
                raise RuntimeError(f"{len(leaked)} subjects in probe split {split!r} were seen in "
                                   f"pretraining, e.g. {sorted(leaked)[:3]}.")

    @torch.no_grad()
    def _features(self, extractors: dict[str, Extractor]) -> dict:
        """One pass over each split, every extractor on every batch -> (feats, y, volume ids). Features stay in
        host memory; `run` moves one candidate at a time."""
        task, out = self.task, {}
        gen = torch.Generator(self.device).manual_seed(int(self.cfg["seed"]))
        for split, rows in task.rows.items():
            dl = loader(rows, {**self.data_cfg, "num_workers": self.cfg.get("num_workers", 8)},
                        self.cfg.get("batch_size", 8), train=False, target=task.worker_target)
            feats, ys, groups = {}, [], []
            for batch in dl:
                vols = to_device(batch, self.device)
                y = task.targets(split, batch).to(self.device)
                vol = batch["index"].to(self.device)
                if task.level == "patch":                       # drop all-zero patches and the batch padding
                    keep = (per_patch(vols, y.shape[1], lambda x: x.abs().amax(-1)) > 0) | (y > 0)
                    cap = self.cfg.get("train_patches_per_volume") if split == "fit" else None
                    if cap:                                      # random subset per volume
                        score = torch.rand(keep.shape, device=self.device, generator=gen).masked_fill(~keep, -1)
                        keep &= score >= score.topk(min(cap, keep.shape[1]), dim=1).values[:, -1:]
                    y, vol = y[keep], vol[:, None].expand_as(keep)[keep]
                for name, extract in extractors.items():
                    (f,) = extract(vols).values()
                    feats.setdefault(name, []).append((f[keep] if task.level == "patch" else f).float().cpu())
                ys.append(y)
                groups.append(vol)
            out[split] = ({k: torch.cat(v) for k, v in feats.items()}, torch.cat(ys), torch.cat(groups))
            print(f"[probe] {split}: {len(rows)} volumes, {len(out[split][1])} samples", flush=True)
        return out

    @torch.no_grad()
    def run(self, candidates: dict[str, tuple[Extractor, float]], out_dir: RunDir) -> dict | None:
        seed_all(int(self.cfg["seed"]))
        task, key, results, preds = self.task, self.task.key, {}, {}
        mine = dict(list(candidates.items())[self.rank::self.world])
        if mine:
            data = self._features({n: e for n, (e, _) in mine.items()})
            (ffit, yfit, _), (fte, yte, gte) = data["fit"], data["test"]
            preds = {"y": yte.cpu(), "volume": gte.cpu()}
            for name, (_, l2) in mine.items():
                x = ffit[name].to(self.device)
                mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
                pred = task.fit((x - mu) / sd, yfit, l2)((fte[name].to(self.device) - mu) / sd)
                results[name] = {"l2": l2, "test": task.metrics(pred, yte),
                                 f"test_{key}_ci95": bootstrap_ci(pred, yte, gte, task.metrics, key)}
                preds[name] = pred.half().cpu()
        if self.world > 1:
            parts = [None] * self.world
            dist.gather_object((results, preds), parts if self.rank == 0 else None, dst=0)
            if self.rank:
                return None
            for r, p in parts:
                results |= r
                preds |= p
        results = {n: results[n] for n in candidates}       # config order
        for name, res in results.items():
            print(f"[probe] {name}: l2={res['l2']} test {key}={res['test'][key]:.3f} "
                  f"CI95={[round(v, 3) for v in res[f'test_{key}_ci95']]}", flush=True)
        torch.save(preds, out_dir.path / "test_preds.pt")  # per-candidate test outputs, for paired comparisons
        return {"task": self.cfg["task"], "label": self.cfg["label"], "metric": key,
                "n": {s: len(r) for s, r in task.rows.items()}, "groups": results}


def probe_run(cfg: dict, run_dir: str | Path) -> dict | None:
    """Every saved checkpoint of a run (`step000000` = its own init), features averaged over `cfg['curves']`
    (default: the run's view_curves), L2 `cfg['l2']['encoder']`."""
    run = RunDir(Path(run_dir))
    rcfg, device = run.config, torch.device("cuda")

    def load(path: Path) -> Encoder:
        enc = Encoder.from_config(rcfg)
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        return enc.to(device).eval().requires_grad_(False)

    ckpts = sorted(run.path.glob("encoder_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No encoder_step*.pt in {run.path}")
    probe = LinearProbe(cfg, rcfg["data"], set(pd.read_csv(run.path / "split.csv")["subject"].astype(str)))
    obj = rcfg["objective"]
    curves = cfg.get("curves", obj["view_curves"])
    candidates = {c.stem.removeprefix("encoder_"): (encoder_extractor(load(c), curves, probe.task.level, int(rcfg["seed"]),
                                                                      obj.get("fg_threshold", 0.05)), cfg["l2"]["encoder"])
                  for c in ckpts}
    out = RunDir.create(run.path, cfg, prefix=f"probe-{cfg['label']}")
    result = probe.run(candidates, out)
    if result is None:                                     # non-zero rank
        return None
    result |= {"curve": "+".join(curves), "objective": "lejepa"}
    out.save_json(result, "metrics.json")
    return result


def probe_raw(cfg: dict, pretrain_cfg: dict) -> dict | None:
    """Encoder-free baselines on the same probe splits, written next to the pretraining runs."""
    df = pd.concat([pd.read_csv(m) for m in pretrain_cfg["data"]["manifests"]], ignore_index=True)
    seen = set(df[df["split"].isin(pretrain_cfg["data"]["splits"])]["subject"].astype(str))
    probe = LinearProbe(cfg, pretrain_cfg["data"], seen)
    out = RunDir.create(pretrain_cfg["output_root"], cfg, prefix=f"raw-probe-{cfg['label']}")
    candidates = {"raw": (raw_extractor(probe.task.level), cfg["l2"]["raw"])}
    if probe.task.level == "patch":
        candidates["position"] = (position_extractor(), cfg["l2"]["position"])
    result = probe.run(candidates, out)
    if result is None:
        return None
    result |= {"curve": "none", "objective": "raw"}
    out.save_json(result, "metrics.json")
    return result
