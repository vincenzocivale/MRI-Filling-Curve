"""Frozen-feature linear probes. All model selection (checkpoint, pooling, L2) uses the probe's val
split only; test is touched once, for the selected candidate, with a bootstrap CI over volumes.

Tasks (cfg['task']):
- classification: volume-level label (e.g. sex), L2 logistic regression; selects on ROC AUC
  (binary) or balanced accuracy.
- regression: volume-level target (e.g. age) centred on each cohort's train mean, restricted to
  `cohorts`, ridge regression; selects on R^2. Cohort centring removes "which scanner" as a cue.
- segmentation: per-patch majority class (TotalSegmentator), multinomial logistic regression on
  each patch's token; selects on macro average precision over foreground classes.

With `cv_folds` in the probe config (volume-level tasks), selection uses subject-grouped k-fold
cross-validation on train+val instead of the val split alone, and the selected candidate is
refitted on train+val before the single test evaluation.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .curves import CurveViews
from .data.dataset import loader
from .data.totalseg import PatchLabels
from .data.totalseg import classes as seg_classes
from .data.volume import VolumeStore
from .io import RunDir, seed_all
from .model import Encoder

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
FG_THRESHOLD = 0.05  # foreground patch = mean intensity above this (as in LeJEPA's anchor sampling)
# patches [B,N,V] -> {name: [B,F] (volume level) or [B,N,F] (patch level)}
Extractor = Callable[[torch.Tensor], dict[str, torch.Tensor]]


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
    l2_grid = L2_GRID
    worker_target = None  # optional per-row target computed in DataLoader workers

    def __init__(self, cfg: dict, df: pd.DataFrame):
        self.cfg, self.label = cfg, cfg["label"]
        self.rows = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}

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
    l2_grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

    def __init__(self, cfg, df):
        df = df[df[cfg["label"]].notna() & df["cohort"].isin(cfg["cohorts"])]
        super().__init__(cfg, df)
        means = self.rows["train"].groupby("cohort")[self.label].mean()
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
        names = seg_classes(self.rows["train"]["mask_path"].iloc[0])
        self.n_classes = len(names) + 1
        self.worker_target = PatchLabels(VolumeStore(data_cfg["cache_dir"], data_cfg["target_shape"]),
                                         data_cfg["patch_size"], names)

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
def encoder_extractor(encoder: Encoder, perm: torch.Tensor, level: str) -> Extractor:
    """volume: mean (all tokens), late (second half), last (final causal state), fg_mean / fg_meanstd
    (mean, and mean ++ std, over foreground tokens only: background is 60-75% of the sequence).
    patch: each patch's own output token (`token`, causal: it has seen only what precedes it), and
    `bi` = token ++ the token of the same curve traversed backwards, both in canonical order."""
    rank, rev = perm.argsort(), perm.flip(0)
    rank_rev = rev.argsort()

    def extract(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = encoder(x, perm).float()
            hb = encoder(x, rev).float() if level == "patch" else None
        if level == "patch":
            fwd = h[:, rank]
            return {"token": fwd, "bi": torch.cat([fwd, hb[:, rank_rev]], -1)}
        w = (x.mean(-1) > FG_THRESHOLD)[:, perm].float()[..., None]
        n = w.sum(1).clamp_min(1)
        m = (h * w).sum(1) / n
        sd = (((h - m[:, None]).square() * w).sum(1) / n).sqrt()
        return {"mean": h.mean(1), "late": h[:, h.shape[1] // 2:].mean(1), "last": h[:, -1],
                "fg_mean": m, "fg_meanstd": torch.cat([m, sd], -1)}
    return extract


def raw_extractor(level: str) -> Extractor:
    """What a linear model gets with no encoder: per-patch mean intensities (volume level) or the
    patch's own voxels (patch level)."""
    def extract(x):
        return {"voxels": x} if level == "patch" else {"patch_mean": x.mean(-1)}
    return extract


def position_extractor(n_features: int = 256, scale: float = 4.0, seed: int = 0) -> Extractor:
    """Patch level, no image content: random Fourier features (cos, sin) of the patch's canonical
    grid coordinates in [0, 1]^3. A linear probe on them is a smooth map from location to class,
    i.e. how much of the segmentation is "where", not "what"."""
    w = torch.randn(3, n_features, generator=torch.Generator().manual_seed(seed)) * 2 * torch.pi * scale

    def extract(x):
        g = round(x.shape[1] ** (1 / 3))
        c = torch.stack(torch.meshgrid(*[torch.arange(g, device=x.device)] * 3, indexing="ij"), -1)
        proj = ((c.view(-1, 3).float() + 0.5) / g) @ w.to(x.device)
        return {"rff": torch.cat([proj.cos(), proj.sin()], -1).expand(len(x), -1, -1)}
    return extract


# ------------------------------------------------------------------------------------------ probe
class LinearProbe:
    """`groups` maps a reported name to the candidate extractors it selects among (e.g.
    "pretrained" -> every checkpoint x pooling). Each group yields one test result."""

    def __init__(self, cfg: dict, data_cfg: dict, seen_subjects: set[str]):
        self.cfg, self.data_cfg = cfg, data_cfg
        self.device = torch.device("cuda")
        self.task = make_task(cfg, data_cfg)
        for split, rows in self.task.rows.items():
            if leaked := seen_subjects & set(rows["subject"].astype(str)):
                raise RuntimeError(f"{len(leaked)} subjects in probe split {split!r} were seen in "
                                   f"pretraining, e.g. {sorted(leaked)[:3]}.")

    @torch.no_grad()
    def _features(self, extractors: dict[str, Extractor]) -> dict:
        """One pass over each split, every extractor on every batch -> (feats, y, volume ids).
        Features stay in host memory (7 checkpoints x patch features exceed a shared GPU); `run`
        moves one candidate at a time."""
        task, out = self.task, {}
        gen = torch.Generator(self.device).manual_seed(int(self.cfg["seed"]))
        for split, rows in task.rows.items():
            dl = loader(rows, {**self.data_cfg, "num_workers": self.cfg.get("num_workers", 8)},
                        self.cfg.get("batch_size", 8), train=False, target=task.worker_target)
            feats, ys, groups = {}, [], []
            for batch in dl:
                x = batch["patches"].to(self.device, non_blocking=True).float()
                y = task.targets(split, batch).to(self.device)
                vol = batch["index"].to(self.device)
                if task.level == "patch":
                    keep = (x.amax(-1) > 0) | (y > 0)           # drop zero padding
                    cap = self.cfg.get("train_patches_per_volume") if split == "train" else None
                    if cap:                                      # random subset per volume
                        score = torch.rand(keep.shape, device=self.device, generator=gen).masked_fill(~keep, -1)
                        keep &= score >= score.topk(min(cap, keep.shape[1]), dim=1).values[:, -1:]
                    y, vol = y[keep], vol[:, None].expand_as(keep)[keep]
                for name, extract in extractors.items():
                    for pooling, f in extract(x).items():
                        f = f[keep] if task.level == "patch" else f
                        feats.setdefault(f"{name}/{pooling}", []).append(f.float().cpu())
                ys.append(y)
                groups.append(vol)
            out[split] = ({k: torch.cat(v) for k, v in feats.items()}, torch.cat(ys), torch.cat(groups))
            print(f"[probe] {split}: {len(rows)} volumes, {len(out[split][1])} samples", flush=True)
        return out

    @torch.no_grad()
    def run(self, groups: dict[str, dict[str, Extractor]], out_dir: RunDir) -> dict:
        seed_all(int(self.cfg["seed"]))
        task, key = self.task, self.task.key
        data = self._features({f"{g}:{k}": e for g, ext in groups.items() for k, e in ext.items()})
        (ftr, ytr, gtr), (fva, yva, gva), (fte, yte, gte) = data["train"], data["val"], data["test"]
        folds = int(self.cfg.get("cv_folds", 0))
        if folds:  # merge train+val; folds by subject; the selected candidate is refitted on all of it
            subj = np.concatenate([task.rows["train"]["subject"].astype(str).to_numpy()[gtr.cpu().numpy()],
                                   task.rows["val"]["subject"].astype(str).to_numpy()[gva.cpu().numpy()]])
            uniq = np.random.default_rng(int(self.cfg["seed"])).permutation(np.unique(subj))
            fold = torch.as_tensor(pd.Series(np.arange(len(uniq)) % folds, index=uniq)[subj].to_numpy().copy(),
                                   device=self.device)
            ftr, ytr = {c: torch.cat([ftr[c], fva[c]]) for c in ftr}, torch.cat([ytr, yva])
        grid, results = [], {}
        for cand in ftr:
            xtr = ftr[cand].to(self.device)
            xva = None if folds else fva[cand].to(self.device)
            mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
            for l2 in task.l2_grid:
                head = task.fit((xtr - mu) / sd, ytr, l2)
                if folds:
                    oof = None
                    for k in range(folds):
                        tr = fold != k
                        m_k, s_k = xtr[tr].mean(0), xtr[tr].std(0).clamp_min(1e-6)
                        o = task.fit((xtr[tr] - m_k) / s_k, ytr[tr], l2)((xtr[~tr] - m_k) / s_k)
                        oof = o.new_zeros(len(ytr), *o.shape[1:]) if oof is None else oof
                        oof[~tr] = o
                    val = task.metrics(oof, ytr)
                else:
                    val = task.metrics(head((xva - mu) / sd), yva)
                grid.append({"candidate": cand, "l2": l2, **{f"val_{k}": v for k, v in val.items()},
                             "_head": head, "_norm": (mu, sd)})
        # each group selects among its candidates; patch-level probes also report one result per
        # feature kind (`token` vs `bi`), since the backward half can mask what the curve changes
        selections = {g: (lambda c, g=g: c.startswith(f"{g}:")) for g in groups}
        if task.level == "patch":
            kinds = sorted({c.rsplit("/", 1)[1] for c in ftr})
            selections |= {f"{g}@{k}": (lambda c, g=g, k=k: c.startswith(f"{g}:") and c.endswith(f"/{k}"))
                           for g in groups for k in kinds if any(c.startswith(f"{g}:") and c.endswith(f"/{k}") for c in ftr)}
        preds = {"y": yte.cpu(), "volume": gte.cpu()}
        for group, keep in selections.items():
            best = max((r for r in grid if keep(r["candidate"])), key=lambda r: r[f"val_{key}"])
            mu, sd = best["_norm"]
            pred = best["_head"]((fte[best["candidate"]].to(self.device) - mu) / sd)
            res = {"selected": best["candidate"].split(":", 1)[1], "l2": best["l2"],
                   "val": {k[4:]: v for k, v in best.items() if k.startswith("val_")},
                   "test": task.metrics(pred, yte),
                   f"test_{key}_ci95": bootstrap_ci(pred, yte, gte, task.metrics, key)}
            results[group] = res
            preds[group] = pred.half().cpu()
            print(f"[probe] {group}: {res['selected']} l2={res['l2']} val {key}={res['val'][key]:.3f} "
                  f"test {key}={res['test'][key]:.3f} CI95={[round(v, 3) for v in res[f'test_{key}_ci95']]}",
                  flush=True)
        out_dir.save_table(pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in grid]),
                           "val_grid.csv")
        torch.save(preds, out_dir.path / "test_preds.pt")  # per-group test outputs, for paired comparisons
        return {"task": self.cfg["task"], "label": self.cfg["label"],
                "select_on": f"cv{folds}_{key}" if folds else f"val_{key}",
                "n": {s: len(r) for s, r in task.rows.items()}, "groups": results}


def probe_run(cfg: dict, run_dir: str | Path) -> dict:
    """Pretrained (selected among all saved checkpoints) vs its own step-0 initialisation, once per
    inference curve in the run's `view_curves` (groups `pretrained_<curve>`, `init_<curve>`)."""
    run = RunDir(Path(run_dir))
    rcfg, device = run.config, torch.device("cuda")

    def load(path: Path) -> Encoder:
        enc = Encoder.from_config(rcfg)
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        return enc.to(device).eval().requires_grad_(False)

    init, *ckpts = sorted(run.path.glob("encoder_step*.pt"))  # step000000 = the run's own init
    if init.stem != "encoder_step000000" or not ckpts:
        raise FileNotFoundError(f"Need encoder_step000000.pt and trained checkpoints in {run.path}")
    probe = LinearProbe(cfg, rcfg["data"], set(pd.read_csv(run.path / "split.csv")["subject"].astype(str)))
    level = probe.task.level
    curves = rcfg["objective"]["view_curves"]
    encoders = {c.stem.removeprefix("encoder_"): load(c) for c in [init, *ckpts]}
    groups = {}
    for curve in curves:
        perm = CurveViews(curve, encoders["step000000"].grid, int(rcfg["seed"])).perms[0].to(device)
        ext = {name: encoder_extractor(enc, perm, level) for name, enc in encoders.items()}
        groups[f"pretrained_{curve}"] = {n: e for n, e in ext.items() if n != "step000000"}
        groups[f"init_{curve}"] = {"step000000": ext["step000000"]}
    out = RunDir.create(run.path, cfg, prefix=f"probe-{cfg['label']}")
    result = probe.run(groups, out) | {"curve": "+".join(curves), "objective": "lejepa"}
    out.save_json(result, "metrics.json")
    return result


def probe_raw(cfg: dict, pretrain_cfg: dict) -> dict:
    """Encoder-free baseline on the same probe splits, written next to the pretraining runs."""
    df = pd.concat([pd.read_csv(m) for m in pretrain_cfg["data"]["manifests"]], ignore_index=True)
    seen = set(df[df["split"].isin(pretrain_cfg["data"]["splits"])]["subject"].astype(str))
    probe = LinearProbe(cfg, pretrain_cfg["data"], seen)
    out = RunDir.create(pretrain_cfg["output_root"], cfg, prefix=f"raw-probe-{cfg['label']}")
    groups = {"raw": {"raw": raw_extractor(probe.task.level)}}
    if probe.task.level == "patch":
        groups["position"] = {"position": position_extractor()}
    result = probe.run(groups, out) | {"curve": "none", "objective": "raw"}
    out.save_json(result, "metrics.json")
    return result
