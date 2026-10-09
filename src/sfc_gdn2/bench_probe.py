"""Frozen-feature probes of the downstream benchmark, the same for every model (external FMs and ours).

Input: the feature archive `<root>/<model>/<id>.pt` written by `sfc bench-extract` (any model writing that format
plugs in), `volumes.csv` and the fixed `splits.csv` (bench.py). Only volumes that every compared model has are used,
so all models see the same rows. Per task and split seed, two heads on the same input: `linear` and `mlp` (one hidden
layer of 256, GELU), both fitted full batch with L-BFGS on train. The grid is the same for every model: input
feature (each 1-D tensor the model stores, and the spatial / token mean of each map it stores for every volume) x
L2 (on weight matrices). The best (feature, L2) on val is evaluated once on test, with a 95% CI over 1000 bootstrap
resamples of test subjects (fixed seed: the same resamples for every model, so differences are paired).

Task types (`configs/leonardo/probe_tasks.yaml`):
- `class`: softmax cross-entropy, class-balanced; AUROC (binary) or balanced accuracy.
- `ordinal`: CORAL (one score, K-1 thresholds, BCE on y > k); quadratic weighted kappa (+ MAE).
- `dex`: DEX / SFCN, softmax over bins of width `bin` with Gaussian soft labels (sigma = 1 bin), prediction =
  expectation; MAE (+ Pearson r, MAE per dataset, MAE after the linear bias correction fitted on val).
- `cox`: Cox partial likelihood (Breslow); Harrell's C.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HIDDEN = {"linear": 0, "mlp": 256}


# ------------------------------------------------------------------------------------------- features
def _vec(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    if t.ndim == 1:
        return t
    return t.mean(0) if t.ndim == 2 else t.flatten(1).mean(1)  # tokens [T, C] / map [C, *spatial]


def load_globals(model_dir: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    """(ids, {feature: [N, D] fp16}) of every volume of a model; cached in `<model_dir>/_globals.pt`."""
    cache = model_dir / "_globals.pt"
    ids = sorted(p.stem for p in model_dir.glob("*.pt") if not p.name.startswith("_"))
    if cache.exists():
        c = torch.load(cache)
        if c["ids"] == ids:
            return c["ids"], c["feats"]
    rows = []
    for i in ids:
        try:
            r = torch.load(model_dir / f"{i}.pt", map_location="cpu", mmap=True, weights_only=False)
        except RuntimeError:  # files of older torch: no mmap
            r = torch.load(model_dir / f"{i}.pt", map_location="cpu", weights_only=False)
        rows.append({k: _vec(v).clone() for k, v in r["features"].items()  # clone: release the file mapping
                     if isinstance(v, torch.Tensor) and v.is_floating_point()})
    names = sorted(set.intersection(*(set(r) for r in rows)))
    c = {"ids": ids, "feats": {n: torch.stack([r[n] for r in rows]).half() for n in names}}
    torch.save(c, cache)
    return c["ids"], c["feats"]


# ----------------------------------------------------------------------------------------- task types
class Kind:
    """Head output size, loss, prediction and metrics of a task type; `key` is the selection metric."""

    def __init__(self, cfg: dict, y_train: torch.Tensor):
        self.cfg = cfg

    def outputs(self) -> int:
        return 1

    def loss(self, out, y):
        raise NotImplementedError

    def predict(self, out):
        return out.squeeze(-1)

    def metrics(self, pred, y, rows: pd.DataFrame) -> dict:
        raise NotImplementedError


class Class(Kind):
    def __init__(self, cfg, y_train):
        super().__init__(cfg, y_train)
        self.k = int(y_train.max()) + 1
        counts = torch.bincount(y_train.long(), minlength=self.k).float()
        self.weight = counts.sum() / (self.k * counts.clamp_min(1))
        self.key = "auroc" if self.k == 2 else "balanced_accuracy"

    def outputs(self):
        return self.k

    def loss(self, out, y):
        return F.cross_entropy(out, y.long(), weight=self.weight.to(out.device))

    def predict(self, out):
        return out.softmax(-1)

    def metrics(self, pred, y, rows):
        y = y.long()
        bal = torch.stack([(pred.argmax(-1)[y == c] == c).float().mean() for c in y.unique()]).mean().item()
        out = {"balanced_accuracy": bal}
        if self.k == 2:
            out["auroc"] = auroc(pred[:, 1], y)
        return out


class Ordinal(Kind):
    key = "qwk"
    ordinal = True  # Head: one score + K-1 learned thresholds (CORAL)

    def __init__(self, cfg, y_train):
        super().__init__(cfg, y_train)
        self.k = int(y_train.max()) + 1

    def outputs(self):
        return self.k - 1

    def loss(self, out, y):
        target = (y[:, None] > torch.arange(self.k - 1, device=y.device)).float()
        return F.binary_cross_entropy_with_logits(out, target)

    def predict(self, out):
        return (out > 0).sum(1).float()

    def metrics(self, pred, y, rows):
        return {"qwk": qwk(pred.long(), y.long(), self.k), "mae": (pred - y).abs().mean().item()}


class Dex(Kind):
    key = "neg_mae"

    def __init__(self, cfg, y_train):
        super().__init__(cfg, y_train)
        w = float(cfg.get("bin", 1.0))
        lo, hi = float(y_train.min()) - 2 * w, float(y_train.max()) + 2 * w
        self.centers = torch.arange(lo, hi + w, w)
        self.sigma = w

    def outputs(self):
        return len(self.centers)

    def loss(self, out, y):
        c = self.centers.to(out.device)
        soft = torch.softmax(-(c[None] - y[:, None]).square() / (2 * self.sigma ** 2), -1)
        return -(soft * out.log_softmax(-1)).sum(-1).mean()

    def predict(self, out):
        return out.softmax(-1) @ self.centers.to(out.device)

    def metrics(self, pred, y, rows):
        mae = (pred - y).abs().mean().item()
        out = {"neg_mae": -mae, "mae": mae, "pearson_r": torch.corrcoef(torch.stack([pred, y]))[0, 1].item()}
        if self.cfg.get("per_dataset", True):
            err = (pred - y).abs().cpu().numpy()
            out["mae_per_dataset"] = {d: float(err[m].mean()) for d, m in rows.groupby("dataset").indices.items()}
        return out


class Cox(Kind):
    key = "c_index"

    def loss(self, out, y):
        time, event = y[:, 0], y[:, 1]
        s = out.squeeze(-1)[time.argsort(descending=True)]
        e = event[time.argsort(descending=True)]
        return -((s - torch.logcumsumexp(s, 0)) * e).sum() / e.sum().clamp_min(1)

    def metrics(self, pred, y, rows):
        return {"c_index": harrell_c(pred, y[:, 0], y[:, 1])}


KINDS = {"class": Class, "ordinal": Ordinal, "dex": Dex, "cox": Cox}


# ------------------------------------------------------------------------------------------- metrics
def auroc(score: torch.Tensor, y: torch.Tensor) -> float:
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = score.argsort().argsort().double() + 1
    return (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)


def qwk(pred: torch.Tensor, y: torch.Tensor, k: int) -> float:
    o = torch.zeros(k, k, device=y.device).index_put_((y, pred.clamp(0, k - 1)), torch.ones_like(y, dtype=torch.float),
                                                      accumulate=True)
    e = o.sum(1, keepdim=True) @ o.sum(0, keepdim=True) / o.sum()
    i = torch.arange(k, device=y.device, dtype=torch.float)
    w = (i[:, None] - i[None]).square() / max((k - 1) ** 2, 1)
    return (1 - (w * o).sum() / (w * e).sum().clamp_min(1e-12)).item()


def harrell_c(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    comparable = (time[:, None] < time[None]) & (event[:, None] == 1)  # i fails before j is censored/fails
    conc = (risk[:, None] > risk[None]).float() + 0.5 * (risk[:, None] == risk[None]).float()
    return ((conc * comparable).sum() / comparable.sum()).item() if comparable.any() else float("nan")


# ------------------------------------------------------------------------------------------- fitting
class Head(nn.Module):
    """Linear (hidden=0) or one hidden layer; `ordinal`: one score + `out` thresholds (CORAL)."""

    def __init__(self, d: int, out: int, hidden: int, ordinal: bool = False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.GELU()) if hidden else nn.Identity()
        self.out = nn.Linear(hidden or d, 1 if ordinal else out)
        self.th = nn.Parameter(torch.zeros(out)) if ordinal else None

    def forward(self, x):
        s = self.out(self.body(x))
        return s if self.th is None else s + self.th


def fit(x: torch.Tensor, y: torch.Tensor, kind: Kind, l2: float, hidden: int, seed: int = 0) -> Head:
    torch.manual_seed(seed)
    head = Head(x.shape[1], kind.outputs(), hidden, getattr(kind, "ordinal", False)).to(x.device)
    opt = torch.optim.LBFGS(head.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = kind.loss(head(x), y) + l2 * sum(p.square().sum() for p in head.parameters() if p.ndim > 1)
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return head.eval().requires_grad_(False)


def bootstrap(kind: Kind, pred, y, rows: pd.DataFrame, n: int = 1000, seed: int = 0) -> list[float]:
    """95% CI of kind.key over resamples of test subjects (fixed seed -> paired across models)."""
    rng = np.random.default_rng(seed)
    groups = rows.groupby("subject").indices
    subjects = sorted(groups)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[s] for s in rng.choice(subjects, len(subjects))])
        t = torch.as_tensor(idx, device=pred.device)
        vals.append(kind.metrics(pred[t], y[t], rows.iloc[idx])[kind.key])  # nan when degenerate
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


# ------------------------------------------------------------------------------------------- runner
def task_rows(vols: pd.DataFrame, splits: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    rows = vols.query(cfg["query"]) if cfg.get("query") else vols
    need = cfg.get("requires") or (cfg["label"] if cfg["type"] == "cox" else [cfg["label"]])
    rows = rows[rows[need].notna().all(axis=1)]  # an expression label (e.g. "cdr > 0") lists its columns in `requires`
    split = splits.set_index("subject")[f"split_s{seed}"]
    return rows.assign(split=rows["subject"].map(split).to_numpy()).sort_values("id").reset_index(drop=True)


def targets(rows: pd.DataFrame, cfg: dict) -> tuple[torch.Tensor, list]:
    if cfg["type"] == "cox":
        return torch.tensor(rows[list(cfg["label"])].to_numpy(dtype=np.float32)), []
    y = rows.eval(cfg["label"])
    if cfg["type"] in ("class", "ordinal"):
        classes = sorted(y.unique())
        return torch.tensor(y.map({c: i for i, c in enumerate(classes)}).to_numpy(), dtype=torch.float32), classes
    return torch.tensor(y.to_numpy(dtype=np.float32)), []


def run_task(name: str, cfg: dict, rows: pd.DataFrame, feats: dict[str, torch.Tensor], index: dict[str, int],
             device: str = "cuda") -> dict:
    """rows: the task's volumes (with `split`); feats/index: the model's globals and id -> row."""
    y_all, classes = targets(rows, cfg)
    sel = {s: np.flatnonzero(rows["split"].to_numpy() == s) for s in ("train", "val", "test")}
    rid = torch.tensor([index[i] for i in rows["id"]])
    y = {s: y_all[sel[s]].to(device) for s in sel}
    kind = KINDS[cfg["type"]](cfg, y["train"].cpu())
    out = {"task": name, "type": cfg["type"], "metric": kind.key, "classes": [str(c) for c in classes],
           "n": {s: len(v) for s, v in sel.items()}, "subjects": {s: int(rows.iloc[v]["subject"].nunique())
                                                                      for s, v in sel.items()}}
    for variant, hidden in HIDDEN.items():
        best = None
        for fname, f in feats.items():
            x = f[rid].float().to(device)
            mu, sd = x[sel["train"]].mean(0), x[sel["train"]].std(0).clamp_min(1e-6)
            x = (x - mu) / sd
            xs = {s: x[sel[s]] for s in sel}
            for l2 in L2_GRID:
                head = fit(xs["train"], y["train"], kind, l2, hidden)
                val = kind.metrics(kind.predict(head(xs["val"])), y["val"], rows.iloc[sel["val"]])
                if best is None or val[kind.key] > best["val"][kind.key]:
                    best = {"feature": fname, "l2": l2, "val": val, "head": head, "xs": xs}
        pred = {s: kind.predict(best["head"](best["xs"][s])) for s in sel}
        test_rows = rows.iloc[sel["test"]]
        res = {"feature": best["feature"], "l2": best["l2"], "val": best["val"],
               "test": kind.metrics(pred["test"], y["test"], test_rows),
               "test_ci95": bootstrap(kind, pred["test"], y["test"], test_rows)}
        if cfg["type"] == "dex":  # age-bias correction (regression to the mean) fitted on val
            a, b = np.polyfit(y["val"].cpu().numpy(), pred["val"].cpu().numpy(), 1)
            res["test"]["mae_bias_corrected"] = float(((pred["test"] - b) / a - y["test"]).abs().mean())
        res["predictions"] = {"id": test_rows["id"].tolist(), "pred": pred["test"].cpu().tolist()}
        out[variant] = res
    return out


def run(models: dict[str, Path], tasks: dict[str, dict], vols: pd.DataFrame, splits: pd.DataFrame,
        out_dir: Path, seeds: list[int], device: str = "cuda") -> pd.DataFrame:
    """Every task x seed x model; only the volumes all `models` have. Writes `<out>/<task>/s<seed>/<model>.json`."""
    loaded = {m: load_globals(d) for m, d in models.items()}
    common = set.intersection(*(set(ids) for ids, _ in loaded.values()))
    vols = vols[vols["id"].isin(common)]
    summary = []
    for tname, cfg in tasks.items():
        for seed in seeds:
            rows = task_rows(vols, splits, cfg, seed)
            if cfg["type"] == "seg":
                from . import bench_seg
                for m in loaded:
                    res = bench_seg.run_seg(tname, cfg, rows, m, models[m], seed, device)
                    bench_seg.write(res, out_dir)
                    summary += [{"task": tname, "seed": seed, "model": m, "head": h, "metric": "dice",
                                 "test": res[h]["test_dice"], "ci95": res[h]["test_ci95"], "n_test": res["n"]["test"]}
                                for h in ("linear", "conv", "oracle")]
                    print(f"[probe] {tname} s{seed} {m}: " + ", ".join(
                        f"{h} dice={res[h]['test_dice']:.3f}" for h in ("linear", "conv", "oracle")), flush=True)
                continue
            for m, (ids, feats) in loaded.items():
                res = run_task(tname, cfg, rows, feats, {i: k for k, i in enumerate(ids)}, device)
                dst = out_dir / tname / f"s{seed}" / f"{m}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(res | {"model": m, "seed": seed}, indent=1))
                key = res["metric"]
                for variant in HIDDEN:
                    r = res[variant]
                    summary.append({"task": tname, "seed": seed, "model": m, "head": variant, "metric": key,
                                    "val": r["val"][key], "test": r["test"][key], "ci95": r["test_ci95"],
                                    "feature": r["feature"], "l2": r["l2"], "n_test": res["n"]["test"]})
                print(f"[probe] {tname} s{seed} {m}: " + ", ".join(
                    f"{v} {key}={res[v]['test'][key]:.3f} {res[v]['test_ci95']}" for v in HIDDEN), flush=True)
    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False, mode="a", header=not (out_dir / "summary.csv").exists())
    return df
