"""Frozen-feature probes of the downstream benchmark, the same for every model (external FMs and ours).

Input: the feature archive `<root>/<model>/<id>.pt` written by `sfc bench-extract` (any model writing that format
plugs in), `volumes.csv` and the fixed `splits.csv` (bench.py). Only volumes that every compared model has are used,
so all models see the same rows. Per task and split seed, two heads on the same input: `linear` and `mlp` (one hidden
layer of 256, GELU), both fitted full batch with L-BFGS on train. The grid is the same for every model: input
feature (each 1-D tensor the model stores, and the spatial / token mean of each map it stores for every volume) x
L2 (on weight matrices). The best (feature, L2) on val is evaluated once on test, with a 95% CI over 1000 bootstrap
resamples of test subjects (fixed seed: the same resamples for every model, so differences are paired).

Task types (`configs/bench/probe_tasks.yaml`):
- `class`: softmax cross-entropy, class-balanced; AUROC (binary) or balanced accuracy.
- `ordinal`: CORAL (one score, K-1 thresholds, BCE on y > k); quadratic weighted kappa (+ MAE).
- `dex`: DEX / SFCN, softmax over bins of width `bin` with Gaussian soft labels (sigma = 1 bin), prediction =
  expectation; MAE (+ Pearson r, MAE per dataset, MAE after the linear bias correction fitted on val).
- `cox`: Cox partial likelihood (Breslow); Harrell's C.

Pretraining tuning (`tune`, `bench-probe --tune`): only subjects that are in no seed's test split (`tune_subjects`),
out-of-fold predictions, a cross-fitted choice of head; one `skill` per task (0 = chance, 1 = perfect).
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
HIDDEN = {"linear": 0, "mlp": 256}
# val with fewer subjects than this (user, 21/10: e.g. cmb_any, 7 val subjects -> val AUROC 0.32, test 0.74): the
# (feature, L2) is chosen on out-of-fold predictions of a CV_FOLDS-fold split of train + val by subject, then the head
# is refitted on train + val; test is never seen by the choice. Same rule for every model.
CV_MIN_VAL, CV_FOLDS = 25, 3  # 3 folds, not 5 (user, 24/10: the cost of the noisy tasks)


# ------------------------------------------------------------------------------------------- features
def _vec(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    if t.ndim == 1:
        return t
    return t.mean(0) if t.ndim == 2 else t.flatten(1).mean(1)  # tokens [T, C] / map [C, *spatial]


def _pmap(fn, items, initargs=()):
    """map over files in SFC_IO_WORKERS processes (the second-order pass of 18k files was I/O-bound on one core)."""
    import multiprocessing as mp
    w = int(os.environ.get("SFC_IO_WORKERS", "1"))
    if w <= 1:
        if initargs:
            _init_basis(*initargs)
        yield from map(fn, items)
        return
    # results come back as numpy (pickled through the pipe): torch tensors travel as shared-memory fds, and on the
    # serial partition mapping them failed (ENOMEM) in the pool's result thread, which hangs the pool (20/10)
    # spawn, not fork: a pool forked after the parent ran torch's OpenMP threads hung in its first parallel op (the
    # 12 mm readings after the 16 mm second-order pass, 20/10: every worker asleep at 0 % CPU)
    with mp.get_context("spawn").Pool(w, initializer=_init_basis if initargs else None, initargs=initargs) as pool:
        for r in pool.imap(functools.partial(_as_np, fn), items, chunksize=8):
            yield _as_torch(r)


def _as_np(fn, item):
    r = fn(item)
    return {k: (v.float() if v.dtype == torch.bfloat16 else v).numpy() if torch.is_tensor(v) else v
            for k, v in r.items()} if isinstance(r, dict) else r


def _as_torch(r):
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in r.items()} if isinstance(r, dict) else r


def _globals_one(path: Path) -> dict:
    r = torch.load(path, map_location="cpu", weights_only=False)
    return {k: _vec(v).clone() for k, v in r["features"].items()
            if isinstance(v, torch.Tensor) and v.is_floating_point() and "tricov" not in k}   # covariances: second_order


def load_globals(model_dir: Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    """(ids, {feature: [N, D] fp16}) of every volume of a model; cached in `<model_dir>/_globals.pt`."""
    cache = model_dir / "_globals.pt"
    ids = sorted(p.stem for p in model_dir.glob("*.pt") if not p.name.startswith("_"))
    if cache.exists():
        c = torch.load(cache)
        if c["ids"] == ids:
            return c["ids"], c["feats"]
    rows = list(_pmap(_globals_one, [model_dir / f"{i}.pt" for i in ids]))  # no mmap (OOM kill on 18k files)
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
        err = (pred - y).abs().cpu().numpy()
        return {"neg_mae": -mae, "mae": mae, "pearson_r": torch.corrcoef(torch.stack([pred, y]))[0, 1].item(),
                "mae_per_dataset": {d: float(err[m].mean()) for d, m in rows.groupby("dataset").indices.items()}}


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


GRID_PARAMS = 12e6  # ponytail: stacked parameters per L-BFGS (its 100-step history ~ 200x that in fp32, ~10 GB); bigger grids in chunks


def fit_grid(x: torch.Tensor, y: torch.Tensor, kind: Kind, hidden: int, l2s=L2_GRID, seed: int = 0) -> list[Head]:
    """fit() for every L2 of `l2s` at once (user, 24/10: the one-by-one fits, launch-bound, timed the cv5 probes out):
    the heads (same init as fit) are stacked and one L-BFGS minimises the sum of their independent objectives. The shared
    line search makes the optimum slightly different from fit()'s (the external models keep their earlier fit() probes: same test, user 24/10). A non-finite
    head (one diverged L2 poisons the shared steps) -> that chunk is fitted one by one with fit()."""
    def new():
        torch.manual_seed(seed)
        return Head(x.shape[1], kind.outputs(), hidden, getattr(kind, "ordinal", False)).to(x.device)
    per = max(1, int(GRID_PARAMS // sum(p.numel() for p in new().parameters())))
    if len(l2s) > per:
        return [h for i in range(0, len(l2s), per) for h in fit_grid(x, y, kind, hidden, l2s[i:i + per], seed)]
    heads = [new() for _ in l2s]
    params, _ = torch.func.stack_module_state(heads)
    base = copy.deepcopy(heads[0]).to("meta")
    lam = torch.tensor(l2s, device=x.device)
    opt = torch.optim.LBFGS(list(params.values()), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        out = torch.vmap(lambda p: torch.func.functional_call(base, p, (x,)))(params)  # [G, N, out]
        loss = sum(kind.loss(o, y) for o in out) + sum(
            (lam.view(-1, *[1] * (v.ndim - 1)) * v.square()).sum() for v in params.values() if v.ndim > 2)
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        for g, h in enumerate(heads):
            for k, p in h.named_parameters():
                p.copy_(params[k][g])
    heads = [h.eval().requires_grad_(False) for h in heads]
    if not all(torch.isfinite(h(x)).all() for h in heads):
        heads = [fit(x, y, kind, l2, hidden, seed) for l2 in l2s]
    return heads


def subject_ci(stat, subjects, n: int = 1000, seed: int = 0) -> list[float]:
    """95% CI of `stat(row positions)` over resamples of the subjects (`subjects`: one per row; fixed seed -> the
    same resamples for every model, paired). Also the segdec CI (bench/segdec.py)."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(range(len(subjects))).groupby(np.asarray(subjects)).indices
    keys = sorted(groups)
    vals = [stat(np.concatenate([groups[s] for s in rng.choice(keys, len(keys))])) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def bootstrap(kind: Kind, pred, y, rows: pd.DataFrame) -> list[float]:
    """95% CI of kind.key over resamples of test subjects."""
    def stat(idx):
        t = torch.as_tensor(idx, device=pred.device)
        return kind.metrics(pred[t], y[t], rows.iloc[idx])[kind.key]  # nan when degenerate
    return subject_ci(stat, rows["subject"].to_numpy())


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
    cv = out["subjects"]["val"] < CV_MIN_VAL
    out["selection"] = f"cv{CV_FOLDS}" if cv else "val"
    if cv:  # selection split: train + val, folds by subject (shuffled with a fixed seed)
        sel["fit"] = np.concatenate([sel["train"], sel["val"]])
        subj = rows.iloc[sel["fit"]]["subject"].to_numpy()
        order = np.random.default_rng(0).permutation(np.unique(subj))
        fold = pd.Series(np.arange(len(order)) % CV_FOLDS, index=order)[subj].to_numpy()
        y["fit"] = y_all[sel["fit"]].to(device)
    fit_on = "fit" if cv else "train"

    def standardize(x, idx):
        return (x - x[idx].mean(0)) / x[idx].std(0).clamp_min(1e-6)

    for variant, hidden in HIDDEN.items():
        best, cands = None, {}
        for fname, f in feats.items():
            x = f[rid].float().to(device)
            xs = {s: v[sel[s]] for v in (standardize(x, sel[fit_on]),) for s in sel}
            if cv:  # out-of-fold outputs, each fold standardized on its own training part
                oofs = [None] * len(L2_GRID)
                for k in range(CV_FOLDS):
                    tr, te = sel["fit"][fold != k], sel["fit"][fold == k]
                    xk = standardize(x, tr)
                    for j, h in enumerate(fit_grid(xk[tr], y_all[tr].to(device), kind, hidden)):
                        o = h(xk[te])
                        oofs[j] = o.new_zeros((len(sel["fit"]), *o.shape[1:])) if oofs[j] is None else oofs[j]
                        oofs[j][torch.from_numpy(np.flatnonzero(fold == k)).to(device)] = o
            else:
                heads = fit_grid(xs["train"], y["train"], kind, hidden)
            for j, l2 in enumerate(L2_GRID):
                if cv:
                    o = oof = oofs[j]
                    val = kind.metrics(kind.predict(oof), y["fit"], rows.iloc[sel["fit"]])
                else:
                    head = heads[j]
                    o = head(xs["val"])
                    val = kind.metrics(kind.predict(o), y["val"], rows.iloc[sel["val"]])
                if not torch.isfinite(o).all():   # a diverged L-BFGS fit (21/10: NaN scores ranked as ties gave the
                    continue                      # same fake AUROC 0.714 to six models): never a candidate
                cands[fname] = max(cands.get(fname, -float("inf")), val[kind.key])   # best val per feature
                if best is None or val[kind.key] > best["val"][kind.key]:
                    best = {"feature": fname, "l2": l2, "val": val, "xs": xs, "oof": oof if cv else None}
                    if not cv:
                        best["head"] = head
        if cv:  # the chosen (feature, L2) refitted on train + val
            best["head"] = fit(best["xs"]["fit"], y["fit"], kind, best["l2"], hidden)
        pred = {s: kind.predict(best["head"](best["xs"][s])) for s in sel}
        test_rows = rows.iloc[sel["test"]]
        res = {"feature": best["feature"], "l2": best["l2"], "val": best["val"], "candidates": cands,
               "test": kind.metrics(pred["test"], y["test"], test_rows),
               "test_ci95": bootstrap(kind, pred["test"], y["test"], test_rows)}
        if cfg["type"] == "dex":  # age-bias correction (regression to the mean) fitted on val (cv: out-of-fold)
            yv, pv = (y["fit"], kind.predict(best["oof"])) if cv else (y["val"], pred["val"])
            a, b = np.polyfit(yv.cpu().numpy(), pv.cpu().numpy(), 1)
            res["test"]["mae_bias_corrected"] = float(((pred["test"] - b) / a - y["test"]).abs().mean())
        res["predictions"] = {"id": test_rows["id"].tolist(), "pred": pred["test"].cpu().tolist()}
        out[variant] = res
    return out


def run(models: dict[str, Path], tasks: dict[str, dict], vols: pd.DataFrame, splits: pd.DataFrame,
        out_dir: Path, seeds: list[int], device: str = "cuda", common: dict[str, Path] | None = None,
        menu: dict[str, list[str]] | None = None, extra: dict | None = None) -> pd.DataFrame:
    """Every task x seed x model; only the volumes all `common` models (default `models`) have.
    `menu`: per model, the only stored features its probes may choose from (default: all of them).
    Writes `<out>/<task>/s<seed>/<model>.json`."""
    fit = set(vols.loc[vols["subject"].isin(tune_subjects(splits)), "id"])  # second order: PCA from these only
    loaded = {m: readings(m, d, fit, (extra or {}).get(m)) for m, d in models.items()}
    for m, keep in (menu or {}).items():
        if m in loaded:
            ids, feats = loaded[m]
            assert set(keep) <= set(feats), (m, set(keep) - set(feats))
            loaded[m] = ids, {k: feats[k] for k in keep}
    others = [load_globals(d)[0] for m, d in (common or {}).items() if m not in loaded]
    common = set.intersection(*(set(ids) for ids, _ in loaded.values()), *(set(i) for i in others))
    vols = vols[vols["id"].isin(common)]
    summary = []
    for tname, cfg in tasks.items():
        if cfg["type"] == "seg":  # segmentation: bench-segdec (each model's own network, full resolution)
            continue
        for seed in seeds:
            rows = task_rows(vols, splits, cfg, seed)
            for m, (ids, feats) in loaded.items():
                dst = out_dir / tname / f"s{seed}" / f"{m}.json"
                if dst.exists():  # a chained job resumes: the seeds already written are not fitted again
                    res = json.loads(dst.read_text())
                else:
                    res = run_task(tname, cfg, rows, feats, {i: k for k, i in enumerate(ids)}, device)
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


# ------------------------------------------------------------------------------------------- pretraining tuning
def tune_subjects(splits: pd.DataFrame) -> set:
    """Subjects in no seed's test split, the only ones pretraining choices may be scored on: the seeds' splits are
    drawn independently, so a seed-0 val subject is often another seed's test subject."""
    cols = [c for c in splits if c.startswith("split_s")]
    return set(splits.loc[(splits[cols] != "test").all(axis=1), "subject"])


CHANCE = {"auroc": 0.5, "c_index": 0.5, "qwk": 0.0}


def tune_task(name: str, cfg: dict, rows: pd.DataFrame, feats: dict[str, torch.Tensor], index: dict[str, int],
              device: str = "cuda", folds: int = 5, seed: int = 0, keep: bool = False) -> dict:
    """Skill of one task on `rows` (tune_subjects only, no split used): out-of-fold predictions over `folds` subject
    folds for every (head, feature, L2) of the benchmark grid; the candidate is chosen on one half of the subjects and
    scored on the other, and vice versa (cross-fitted: no selection bias), averaged. skill: AUROC and C-index
    2(m - 0.5), QWK as is, balanced accuracy (m - 1/k) / (1 - 1/k); dex: 1 - mean over datasets of the MAE within the
    dataset over that of its out-of-fold train median (the site-constant predictor), so site is no help.
    seed: the subjects' permutation (halves and folds); keep: also return every candidate's skill per half."""
    y, _ = targets(rows, cfg)
    y = y.to(device)
    kind = KINDS[cfg["type"]](cfg, y.cpu())
    subjects = rows["subject"].unique()
    p = rows["subject"].map(dict(zip(subjects, np.random.default_rng(seed).permutation(len(subjects))))).to_numpy()
    half, fold = p % 2, (p // 2) % folds
    ds, yy = rows["dataset"].to_numpy(), y.cpu().numpy()
    if cfg["type"] == "dex":  # out-of-fold train median of the volume's dataset
        base = np.full(len(rows), np.nan)
        for k in range(folds):
            med = pd.Series(yy[fold != k]).groupby(ds[fold != k]).median()
            base[fold == k] = pd.Series(ds[fold == k]).map(med).to_numpy()
        base = np.abs(base - yy)

    def skill(pred, idx):
        if cfg["type"] == "dex":
            err = (pred[idx] - y[idx]).abs().cpu().numpy()
            r = [err[g].mean() / base[idx][g].mean() for g in pd.Series(ds[idx]).groupby(ds[idx]).indices.values()
                 if base[idx][g].mean() > 0]
            return 1 - float(np.mean(r))
        m = kind.metrics(pred[idx], y[idx], rows.iloc[idx])[kind.key]
        lo = 1 / kind.k if kind.key == "balanced_accuracy" else CHANCE[kind.key]
        return (m - lo) / (1 - lo)

    rid = torch.tensor([index[i] for i in rows["id"]])
    halves = [np.flatnonzero(half == h) for h in (0, 1)]
    cands = []
    for variant, hidden in HIDDEN.items():
        for fname, f in feats.items():
            x = f[rid].float().to(device)
            preds = [None] * len(L2_GRID)
            for k in range(folds):
                tr, te = (torch.as_tensor(np.flatnonzero(m), device=device) for m in (fold != k, fold == k))
                mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
                for j, h in enumerate(fit_grid((x[tr] - mu) / sd, y[tr], kind, hidden)):
                    pk = kind.predict(h((x[te] - mu) / sd))
                    preds[j] = pk.new_zeros(len(rows), *pk.shape[1:]) if preds[j] is None else preds[j]
                    preds[j][te] = pk
            for l2, pred in zip(L2_GRID, preds):
                cands.append({"head": variant, "feature": fname, "l2": l2, "skill": [skill(pred, h) for h in halves]})
    a, b = (max(cands, key=lambda c, h=h: c["skill"][h]) for h in (0, 1))
    return {"task": name, "metric": kind.key, "skill": (a["skill"][1] + b["skill"][0]) / 2, "n": len(rows),
            "subjects": len(subjects), "chosen": [{k: c[k] for k in ("head", "feature", "l2")} for c in (a, b)]} \
        | ({"cands": cands} if keep else {})


_BASIS = None


def _init_basis(basis, dim):
    global _BASIS
    torch.set_num_threads(1)
    _BASIS = basis, dim


def _covsqrt_one(path: Path) -> dict:
    basis, dim = _BASIS
    f = torch.load(path, map_location="cpu", weights_only=False)["features"]
    t = f["map"].float().flatten(1).T
    keep = t.norm(dim=1) > 0
    toks = {"covsqrt": t[keep]} | ({"covsqrt_fg": t[keep & f["fg"].flatten()]} if "fg" in f else {})
    iu, out = torch.triu_indices(dim, dim), {}
    for name, tk in toks.items():
        mu, p = basis[name]
        z = (tk - mu) @ p
        z = z - z.mean(0)
        e, v = torch.linalg.eigh(z.T @ z / max(len(z) - 1, 1) + 1e-6 * torch.eye(dim))
        out[name] = ((v * e.clamp(min=0).sqrt()) @ v.T)[iu[0], iu[1]].half()
    return out


def _unpack(v: torch.Tensor) -> torch.Tensor:
    """Upper triangles [..., d (d + 1) / 2] -> symmetric matrices [..., d, d]."""
    d = int((math.isqrt(8 * v.shape[-1] + 1) - 1) // 2)
    i = torch.triu_indices(d, d)
    c = v.new_zeros(*v.shape[:-1], d, d)
    c[..., i[0], i[1]] = v
    return c + c.transpose(-1, -2) - torch.diag_embed(c.diagonal(dim1=-2, dim2=-1))


def _tri_one(path: Path) -> dict:   # every stored covariance (fm/sfc: per level and block), in name order
    f = torch.load(path, map_location="cpu", weights_only=False)["features"]
    return {"C": torch.stack([f[k] for k in sorted(k for k in f if "tricov" in k)]).float()}


def _tri_sqrt_one(path: Path) -> dict:
    P, dim = _BASIS   # [levels x blocks, d, dim]
    z = P.transpose(1, 2) @ _unpack(_tri_one(path)["C"]) @ P + 1e-6 * torch.eye(dim)
    e, v = torch.linalg.eigh(z)
    iu = torch.triu_indices(dim, dim)
    return {"S": ((v * e.clamp(min=0).sqrt()[:, None]) @ v.transpose(1, 2))[:, iu[0], iu[1]].half()}


def second_order_exact(d: Path, ids: list[str], dim: int = 16, fit: set | None = None) -> dict[str, torch.Tensor]:
    """second_order from the covariances stored at extraction (fm/sfc multigrain, 27/10; user: exact, every level): the
    `dim` principal directions per level and block (16, user 27/10: the 16 together ~ the 2080 numbers of the other
    models' one covsqrt per map) = eigenvectors of the mean covariance of the `fit` volumes (exact, every
    foreground token; the within-volume variation the reading describes, not the spread between scans), then per volume
    the square root of P^T C P of each, upper triangle, concatenated -> "covsqrt" (16 / 8 / 4 / 2 mm x the depth
    quartiles). Every covariance of a volume in one batched matmul + eigh, the files read by SFC_IO_WORKERS processes."""
    cand = [d / f"{i}.pt" for i in ids if fit is None or i in fit]
    tot = sum(r["C"] for r in _pmap(_tri_one, cand)) / len(cand)
    P = torch.linalg.eigh(_unpack(tot))[1][..., -dim:].contiguous()
    S = torch.stack([r["S"] for r in _pmap(_tri_sqrt_one, [d / f"{i}.pt" for i in ids], (P, dim))])   # [N, cov, T]
    return {"covsqrt": S.flatten(1)}


def second_order(d: Path, ids: list[str], dim: int = 64, sample: int = 400, fit: set | None = None) -> dict[str, torch.Tensor]:
    """{"covsqrt": [N, dim (dim + 1) / 2], and "covsqrt_fg" when the files hold `fg`}: per volume the square root of
    the covariance of its tokens (`map`: every patch; _fg: foreground patches only) projected on the first `dim`
    principal directions of 200 such tokens from each of `sample` random volumes (no label; iSQRT-COV normalisation,
    upper triangle): how the tokens vary together, which their mean drops (notes/loss_stato.md; age 0.374 -> 0.405
    with the mean at two epochs). One pass over the files for both; cached in <d>/_second_order.pt. `fit`: the ids
    the principal directions may come from (the benchmark: tune_subjects' volumes, no test subject of any seed)."""
    cache = d / "_second_order.pt"
    if cache.exists() and (c := torch.load(cache))["ids"] == ids:
        return c["feats"]
    if "tricov" in torch.load(d / f"{ids[0]}.pt", map_location="cpu", weights_only=False)["features"]:
        feats = second_order_exact(d, ids, fit=fit)   # dim 16: 16 covariances x 136 ~ one 64-dim covsqrt
        torch.save({"ids": ids, "feats": feats}, cache)
        return feats
    g0 = load_globals(d)
    if "covsqrt" in g0[1]:  # computed at extraction (args.cov_basis): nothing to fit or read again
        pos = {i: k for k, i in enumerate(g0[0])}
        return {k: g0[1][k][[pos[i] for i in ids]] for k in ("covsqrt", "covsqrt_fg") if k in g0[1]}

    def tokens(i):  # {"covsqrt": all tokens, "covsqrt_fg": foreground tokens}
        f = torch.load(d / f"{i}.pt", map_location="cpu", weights_only=False)["features"]
        t = f["map"].float().flatten(1).T
        keep = t.norm(dim=1) > 0
        return {"covsqrt": t[keep]} | ({"covsqrt_fg": t[keep & f["fg"].flatten()]} if "fg" in f else {})

    g = torch.Generator().manual_seed(0)
    pool: dict[str, list] = {}
    cand = [i for i in ids if fit is None or i in fit]
    for k in torch.randperm(len(cand), generator=g)[:sample].tolist():
        for name, t in tokens(cand[k]).items():
            pool.setdefault(name, []).append(t[torch.randperm(len(t), generator=g)[:200]])
    basis = {}
    for name, ts in pool.items():
        x = torch.cat(ts)
        mu = x.mean(0)
        basis[name] = mu, torch.linalg.eigh((x - mu).T @ (x - mu) / len(x))[1][:, -dim:]
    out = {name: [] for name in basis}
    for n, r in enumerate(_pmap(_covsqrt_one, [d / f"{i}.pt" for i in ids], (basis, dim))):
        for name, c in r.items():
            out[name].append(c)
        if n % 1000 == 0:
            print(f"[tune] second order {n}/{len(ids)}", flush=True)
    feats = {name: torch.stack(v) for name, v in out.items()}
    torch.save({"ids": ids, "feats": feats}, cache)
    return feats


def _map_keys(d: Path, ids: list[str]) -> list[str]:
    """A model's stored spatial features: maps [C, *spatial] and token sets [T, C] (not the global vectors)."""
    f = torch.load(d / f"{ids[0]}.pt", map_location="cpu", weights_only=False)["features"]
    return sorted(k for k, v in f.items() if isinstance(v, torch.Tensor) and v.is_floating_point() and v.ndim >= 2
                  and math.prod(v.shape[1:] if v.ndim > 2 else v.shape[:1]) > 1)


def _tokens(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    return v if v.ndim == 2 else v.flatten(1).T   # tokens [T, C] / map [C, *spatial] -> [N, C]


def _ext_one(path: Path) -> dict:
    """One file of another model -> per stored map: the per-channel std and the covariance square root (in that map's
    basis) of its tokens (the readings ours had alone, 21/10)."""
    basis, _ = _BASIS
    f = torch.load(path, map_location="cpu", weights_only=False)["features"]
    out = {}
    for k, (mu, p) in basis.items():
        t = _tokens(f[k])
        out[f"std:{k}"] = t.std(0).half() if len(t) > 1 else torch.zeros(t.shape[1]).half()
        z = (t - mu) @ p
        z = z - z.mean(0)
        dk = p.shape[1]
        e, v = torch.linalg.eigh(z.T @ z / max(len(z) - 1, 1) + 1e-6 * torch.eye(dk))
        iu = torch.triu_indices(dk, dk)
        out[f"cov:{k}"] = ((v * e.clamp(min=0).sqrt()) @ v.T)[iu[0], iu[1]].half()
    return out


def second_order_ext(d: Path, ids: list[str], keys: list[str], dim: int = 64, sample: int = 400,
                     fit: set | None = None) -> dict:
    """Another model's second-order readings, as second_order for ours (user, 21/10: the same readings for every
    model): for each stored map, the per-channel std of its tokens and the covariance square root on the first `dim`
    principal directions (fit on `sample` tune volumes, 200 tokens each). One pass; cached in <d>/_second_order_ext.pt."""
    cache = d / "_second_order_ext.pt"
    if cache.exists() and (c := torch.load(cache))["ids"] == ids:
        return c["feats"]
    g = torch.Generator().manual_seed(0)
    pool: dict[str, list] = {}
    cand = [i for i in ids if fit is None or i in fit]
    for k in torch.randperm(len(cand), generator=g)[:sample].tolist():
        f = torch.load(d / f"{cand[k]}.pt", map_location="cpu", weights_only=False)["features"]
        for key in keys:
            t = _tokens(f[key])
            pool.setdefault(key, []).append(t[torch.randperm(len(t), generator=g)[:200]])
    basis = {}
    for key, ts in pool.items():
        x = torch.cat(ts)
        mu = x.mean(0)
        basis[key] = mu, torch.linalg.eigh((x - mu).T @ (x - mu) / len(x))[1][:, -min(dim, x.shape[1]):]
    out: dict[str, list] = {}
    for n, r in enumerate(_pmap(_ext_one, [d / f"{i}.pt" for i in ids], (basis, dim))):
        for name, c in r.items():
            out.setdefault(name, []).append(c)
        if n % 1000 == 0:
            print(f"[probe] second order (ext) {d.name} {n}/{len(ids)}", flush=True)
    feats = {name: torch.stack(v) for name, v in out.items()}
    torch.save({"ids": ids, "feats": feats}, cache)
    return feats


def readings(m: str, d: Path, fit: set | None = None, extra: dict | None = None) -> tuple[list[str], dict[str, torch.Tensor]]:
    """The readings a model's tuning probe chooses from on val, together with head and L2 (user, 16/10: try them all,
    keep the best on val). Ours: the last block's foreground mean (`mean`), the intermediate blocks' (mean_b<k>), all
    of them concatenated (`blocks`), each with the per-channel std, with the covariance square root (`second_order`,
    all patches or foreground only). Another model: its stored features."""
    ids, f = load_globals(d)
    if not m.startswith("sfc_gdn2"):   # its stored features, plus each map's std / covariance (as ours, 21/10)
        keys = [k for k in _map_keys(d, ids) if k in f]   # the maps every volume holds (load_globals' intersection)
        so = second_order_ext(d, ids, keys, fit=fit)
        menu = dict(f)
        for k in keys:
            menu[f"{k}+std"] = torch.cat([f[k], so[f"std:{k}"].float()], 1)
            menu[f"{k}+covsqrt"] = torch.cat([f[k], so[f"cov:{k}"].float()], 1)
        if len(keys) > 1:
            menu["maps+std"] = torch.cat([f[k] for k in keys] + [so[f"std:{k}"].float() for k in keys], 1)
        return ids, menu
    w = f["mean"].shape[1] // 2
    if torch.equal(f["mean"][:, :w], f["mean"][:, w:]):  # one-stream encoder: bi = h ++ h, keep h (half the fits)
        f = {k: v[:, :w] if k.startswith(("mean", "std")) else v for k, v in f.items()}
    for k in [k for k in f if f"g8:{k}" in f]:   # multigrain (fm/sfc.multigrain, 27/10): each statistic of every level
        f[k] = torch.cat([f[k], *(f[f"g{m}:{k}"] for m in (8, 4, 2))], 1)   # (16 / 8 / 4 / 2 mm), concatenated
    so = second_order(d, ids, fit=fit)
    cov = so["covsqrt"]

    def cat(*xs):
        return torch.cat(xs, 1)

    inter = sorted(k[4:] for k in f if k.startswith("mean_b"))  # "_b8", "_b12"
    menu = {"mean": f["mean"], "mean+covsqrt": cat(f["mean"], cov)}
    if inter:
        blocks = cat(*(f["mean" + b] for b in inter), f["mean"])
        menu |= {"mean" + b: f["mean" + b] for b in inter} | {"blocks": blocks, "blocks+covsqrt": cat(blocks, cov)}
    if "std" in f:
        menu |= {"mean+std": cat(f["mean"], f["std"]), "mean+std+covsqrt": cat(f["mean"], f["std"], cov)}
        if inter:
            menu["blocks+std"] = cat(blocks, *(f["std" + b] for b in inter), f["std"])
    if "covsqrt_fg" in so:
        menu["mean+covsqrt_fg"] = cat(f["mean"], so["covsqrt_fg"])
    for c in sorted({k.split("@")[1].split("_b")[0] for k in f if k.startswith("mean@")}):  # one curve, not the average
        menu[f"mean@{c}"] = f[f"mean@{c}"]
        if inter and f"std@{c}" in f:
            menu[f"blocks+std@{c}"] = cat(*(f[f"mean@{c}" + b] for b in inter), f[f"mean@{c}"],
                                          *(f[f"std@{c}" + b] for b in inter), f[f"std@{c}"])
    for mm, dd in (extra or {}).items():  # the same model read at another patch size (probe.extra, 20/10): mean@<mm>mm, ...
        ids2, g = load_globals(Path(dd))
        pos = {i: k for k, i in enumerate(ids2)}
        assert set(ids) <= set(pos), (dd, len(set(ids) - set(pos)))
        g = {k: v[[pos[i] for i in ids]] for k, v in g.items()}
        g = {k: v[:, :w] if k.startswith(("mean", "std")) and v.shape[1] == 2 * w else v for k, v in g.items()}
        menu[f"mean@{mm}mm"] = g["mean"]
        if "covsqrt" in g:
            menu[f"mean+covsqrt@{mm}mm"] = cat(g["mean"], g["covsqrt"])
        if inter and "std" in g:
            menu[f"blocks+std@{mm}mm"] = cat(*(g["mean" + b] for b in inter), g["mean"], *(g["std" + b] for b in inter), g["std"])
            if "covsqrt" in g:
                menu[f"blocks+covsqrt@{mm}mm"] = cat(*(g["mean" + b] for b in inter), g["mean"], g["covsqrt"])
    return ids, menu  # the 2x-reading means (*_tta) were never chosen on p7 (17/10): dropped


def _tune_one(m: str, d: Path, tname: str, cfg: dict, rows: pd.DataFrame, device: str, keep=None) -> dict:
    torch.set_num_threads(2)
    ids, feats = readings(m, d)
    feats = {k: f for k, f in feats.items() if keep is None or k in keep}  # the official menu (bench.yaml probe.menu)
    r = tune_task(tname, cfg, rows, feats, {i: k for k, i in enumerate(ids)}, device)
    print(f"[tune] {m} {tname}: skill {r['skill']:.3f} ({r['subjects']} subjects)", flush=True)
    return {"model": m, **{k: r[k] for k in ("task", "skill", "subjects")}, "chosen": r["chosen"]}


def tune(models: dict[str, Path], tasks: dict[str, dict], vols: pd.DataFrame, splits: pd.DataFrame, out_dir: Path,
         device: str = "cuda", menu: dict | None = None) -> pd.DataFrame:
    """tune_task for every model x non-seg task on the tune_subjects -> <out>/tune.csv (+ the mean skill per model).
    No test subject of any seed is read. The (model, task) pairs run in one process per visible GPU (the L-BFGS fits are
    launch-bound: processes sharing a GPU slow each other down); tune.csv is rewritten as each pair ends."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    keep, jobs, out = tune_subjects(splits), [], []
    for m, d in models.items():
        ids, _ = readings(m, d)  # builds the caches once, before the workers read them
        v = vols[vols["id"].isin(set(ids)) & vols["subject"].isin(keep)]
        jobs += [(m, d, t, cfg, task_rows(v, splits, cfg, 0)) for t, cfg in tasks.items() if cfg["type"] != "seg"]
    gpus = torch.cuda.device_count() if device == "cuda" else 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(max(gpus, 1), len(jobs)), mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(_tune_one, *j, f"cuda:{k % gpus}" if gpus else device, (menu or {}).get(j[0]))
                for k, j in enumerate(jobs)]
        for f in as_completed(futs):
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001 - one degenerate task must not lose the others
                print(f"[tune] failed: {e!r}", flush=True)
                continue
            pd.DataFrame(out).to_csv(out_dir / "tune.csv", index=False)
    df = pd.DataFrame(out, columns=["model", "task", "skill", "subjects", "chosen"]).sort_values(["model", "task"])
    df.to_csv(out_dir / "tune.csv", index=False)
    print(df.groupby("model")["skill"].mean().rename("mean skill").to_string(), flush=True)
    return df
