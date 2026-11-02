"""Inference efficiency of the frozen encoders (01/11, user: Pareto FLOPs vs performance, latency, memory).

Every model on the same volumes (default: 10 test FLAIRs of wmh_seg, evenly spaced over the task's test FLAIRs as
bench-segdec draws them), batch 1, one GPU. All volumes are preprocessed first (the wrapper's own `preprocess`, CPU,
timed: `prep_s`), then 2 warm-up volumes (the first two of the list, run once untimed), then the timed pass:
- `enc_s`: the encoder forward the benchmark's features come from, CUDA-synchronised wall clock, no preprocessing.
  Ours: the 16 mm extraction (model.encoder_extractor, the run's 3 curves, the patch embedding included; 8 / 4 mm
  with args.patch_mm); the volume's upload and patchify are outside. Externals: the forward of their encoder
  module(s) (`ENCODER`), summed over the calls the wrapper's own `features` makes (sliding window + centre crop, as
  bench-extract runs it), timed by forward hooks.
- `feat_s`: the wrapper's whole `features` call (ours: + multigrain 8 / 4 / 2 mm, the probes' reading; externals:
  + stitching / pooling, some on CPU).
- `peak_gb`: torch.cuda.max_memory_allocated during the encoder pass (ours) / the `features` call (externals), weights
  included; `act_gb` = the same minus the memory allocated just before. `feat_peak_gb`: ours' whole `features` call.
- `flops_counter`: torch.utils.flop_counter.FlopCounterMode over the same encoder calls (an untimed pass; FLOPs =
  2 x MAC of matmuls, convolutions, SDPA). It does not see Triton kernels (GDN-2's chunked recurrence and short
  convolutions) nor flash_attn_varlen_func (our attention blocks): for ours also `flops_analytic` (`Analytic`).
- `params_m`: the encoder's parameters (ours: Encoder; externals: the `ENCODER` modules).

    python -m sfc_gdn2.abl.efficiency --out $SCRATCH/sfc_runs/abl/efficiency [--models small,base,...] [--n 10]

Ours runs in this process (sfc-gdn2 env), each external through fm.api.run_in_env (`<this module> <job.json>`).
Out: <out>/<model>.json (summary + per volume), <out>/summary.csv (every json there).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]   # the repo (worktree) holding configs/
RUNS = "/leonardo_scratch/large/userexternal/fcorrent/sfc_runs"
SMALL = f"{RUNS}/tune/p32_heal8/run/lejepa-48e158e8b4/encoder_step002600.pt"
BASE = f"{RUNS}/tune/p28_final/gp_main/lejepa-9270a0b6e4/encoder_step024000.pt"
# ours: name -> (checkpoint, wrapper args, model overrides (random weights when set))
OURS = {"small": (SMALL, {}, None), "base": (BASE, {}, None),
        "large": (SMALL, {}, {"d_model": 2048, "depth": 8, "num_heads": 32, "attn_every": 2}),
        "small_8mm": (SMALL, {"patch_mm": 8}, None), "small_4mm": (SMALL, {"patch_mm": 4}, None)}
EXTERNAL = ["openmind_resencl_mae", "bsf_ukb_pretrain", "openmind_primusm_mae", "nnfoundation_vit",
            # (02/11, user: the rest of the benchmark on the Pareto)
            "nnunet_totalseg_mr_870", "brainiac", "fomo26", "nnfoundation_cnn", "medicalnet_resnet50_23dataset",
            "openmind_resencl_simclr", "bsf_atlas_pretrain"]
# external wrapper -> its encoder modules (what is timed / counted; Primus' decoder head up_projection is not)
# the timed pass runs twice over the volumes: enc_s / feat_s the first time (after the warm-up: a volume of a new
# length may still meet a Triton compile / autotune), enc_s2 / feat_s2 the second (steady state); memory: the second
REPS = (1, 2)
ENCODER = {"openmind": lambda w: [w.network.down_projection, w.network.eva] if w.is_primus else [w.network.encoder],
           "nnfoundation": lambda w: [w.network.down_projection, w.network.eva] if w.is_primus else [w.network.encoder],
           "brainsegfounder": lambda w: [w.swin],
           # the trained network nnU-Net's predictor runs (features' _Replay swap only replays recorded maps)
           "nnunet": lambda w: [w.predictor.network.encoder],
           "brainiac": lambda w: [w.model.backbone],   # MONAI ViT-B/16 at 96^3
           "fomo26": lambda w: [w.model.encoder],      # resenc_unet_b's encoder, called by _encode (late fusion)
           # MedicalNet ResNet-50: the backbone, without the conv_seg head (its forward runs it after layer4)
           "medicalnet": lambda w: [getattr(w.model, n) for n in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
                                                                  "layer3", "layer4")]}
# encoder calls per volume in the benchmark's features() that repeat the same forward: time and FLOPs divided by it
# (BrainIAC's features runs backbone(x) and model(x) = backbone(x)[0][:, 0], two identical forwards)
REPEAT = {"brainiac": 2}


def volumes(task: str, n: int) -> list[str]:
    from .tokens import task_items

    flair = [i["image"] for i in task_items(task) if i["split"] == "test" and i["id"].endswith("FLAIR")]
    return [flair[j * len(flair) // n] for j in range(n)]


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def q(v: list, p: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def summary(name: str, rows: list, params: int, extra: dict) -> dict:
    keys = ("prep_s", "enc_s", "feat_s", "enc_s2", "feat_s2", "peak_gb", "act_gb", "feat_peak_gb", "flops_counter", "flops_analytic",
            "flops_analytic_feat", "tokens")
    s = {"model": name, "params_m": params / 1e6, "volumes": len(rows)} | extra
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            s[f"{k}_median"] = statistics.median(v)
            if k.endswith(("_s", "_s2")):
                s[f"{k}_p90"] = q(v, 0.9)
            if k.startswith("flops") or k == "tokens":
                s[f"{k}_mean"] = statistics.fmean(v)
            if k.endswith("_gb"):
                s[f"{k}_max"] = max(v)
    return s


# ---------------------------------------------------------------------------------------------------- externals
class Hooks:
    """Forward hooks on the encoder modules: CUDA-synchronised time, or (count) FlopCounterMode, per call."""

    def __init__(self, mods: list):
        self.count, self.t, self.flops, self.calls = False, 0.0, 0, 0
        for m in mods:
            m.register_forward_pre_hook(self.pre)
            m.register_forward_hook(self.post)

    def pre(self, m, i):
        if self.count:
            from torch.utils.flop_counter import FlopCounterMode
            self.cm = FlopCounterMode(display=False)
            self.cm.__enter__()
        else:
            self.t0 = sync_time()

    def post(self, m, i, o):
        if self.count:
            self.cm.__exit__(None, None, None)
            self.flops += self.cm.get_total_flops()
        else:
            self.t += sync_time() - self.t0
        self.calls += 1


def external(job: dict) -> None:
    """In the model's env: job = {cfg, name, images, warmup, out}."""
    from ..fm.worker import build

    cfg, images = job["cfg"], job["images"]
    w = build(cfg, "cuda")
    mods = ENCODER[cfg["model"]](w)
    hooks = Hooks(mods)
    params = sum(p.numel() for m in mods for p in m.parameters())
    preps, rows = [], []
    for im in images:
        t = time.perf_counter()
        preps.append(w.preprocess(im))
        rows.append({"image": im, "prep_s": time.perf_counter() - t})
    for p in preps[: job["warmup"]]:
        w.features(p)
    for p, r in [(p, r) for rep in REPS for p, r in zip(preps, rows)]:   # (see REPS)
        sfx = "2" if "enc_s" in r else ""
        hooks.t, hooks.calls = 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t = sync_time()
        out = w.features(p)
        r["feat_s" + sfx] = sync_time() - t
        r["enc_s" + sfx], r["calls"] = hooks.t / REPEAT.get(cfg["model"], 1), hooks.calls
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        r["act_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
        r["meta"] = {k: out["meta"].get(k) for k in ("input_shape", "preprocessed_shape", "padded_shape", "patch_size",
                                                      "precision", "inference") if k in out["meta"]}
        del out
    hooks.count = True
    try:
        for p, r in zip(preps, rows):
            hooks.flops = 0
            w.features(p)
            r["flops_counter"] = hooks.flops / REPEAT.get(cfg["model"], 1)
    except Exception as e:   # noqa: BLE001 (an env whose torch lacks a usable flop counter: latency / memory only)
        print(f"{job['name']}: FlopCounterMode failed: {e!r}", flush=True)
    s = summary(job["name"], rows, params, {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                            "encoder_modules": [type(m).__name__ for m in mods],
                                            "repeat_divisor": REPEAT.get(cfg["model"], 1)})
    Path(job["out"]).write_text(json.dumps({"summary": s, "volumes": rows}, indent=1, default=str))
    print(json.dumps(s), flush=True)


# ---------------------------------------------------------------------------------------------------------- ours
class Analytic:
    """Our encoder's forward FLOPs from its shapes, per `Encoder.flat` call (n sequences of lengths L_i, T = sum L_i):
        sum over blocks of 2 * M_b * T                     (M_b: the block's matrix weights each token touches: every
                                                           nn.Linear weight, the depthwise short convs' and the CPE
                                                           TokenConv3D's 27 d before attention blocks)
      + per attention block  sum_i 4 * L_i^2 * d           (QK^T and AV, d = heads x head_dim, full non-causal)
      + per GDN-2 block      4 * d_k * d_v * H * T         (state update k v^T and readout q S per head and token)
    + per patch embedding call 2 * n * P * d (n patches of P voxels). Norms, gates, activations, the curve sort: not
    counted. The chunked GDN-2 kernel does more work than the recurrence (intra-chunk terms): a lower bound."""

    def __init__(self, enc):
        from ..model import AttnBlock

        self.on, self.flops, self.tokens = False, 0, 0
        lin = (torch.nn.Linear, torch.nn.Conv1d)
        self.blocks = []
        for i, b in enumerate(enc.blocks):
            m = sum(x.weight.numel() for x in b.modules() if isinstance(x, lin))
            if enc.cpe is not None and str(i) in enc.cpe:
                m += enc.cpe[str(i)].w.numel()
            gdn = None if isinstance(b, AttnBlock) else 4 * b.mix.head_k_dim * b.mix.head_v_dim * b.mix.num_heads
            self.blocks.append((m, gdn))
        self.d = enc.dim
        flat = enc.flat

        def logged(x, cu, *a, **k):
            if self.on:
                L = (cu[1:] - cu[:-1]).long()
                T = int(L.sum())
                self.tokens += T
                for m, gdn in self.blocks[: enc.depth]:
                    self.flops += 2 * m * T + (gdn * T if gdn else 4 * self.d * int((L * L).sum()))
            return flat(x, cu, *a, **k)
        enc.flat = logged
        enc.patch_embed.register_forward_hook(self.embed)

    def embed(self, m, i, o):
        if self.on:
            self.flops += 2 * i[0].shape[0] * i[0].shape[1] * o.shape[-1]


def ours(name: str, images: list, warmup: int, out: Path) -> None:
    from torch.utils.flop_counter import FlopCounterMode

    from ..data.volume import patchify
    from ..fm.api import load_config
    from ..fm.sfc import SFC

    ckpt, args, model = OURS[name]
    cfg = load_config(ROOT / "configs/fm/sfc_gdn2.yaml") | {"checkpoint": ckpt, "args": dict(args)}
    w = SFC(cfg, "cuda")
    if model:   # Large: Small's config at another width / depth, random weights (timing only)
        w.run_cfg["model"] |= model
        os.environ["SFC_ENC_RANDOM"] = "1"
    try:
        extract = w.extractor()
    finally:
        os.environ.pop("SFC_ENC_RANDOM", None)
    enc = w._extract_enc
    params = sum(p.numel() for p in enc.parameters())
    tally = Analytic(enc)

    def inputs(p):   # the volume on the GPU, patchified (outside the timed call, as the externals' upload)
        enc.patch_embed.patch_mm = p["meta"]["patch_mm"]
        a, c = p["value"]
        x = patchify(p["volume"].cuda().float() * a + c, p["k"].tolist())
        return {"patches": [x], "grid": p["grid"][None].cuda(), "k": p["k"][None].cuda(),
                "spacing": p["spacing"][None].cuda()}

    preps, rows = [], []
    for im in images:
        t = time.perf_counter()
        preps.append(w.preprocess(im))
        g = preps[-1]["grid"].tolist()
        rows.append({"image": im, "prep_s": time.perf_counter() - t, "grid": g,
                     "spacing": list(preps[-1]["meta"]["spacing"]), "shape": list(preps[-1]["meta"]["input_shape"])})
    with torch.no_grad():
        for p in preps[:warmup]:
            extract(inputs(p))
            w.features(p)
        for p, r in [(p, r) for rep in REPS for p, r in zip(preps, rows)]:   # (see REPS)
            sfx = "2" if "enc_s" in r else ""
            v = inputs(p)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t = sync_time()
            o = extract(v)
            r["enc_s" + sfx] = sync_time() - t
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            r["act_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            del o, v
            torch.cuda.reset_peak_memory_stats()
            t = sync_time()
            w.features(p)
            r["feat_s" + sfx] = sync_time() - t
            r["feat_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        for p, r in zip(preps, rows):
            tally.on, tally.flops, tally.tokens = True, 0, 0
            with FlopCounterMode(display=False) as cm:
                extract(inputs(p))
            r["flops_counter"], r["flops_analytic"], r["tokens"] = cm.get_total_flops(), tally.flops, tally.tokens
            tally.flops, tally.tokens = 0, 0
            w.features(p)
            r["flops_analytic_feat"], r["tokens_feat"] = tally.flops, tally.tokens
            tally.on = False
    m = w.run_cfg["model"]
    arch = {k: m.get(k) for k in ("d_model", "depth", "num_heads", "head_dim", "attn_every", "bidir")}
    s = summary(name, rows, params, {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                     "checkpoint": None if model else ckpt, "random_weights": bool(model),
                                     "patch_mm": w.run_cfg["data"]["patch_mm"], "curves": w.curves, "arch": arch,
                                     "multigrain_in_feat": not args.get("patch_mm")})
    (out / f"{name}.json").write_text(json.dumps({"summary": s, "volumes": rows}, indent=1, default=str))
    print(json.dumps(s), flush=True)


def write_csv(out: Path) -> None:
    import csv

    rows = [json.loads(f.read_text())["summary"] for f in sorted(out.glob("*.json"))]
    cols = list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k], (dict, list))))
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    import argparse

    from ..fm.api import load_config, run_in_env

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join([*OURS, *EXTERNAL]))
    ap.add_argument("--task", default="wmh_seg")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = volumes(a.task, a.n)
    (out / "volumes.txt").write_text("\n".join(images) + "\n")
    for name in a.models.split(","):
        t = time.perf_counter()
        try:
            if name in OURS:
                ours(name, images, a.warmup, out)
            else:
                cfg = load_config(ROOT / f"configs/fm/{name}.yaml")
                run_in_env(cfg["env"], "sfc_gdn2.abl.efficiency",
                           {"cfg": cfg, "name": name, "images": images, "warmup": a.warmup,
                            "out": str(out / f"{name}.json")})
        except Exception as e:   # noqa: BLE001 (one model failing: the others still run)
            print(f"{name}: FAILED {e!r}", flush=True)
        print(f"{name}: {time.perf_counter() - t:.0f} s", flush=True)
        torch.cuda.empty_cache()
    write_csv(out)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        external(json.loads(Path(sys.argv[1]).read_text()))
    else:
        main()
