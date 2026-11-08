"""Our encoder (LeJEPA, causal GDN-2) as a benchmark model: the same `<out>/<id>.pt` as the external FMs, so
bench-extract / bench-probe / bench-segdec treat it like any of them. Runs in our own env (sfc-gdn2), `repo` = this repo.

- preprocessing: the pretraining one (data/volume.py): canonical orientation (axis permutation / flips only), native
  voxels, values / 99th percentile of the voxels > 0, patches of round(patch_mm / spacing) voxels per axis.
- features: model.encoder_extractor (the read of every evaluation): the whole patch grid along each curve of
  `args.curves` (default: the run's view_curves, averaged), forward + reversed. `map` = bi tokens on the patch grid
  [2d, gx, gy, gz]; `mean` = bi averaged over the foreground patches (the pyramid top, what volume probes read:
  canonical). Below it (multigrain, 27/10: the only reading under a head; args.multigrain false: off) 8 mm dense and
  4 / 2 mm in the most varied cells, g<mm>:mean / std per block, concatenated by bench.probe (see `multigrain`).
- meta: `source_to_input` (source voxel index -> canonical voxel index), returned by `seg_input`.
- segmentation (bench-segdec, fm/segrun.py): the frozen encoder + our decoder AGD (fm/hier_dec.py, args.hier_dec /
  args.hier_unet; Dmix, the earlier `SegNet` cascade, removed 02/11: it is in git before that date).
  Input: the native canonical grid (`seg_input`), the spacing as `seg_context`; training tiles from the token cache's
  16 mm grid (SEG_PATCH voxels without one), whole scans at val / test.

`checkpoint`: an `encoder_step*.pt` (its run's config.yaml next to it), a run directory (its last checkpoint, or
`args.step`), or an output_root holding exactly one run.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel.orientations as nio
import numpy as np
import torch

from .base import Image, Wrapper
from .token_cache import fine_ok

SEG_PATCH = (160, 160, 160)   # voxels, as the Primus / nnFoundation ViT runs
MIX_MAX = 1 << 19   # cells per multigrain level at most (more: the level above's)

_DEV = {}


def _dt(vals, dev, dtype=torch.long) -> torch.Tensor:
    """A small constant (cell dims, grid, offsets) on the device, made once: torch.tensor(list, device=cuda) is a
    synchronous pageable copy that drains the GPU queue (27/10 profile: ~115 per tile per iteration, ~25 % of the GPU
    time of the old Dmix decoder's step)."""
    key = (tuple(vals) if not isinstance(vals, torch.Tensor) else tuple(vals.tolist()), str(dev), dtype)
    if (t := _DEV.get(key)) is None:
        t = _DEV[key] = torch.tensor(key[0], device=dev, dtype=dtype)
    return t


def _groups(q, gf) -> torch.Tensor:
    """Cells q [n, 3] on grid gf -> the label of each one's 26-connected group (0 ..), groups in raster order."""
    from scipy import ndimage
    g = [int(a) for a in gf.tolist()]
    m = np.zeros(g, dtype=bool)
    qc = q.cpu().numpy()
    m[qc[:, 0], qc[:, 1], qc[:, 2]] = True
    lab = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))[0]
    return torch.as_tensor(lab[qc[:, 0], qc[:, 1], qc[:, 2]].astype(np.int64) - 1, device=q.device)



FINE = tuple(float(g) for g in os.environ.get("SFC_FINE", "8,4,2").split(","))   # multigrain levels below the 16 mm extraction (02/11: SFC_FINE=8 for the 16+8 reading)


def tricov(t: torch.Tensor) -> torch.Tensor:
    """Tokens [n, d] -> their covariance (population), upper triangle fp16: bench.probe.second_order's exact input."""
    t = t.float() - t.float().mean(0)
    c = t.T @ t / max(len(t), 1)
    i = torch.triu_indices(len(c), len(c), device=c.device)
    return c[i[0], i[1]].half()


@torch.no_grad()
def multigrain(enc, curves: list, seed: int, vol: torch.Tensor, sp: torch.Tensor, thr: float, above: dict,
               keep: dict | None = None) -> dict:
    """The features every head reads on our model (user, 27/10; segmentation decoders keep their own reading): below the
    16 mm extraction, 8 mm on every cell, then 4 and 2 mm only under the cells of the level above whose voxels vary most
    (intensity std of the foreground cells: label-free, the same for every task), as many cells as the level above has
    foreground cells (each level costs about the 8 mm one). The cells read as one sequence per 26-connected group along each curve
    (averaged), as the cascade's live reading. Per level the foreground mean and std of each block (last ++ taps, one
    copy of h) -> g<mm>:mean, g<mm>:std_b<k>, g<mm>:tricov_b<k>, ... (covariance); a level the scan cannot
    hold (cells under 8 voxels, nothing left) repeats the level above (`above`: the 16 mm ones, same names).
    bench.probe concatenates the levels."""
    return multigrain_batch(enc, curves, seed, [vol], [sp], thr, [above], [keep])[0]


def multigrain_batch(enc, curves: list, seed: int, vols: list, sps: list, thr: float, aboves: list, keeps: list) -> list:
    """`multigrain` of several volumes (02/11): each level reads the cells of every volume in one encoder call per curve
    (their groups packed back to back, cu_seqlens), the same per volume as one by one."""
    from ..curves import grid_coords, keys
    from ..data.volume import patch_dims, patch_grid, patchify

    assert enc.bidir != "late", "multigrain: one-stream encoders (bi = h ++ h)"
    dev, d, pe, mm0 = vols[0].device, enc.dim, enc.patch_embed, enc.patch_embed.patch_mm
    names = ["", *(f"_b{t + 1}" for t in enc.taps)]
    B = len(vols)
    outs = [{} for _ in range(B)]
    st = [{"prev": None, "on": True} for _ in range(B)]   # prev: the level above's cell scores (-inf: not a parent)
    for mm in FINE:
        reads = []   # (volume, x [cells, d], q, g, lab, fg, p, sel, k)
        for i, (vol, sp) in enumerate(zip(vols, sps)):
            if not st[i]["on"]:
                continue
            k = patch_dims(sp.tolist(), mm)
            g = _dt(patch_grid(vol.shape, k).tolist(), dev)
            p = patchify(vol, k)
            xyz = grid_coords(g[None], len(p))[0]
            prev = st[i]["prev"]
            if prev is None:   # 8 mm: every cell
                sel = torch.arange(len(p), device=dev)
            else:              # the children of the top parents, at most `budget` cells
                kp, gp = st[i]["kp"], st[i]["gp"]
                par = torch.minimum(((xyz + 0.5) * _dt(k, dev) / _dt(kp, dev)).long(), gp - 1)
                pidx = (par[:, 0] * gp[1] + par[:, 1]) * gp[2] + par[:, 2]
                order = prev.argsort(descending=True)
                n = torch.bincount(pidx, minlength=len(prev))[order].cumsum(0)
                top = order[(n <= st[i]["budget"]) & (prev[order] > -torch.inf)]
                sel = torch.isin(pidx, top).nonzero()[:, 0]
            if not fine_ok(k) or not len(sel) or len(sel) > MIX_MAX:   # (whole-body scans at 8 mm: the level above's)
                st[i]["on"] = False
                continue
            q, fg = xyz[sel], p[sel].mean(1) > thr
            lab = _groups(q, g) if len(sel) < len(p) else torch.zeros(len(sel), dtype=torch.long, device=dev)
            pe.patch_mm = mm
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = pe(p[sel], pe.weights(sp, k))
            reads.append((i, x, q, g, lab, fg, p, sel, k))
        if not reads:
            break
        hs = [0] * len(reads)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for c in curves:   # every volume's groups in one call: groups back to back, volume after volume
                xs, ps, cus, os_, base = [], [], [], [], 0
                for _, x, q, g, lab, *_ in reads:
                    kk = keys(c, q, g, seed)
                    o = (lab * (int(kk.max()) + 1) + kk).argsort()
                    xs.append(x[o]), ps.append(q[o]), os_.append(o)
                    cus.append(torch.bincount(lab).cumsum(0) + base)
                    base += len(x)
                cu = torch.cat([cus[0].new_zeros(1), *cus]).int()
                h = enc.flat(torch.cat(xs), cu, pos=torch.cat(ps)).float()
                for r, (o, hh) in enumerate(zip(os_, h.split([len(x) for x in xs]))):
                    hs[r] = hs[r] + hh[o.argsort()]
        pe.patch_mm = mm0
        for (i, x, q, g, lab, fg, p, sel, k), h in zip(reads, hs):
            nfg, fg = int(fg.sum()), fg if fg.any() else torch.ones_like(fg)
            if keeps[i] is not None:   # each level's cells read and their tokens (fm/dense_dec's input until 02/11)
                keeps[i][mm] = (q.short().cpu(), (h / len(curves)).half().cpu(), list(k))
            h = h[fg] / len(curves)
            for j, nm in enumerate(names):
                s_, m = torch.std_mean(h[:, j * d: (j + 1) * d], 0, correction=0)
                outs[i][f"g{int(mm)}:mean{nm}"], outs[i][f"g{int(mm)}:std{nm}"] = m.half(), s_.half()
                outs[i][f"g{int(mm)}:tricov{nm}"] = tricov(h[:, j * d: (j + 1) * d])
            prev = torch.full((len(p),), -torch.inf, device=dev)
            prev[sel[fg]] = p[sel[fg]].float().std(1)
            st[i] |= {"prev": prev, "budget": max(nfg, 1), "kp": k, "gp": g}
    for out, above in zip(outs, aboves):
        for mm in FINE:   # the levels not read: the level above's
            for nm, v in above.items():
                if f"g{int(mm)}:{nm}" not in out:
                    out[f"g{int(mm)}:{nm}"] = out.get(f"g{int(mm) * 2}:{nm}", v)
    return outs


def checkpoint_path(path: str | Path, step: int | None = None) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not (p / "config.yaml").exists():
        runs = sorted(p.glob("lejepa-*"))
        if len(runs) != 1:
            raise FileNotFoundError(f"sfc: {p} holds {len(runs)} runs; point `checkpoint` at one")
        p = runs[0]
    if step is not None:
        return p / f"encoder_step{int(step):06d}.pt"
    ckpts = sorted(p.glob("encoder_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"sfc: no encoder_step*.pt in {p}")
    return ckpts[-1]


class SFC(Wrapper):
    """args: curves (list, default the run's view_curves), step (int, default the last checkpoint), crop (read only
    the foreground's bounding box, as the training's global views), encoder: none (the patch embedding alone)."""

    name = "sfc_gdn2"

    def __init__(self, cfg: dict, device: str):
        super().__init__(cfg, device)
        from ..io import load_yaml

        self.ckpt = checkpoint_path(cfg["checkpoint"], self.args.get("step"))
        self.run_cfg = load_yaml(self.ckpt.parent / "config.yaml")
        if self.args.get("patch_mm"):  # read at another patch size (18/10: a model trained at several resolutions)
            self.run_cfg["data"]["patch_mm"] = float(self.args["patch_mm"])
        self.curves = list(self.args.get("curves") or self.run_cfg["objective"]["view_curves"])
        self._extract = None  # built on first use: preprocessing workers never load GDN-2

    def preprocess(self, image: Image) -> dict:
        if not isinstance(image, str):
            raise TypeError("sfc: single-modality, pass a NIfTI path")
        from ..data.volume import decode, open_image, patch_dims, patch_grid

        img = open_image(image)
        arr, spacing, (a, c) = decode(img)
        mm = self.run_cfg["data"]["patch_mm"]
        k = patch_dims(spacing, mm)
        # args.max_tokens / args.fallback_mm (20/10): a volume whose grid would exceed max_tokens is read at fallback_mm
        # (8 mm whole-body scans: ~10^6 tokens, ~80 GB for the encoder alone)
        if self.args.get("max_tokens") and np.prod(patch_grid(arr.shape, k).tolist()) > int(self.args["max_tokens"]):
            mm = float(self.args["fallback_mm"])
            k = patch_dims(spacing, mm)
        ornt = nio.io_orientation(img.affine)
        src = np.linalg.inv(nio.inv_ornt_aff(ornt, img.shape[:3]))  # source voxel -> canonical voxel
        meta = {"source_affine": torch.from_numpy(img.affine.copy()), "source_shape": tuple(img.shape[:3]),
                "source_to_input": torch.from_numpy(src), "input_shape": tuple(arr.shape),
                "spacing": tuple(spacing), "patch_mm": mm}
        return {"volume": torch.from_numpy(np.ascontiguousarray(arr)), "value": (a, c), "k": torch.tensor(k),
                "grid": patch_grid(arr.shape, k), "spacing": torch.tensor(spacing, dtype=torch.float32), "meta": meta}

    # ------------------------------------------------------------------ segmentation (fm/segrun.py)
    def seg_input(self, prepared: dict):
        a, c = prepared["value"]
        return (prepared["volume"].float() * a + c)[None], prepared["meta"]["source_to_input"].numpy()

    seg_extent = True   # segrun appends the scan's shape to the context (ctx layout read by fm/hier_dec)

    def seg_context(self, prepared: dict) -> torch.Tensor:
        return prepared["spacing"]

    def _encoder(self, pretrained: bool = True):
        from ..model import Encoder

        pretrained = pretrained and not os.environ.get("SFC_ENC_RANDOM")   # bench-extract's random encoder (probes)
        if seed := os.environ.get("SFC_ENC_SEED"):   # (01/11, user) the random-encoder seeds: only the encoder's draw
            # changes; built under a forked RNG in both builds, so the decoder's init and the data stay those of the job seed
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                enc = Encoder.from_config(self.run_cfg)
        else:
            enc = Encoder.from_config(self.run_cfg)
        if pretrained:
            enc.load_state_dict(torch.load(self.ckpt, map_location="cpu"))
        return enc

    def seg_net(self, n_out: int, pretrained: bool = True):
        enc = self._encoder(pretrained)
        taps = self.args.get("taps") or ()  # also these blocks' outputs (layer-normed) to the decoder; an int k: the k
        if isinstance(taps, int):           # quarter points below the last block (2: n/2, 3n/4; 3: also n/4; 4: also
            taps = [len(enc.blocks) * i // 4 - 1 for i in range(4 - taps, 4)]  # the patch embedding, block -1)
        enc.taps = tuple(taps)
        a = self.args
        # fm/hier_dec (29/10): Dmix gerarchico + kernel continuo; hier_unet (30/10); align_unet (08/11)
        if a.get("hier_dec") or a.get("hier_unet") or a.get("align_unet"):
            from .hier_dec import AlignUNet, HierDec, HierUNet
            # the 16 / 8 mm tokens read along one curve, hilbert (04/11, default; ablation C: one curve reads as the
            # mean of three, a third of the cost); args.token_curves for others (the table's runs: the view curves)
            tc = list(a.get("token_curves") or ["hilbert"])
            net = (AlignUNet if a.get("align_unet") else HierUNet if a.get("hier_unet") else HierDec)(enc, str(a["token_cache"]), n_out, tc,
                                                                 int(self.run_cfg["seed"]))
            net.cache.sig = {"checkpoint": str(self.ckpt) if pretrained else f"random{os.environ.get('SFC_ENC_SEED', '')}", "taps": list(enc.taps),
                             "curves": tc, "blocks": None}
            return net, ["encoder"], SEG_PATCH
        raise ValueError("sfc segmentation: args.hier_dec / args.hier_unet (fm/hier_dec.py); Dmix (SegNet) and the dense "
                         "decoders (fm/dense_dec.py) were removed on 02/11 (in git before that date)")

    # ------------------------------------------------------------------ fine-tuned classification (fm/clsrun.py)
    def cls_input(self, prepared: dict) -> tuple:
        return prepared, None   # the extraction's own input (fm/clsrun stores it as is)

    def cls_net(self, pretrained: bool = True, feature: str = "mean"):
        """The table's probe feature (bench.probe.readings: `mean` or `blocks+std`, multigrain levels concatenated) read
        by the extraction's own code (features_batch: 3 curves, depth quartiles, adaptive finer levels) with gradient
        (04/11, P2: the fine-tuning starts from the table's frozen probe)."""
        assert pretrained, "cls_net: the pretrained encoder"
        self.extractor()
        self._extract_enc.requires_grad_(True)
        self._extract_enc.grad_checkpoint = True
        return TableReader(self, feature), ["encoder"]

    def extractor(self):
        if self._extract is None:
            from ..model import encoder_extractor

            gone = [a for a in ("tta", "cov_basis", "per_curve") if a in self.args]   # removed 27/10 (no probe read them)
            assert not gone, f"sfc extraction: args removed in the 27/10 cleanup: {gone}"
            enc = self._encoder().to(self.device).eval().requires_grad_(False)
            if self.args.get("encoder") != "none":  # also the depth quartiles n/4, n/2, 3n/4 (probe menu: mean_b<block>;
                n = len(enc.blocks)                  # 27/10, the rule of every token model in the benchmark)
                enc.taps = tuple(self.args.get("probe_taps") or (n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1))
            self._taps, self._dim = enc.taps, enc.dim
            obj = self.run_cfg["objective"]
            self._extract = encoder_extractor(enc, self.curves, "both", int(self.run_cfg["seed"]),
                                              obj.get("fg_threshold", 0.05), crop=bool(self.args.get("crop")),
                                              embed_only=self.args.get("encoder") == "none")
            self._extract_enc = enc
        return self._extract

    @torch.no_grad()
    def features(self, prepared: dict) -> dict:
        return self.features_batch([prepared])[0]

    @torch.no_grad()
    def features_batch(self, preps: list) -> list:
        return self._features(preps)

    def _features(self, preps: list) -> list:
        """features_batch without no_grad (04/11: fine-tuning reads the table's features with gradient, cls_net).
        (02/11) several volumes in one encoder call: their 16 mm tokens packed (encoder_extractor, any grid / spacing,
        no resampling); the same features as one by one. One patch size per call (else one by one)."""
        from ..data.volume import patchify

        dev = torch.device(self.device)
        self.extractor()   # built (and the encoder) before its kernel takes this volume's patch size
        pe = self._extract_enc.patch_embed
        mms = {p["meta"].get("patch_mm", pe.patch_mm) for p in preps}
        if len(mms) > 1:
            return [r for p in preps for r in self.features_batch([p])]
        pe.patch_mm = mms.pop()
        vols, xs = [], []
        for p in preps:
            a, c = p["value"]
            vols.append(p["volume"].to(dev).float() * a + c)
            xs.append(patchify(vols[-1], p["k"].tolist()))
        out = self.extractor()({"patches": xs, "grid": torch.stack([p["grid"] for p in preps]).to(dev),
                                "k": torch.stack([p["k"] for p in preps]).to(dev),
                                "spacing": torch.stack([p["spacing"] for p in preps]).to(dev)})
        d, n = self._dim, len(self._taps) + 1

        def block(v, j):  # [..., 2dn] = fwd (last block, taps) ++ bwd (same) -> block j's fwd ++ bwd [..., 2d]
            return torch.cat([v[..., j * d: (j + 1) * d], v[..., (n + j) * d: (n + j + 1) * d]], -1)

        res, fine = [], []
        for i, (prepared, vol, x) in enumerate(zip(preps, vols, xs)):
            grid = prepared["grid"]
            bi, g = block(out["bi"][i, : len(x)], 0), grid.tolist()
            feats = {"map": bi.T.reshape(bi.shape[1], *g), "fg": out["fg"][i, : len(x)].view(*g)}  # last block's tokens
            for stat in ("mean", "std"):  # per block: last (no suffix), taps (_b<block>)
                feats |= {stat + ("" if j == 0 else f"_b{self._taps[j - 1] + 1}"): block(out[stat][i], j) for j in range(n)}
            # multigrain (27/10): the head features below 16 mm; off for another patch size (args.patch_mm) or encoder none
            if self.args.get("multigrain", True) and not self.args.get("patch_mm") and self.args.get("encoder") != "none":
                above = {s: v[: self._dim].half() for s, v in feats.items() if s.startswith(("mean", "std"))}
                fgk = out["fg"][i, : len(x)]   # 16 mm covariances per block, foreground tokens (exact: bench.probe.second_order)
                fgk = fgk if fgk.any() else torch.ones_like(fgk)
                for j in range(n):
                    nm = "" if j == 0 else f"_b{self._taps[j - 1] + 1}"
                    feats[f"tricov{nm}"] = above[f"tricov{nm}"] = tricov(block(out["bi"][i, : len(x)], j)[fgk, : self._dim])
                if pe.patch_mm == self.run_cfg["data"]["patch_mm"]:
                    fine.append((i, vol, prepared["spacing"].to(dev), above))
                else:   # read at fallback_mm (whole-body scans): every level repeats it
                    feats |= {f"g{int(m)}:{s}": v for m in FINE for s, v in above.items()}
            res.append({"features": feats, "canonical": "mean", "derived": [],
                        "meta": prepared["meta"] | {"curves": self.curves, "checkpoint": str(self.ckpt)}})
        if fine:   # the finer levels of every volume of the batch, packed per level
            mg = multigrain_batch(self._extract_enc, self.curves, int(self.run_cfg["seed"]), [f[1] for f in fine],
                                  [f[2] for f in fine], self.run_cfg["objective"].get("fg_threshold", 0.05),
                                  [f[3] for f in fine], [None] * len(fine))
            for (i, *_), m in zip(fine, mg):
                res[i]["features"] |= m
        return res

    def provenance(self) -> dict:
        return super().provenance() | {"checkpoint": str(self.ckpt), "torch": torch.__version__,
                                       "encoder_random": bool(os.environ.get("SFC_ENC_RANDOM")),
                                       "encoder_seed": os.environ.get("SFC_ENC_SEED")}


build = SFC


class TableReader(torch.nn.Module):
    """prepared -> the table probe's feature (bench.probe.readings, one volume), with gradient: the wrapper's extraction."""

    def __init__(self, w, feature: str):
        super().__init__()
        self.w, self.encoder, self.feature = w, w._extract_enc, feature   # the encoder: the trainable part

    def forward(self, prepared: dict, ctx=None) -> torch.Tensor:
        from ..bench.probe import _grains
        f = dict(self.w._features([prepared])[0]["features"])
        w = f["mean"].shape[-1] // 2
        if torch.equal(f["mean"][:w], f["mean"][w:]):   # one-stream encoder: bi = h ++ h, keep h (readings)
            f = {k: v[:w] if k.startswith(("mean", "std")) else v for k, v in f.items()}
        g = _grains()
        for k in [k for k in f if f"g8:{k}" in f]:   # each statistic's multigrain levels concatenated (readings)
            f[k] = torch.cat([f[k] if m == 16 else f[f"g{m}:{k}"] for m in (16, 8, 4, 2) if m in g], -1)
        if self.feature == "mean":
            return f["mean"].float()
        assert self.feature == "blocks+std", self.feature   # (blocks+covsqrt: not needed by P2's chosen heads)
        inter = sorted((k[4:] for k in f if k.startswith("mean_b")), key=lambda b: int(b[2:]))
        return torch.cat([*(f["mean" + b] for b in inter), f["mean"], *(f["std" + b] for b in inter), f["std"]],
                         -1).float()
