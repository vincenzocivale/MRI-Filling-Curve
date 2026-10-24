#!/bin/bash
# Full benchmark of one encoder checkpoint in ONE 4-GPU job (user, 24/10: no chain of ~150 small jobs; as sb2_bench.sh
# for SB epoch 2, with the Dmix cascade of night_dmix/group.sh for segmentation, full 12500 iterations + test):
#   [SEG_SEEDS="0 1 2"] [LANES=4] bench_full.sh <out dir> <checkpoint>
#   python slurm/submit.py slurm/presets/1gpu.yaml --set gres=gpu:4 --set cpus_per_task=32 --set mem=480G \
#     --set time=<estimate> -- bash slurm/bench_full.sh <out dir> <checkpoint>
# Phases (slurm/par.sh, one command per GPU), each skipped / resumed where its outputs exist, so a rerun continues:
#   A. 16 mm extraction (4 shards), 12 / 8 mm covariance bases, block_plan of the seg tasks (one token folder per task)
#   B. 12 / 8 mm extractions
#   C. probe readings (caches)
#   D. segdec (Dmix cascade, 16 tasks x SEG_SEEDS) and probes (every non-seg task, 5 seeds; the externals keep their
#      earlier probes, user 24/10): seg in LANES lanes (one per GPU), probes beside them in 4 lanes of their own
# Segmentation data on $FAST (user 24/10): the inputs in $FAST/fcorrent/segdec_inputs, the token cache live (each run
# reads and writes the 16 / 8 mm tokens its volumes lack: no cache job) in $FAST/fcorrent/tok/<out dir name>/<task>, so
# the seeds after the first only read. block_plan chooses the blocks on seed 0's train / val volumes for every seed.
set -u
S=$1; CK=$2
R=/leonardo_scratch/large/userexternal/fcorrent/sfc_runs; MV=/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2
MX=${MV}_fast; [ -f $MX/volumes.csv ] || MX=$MV   # the extraction inputs on $FAST (fast_move/move_bench_raw.sh), else scratch
TV=configs/bench/probe_tasks_v2.yaml; PAR="bash slurm/par.sh"
F=/leonardo_scratch/fast/IscrC_SFMRI/fcorrent; TOK=$F/tok/$(basename $S); SEG_SEEDS=${SEG_SEEDS:-0}; LANES=${LANES:-4}
BIG="totalseg stroke tumor_ucsf tumor_upenn"   # their token folders on scratch: all seg volumes at 2 blocks ~620 GB, these
tokdir() { case " $BIG " in *" $1 "*) echo $S/tok/$1;; *) echo $TOK/$1;; esac; }   # ~510 (TotalSeg 275): FAST is 1 TB
source ~/mri-sfc/env.sh >/dev/null 2>&1; cd ~/mri-sfc/repo
mkdir -p $S/cmds
SEG="tumor_ucsf tumor_upenn stroke totalseg spine_seg fetal_landmarks vs_seg prostate_zones prostate_lesion wmh_seg cmb_seg heart_seg ms_seg bladder_seg cow_seg hie_seg"
OLD="age sex dementia autism nihss mrs90 idh grade survival"   # the first 9 tasks: on the volumes every model has (--common)
EXTM="brainfm brainiac bsf_atlas_pretrain bsf_ukb_pretrain fomo26 medicalnet_resnet50_23dataset nnfoundation_cnn nnfoundation_vit nnunet_totalseg_mr_870 openmind_primusm_mae openmind_resencl_mae openmind_resencl_simclr"
NEW=$(python -c "import yaml; print(' '.join(k for k, v in yaml.safe_load(open('$TV')).items() if v['type'] != 'seg' and k not in '$OLD'.split()))")
X="sfc bench-extract configs/bench/bench.yaml --group sfc_tune --fm-set checkpoint=$CK"
FINE="args.per_curve=false args.max_tokens=300000 args.fallback_mm=12"

cat > $S/cmds/basis.py <<PY
# as bench_sb2/cov_basis.py: the PCA basis of probe.second_order read at <mm> mm (400 tune volumes x 200 tokens, seed 0)
import sys, torch, pandas as pd
from multiprocessing import Pool
from sfc_gdn2.bench import probe as bp
from sfc_gdn2.fm.api import load_config
from sfc_gdn2.fm.sfc import SFC
mm, S, ck = float(sys.argv[1]), sys.argv[2], sys.argv[3]
v, s = pd.read_csv("$MX/volumes.csv", low_memory=False), pd.read_csv("$MX/splits.csv")
fit = set(v.loc[v["subject"].isin(bp.tune_subjects(s)), "id"])
cand = [i for i in sorted(v.loc[v["dataset"].isin(["crossMoDA", "TopCoW", "UCSF-PDGM"]), "id"]) if i in fit]
g = torch.Generator().manual_seed(0)
pick = [cand[k] for k in torch.randperm(len(cand), generator=g)[:400].tolist()]
image = dict(zip(v["id"], v["input"]))
cfg = load_config("configs/fm/sfc_gdn2.yaml")
cfg["checkpoint"], cfg["args"] = ck, {"patch_mm": mm, "per_curve": False, "max_tokens": 300000, "fallback_mm": 12}
w = SFC(cfg, "cuda")
def prep(i):
    return w.preprocess(image[i])
pool = {}
with Pool(8) as p:
    for pr in p.imap(prep, pick):
        f = w.features(pr)["features"]
        t = f["map"].float().flatten(1).T
        keep = t.norm(dim=1) > 0
        for name, tk in (("covsqrt", t[keep]), ("covsqrt_fg", t[keep & f["fg"].flatten().to(t.device)])):
            pool.setdefault(name, []).append(tk[torch.randperm(len(tk), generator=g)[:200].to(tk.device)].cpu())
basis = {}
for name, ts in pool.items():
    x = torch.cat(ts); mu = x.mean(0)
    basis[name] = mu, torch.linalg.eigh((x - mu).T @ (x - mu) / len(x))[1][:, -64:]
torch.save({"basis": basis, "dim": 64}, f"{S}/basis{int(mm)}.pt")
PY
echo "[bench] $(date +%H:%M) A: 16 mm, bases, block_plan"
{ for g in 0 1 2 3; do echo "$X --shard $g/4 --out $S/fm --override out_dir=$MX"; done
  for mm in 12 8; do echo "[ -f $S/basis$mm.pt ] || python $S/cmds/basis.py $mm $S $CK"; done
  echo "[ -f $TOK/blocks.json ] || { python $R/blocks/mk_job.py $CK $TOK 6 4 $SEG && python -m sfc_gdn2.fm.block_plan $TOK/blocks_job.json; }"
} > $S/cmds/A.txt
$PAR $S/cmds/A.txt

python - <<PY   # one token folder per task, its own blocks (the union over 16 tasks made every file ~3x wider)
import json
from pathlib import Path
from sfc_gdn2.fm.block_plan import layout
pl = json.loads(Path("$TOK/blocks.json").read_text())
for t, bl in pl["tasks"].items():
    cache = sorted({b for mm, v in bl.items() if float(mm) >= 8 for b in v})
    tc = pl["curves"].get(t, {})
    d = Path("$S/tok" if t in "$BIG".split() else "$TOK") / t
    d.mkdir(parents=True, exist_ok=True)
    (d / "blocks.json").write_text(json.dumps({"tasks": {t: bl}, "curves": {t: tc}, "cache": cache, "layout": layout(cache),
                                               "cache_curves": {mm: [tc.get(mm, "mean")] for mm in ("16", "8")}}, indent=1))
PY

echo "[bench] $(date +%H:%M) B: 12 / 8 mm"
{ for mm in 12 8; do for g in 0 1 2 3; do
    echo "$X args.patch_mm=$mm args.cov_basis=$S/basis$mm.pt $FINE --shard $g/4 --out $S/fm$mm --override out_dir=$MX 'fm.save.sfc_gdn2={features: [map, fg, mean, std, mean_b8, mean_b12, std_b8, std_b12, covsqrt, covsqrt_fg]}'"
  done; done
} > $S/cmds/B.txt
$PAR $S/cmds/B.txt

echo "[bench] $(date +%H:%M) C: readings"
echo "SFC_IO_WORKERS=8 python -c \"import pandas as pd; from pathlib import Path; from sfc_gdn2.bench import probe as bp; v, s = pd.read_csv('$MV/volumes.csv', low_memory=False), pd.read_csv('$MV/splits.csv'); bp.readings('sfc_gdn2', Path('$S/fm/sfc_gdn2'), set(v.loc[v['subject'].isin(bp.tune_subjects(s)), 'id']), {12: '$S/fm12/sfc_gdn2', 8: '$S/fm8/sfc_gdn2'})\"" > $S/cmds/C.txt
$PAR $S/cmds/C.txt

echo "[bench] $(date +%H:%M) D: segdec + probes"
DM="args.taps=3 args.image=true args.erase=head args.fine_curves=[raster] args.fine_encoder=true args.matryoshka=true args.head_w=0.2 args.decoder_c=true args.c_local=true args.refine=3 args.cascade=true args.fine_live4=true args.fine_split=true args.fine_head=mix args.fine_pca=0 args.blocks=auto"
MENU="'probe.menu.sfc_gdn2=[mean, blocks+std, blocks+covsqrt, mean@12mm, blocks+std@12mm, blocks+covsqrt@12mm, mean@8mm, blocks+std@8mm, blocks+covsqrt@8mm]' 'probe.extra.sfc_gdn2={12: $S/fm12/sfc_gdn2, 8: $S/fm8/sfc_gdn2}'"
P="env SFC_IO_WORKERS=8 sfc bench-probe configs/bench/bench.yaml --seeds 0,1,2,3,4"; O="--override out_dir=$MV probe.tasks=$TV fm.common_root=$R/fm_bench"
{ for sd in $SEG_SEEDS; do for t in $SEG; do   # the finest level per task (summary 24/10): 2 mm for small lesions / landmarks, 8 mm for stroke
    case $t in vs_seg|wmh_seg|cmb_seg|fetal_landmarks) gmin=2;; stroke) gmin=8;; *) gmin=4;; esac
    echo "sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2 --init pretrained --seeds $sd --resume --tasks $t --fm-set checkpoint=$CK $DM args.cascade_g0=16 args.cascade_gmin=$gmin args.token_cache=live:$(tokdir $t) --override segdec.out=$S/seg segdec.cache=$F/segdec_inputs out_dir=$MV probe.tasks=$TV"
  done; done
} > $S/cmds/Dseg.txt
{ for t in $OLD; do echo "$P --models sfc_gdn2 --common $EXTM --tasks $t $O fm.out_root=$S/fm probe.out=$S/probe $MENU"; done
  for t in $NEW; do echo "$P --models sfc_gdn2 --tasks $t $O fm.out_root=$S/fm probe.out=$S/probe $MENU"; done
} > $S/cmds/Dprobe.txt
# segdec: LANES lanes, one per GPU (its training is CPU-bound: 24/10, two fetal runs on 8 CPUs 2.25 s/it each, ~1 alone);
# probes beside them in their own lanes on the same GPUs (L-BFGS: one Python thread, a few GB), 2 threads each
lanes() {   # lanes <file> <n> <tag> [env]: the file's lines round-robin over n lanes, lane k on GPU k % 4
  for k in $(seq 0 $(($2 - 1))); do
    ( n=0; sed -n "$((k + 1))~${2}p" $1 | while IFS= read -r l; do n=$((n + 1))
        env CUDA_VISIBLE_DEVICES=$((k % 4)) ${4:-} bash -c "$l" < /dev/null > $S/cmds/$3.$k.$n.log 2>&1
        echo "[bench] $3 lane $k #$n exit $? $(date +%H:%M)"
      done ) &
  done
}
lanes $S/cmds/Dseg.txt $LANES seg
lanes $S/cmds/Dprobe.txt 4 probe OMP_NUM_THREADS=2
wait
echo "[bench] $(date +%H:%M) done"
