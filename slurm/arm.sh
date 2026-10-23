#!/bin/bash
# One screening arm in ONE job (user, 23/10: chains of dependent jobs paid 3 queue waits each): the pretraining on the
# job's 2 GPUs, then its quick evaluation two steps at a time, one per GPU. Submit with
#   python slurm/submit.py slurm/presets/1gpu.yaml --job-name <name> --set gres=gpu:2 --set cpus_per_task=16 \
#     --set mem=240G --set time=09:30:00 -- bash slurm/arm.sh <run dir> <seed> [pretrain override ...]
# Environment: NGPU (2; 1: steps one after the other), STEPS (default 3000; the pretraining is skipped once encoder_step<STEPS>.pt exists), DECAY (600),
# TASKS (default "wmh_seg cmb_seg vs_seg fetal_landmarks"), SEG_ITERS (2500), PROBE=0 / CACHE=0 (skip; smoke tests). Every step is skipped when its output
# exists (tune/tune.csv, tok/, the segdec json) and runs with its own log under <run dir>/logs; a failing step does
# not stop the others; a summary of the exit codes ends the job. A job that times out is resubmitted identical: the
# pretraining resumes from its ckpt.pt, decoders from their .last.pt.
set -u
D=$1; SEED=$2; shift 2
STEPS=${STEPS:-3000}; DECAY=${DECAY:-600}; SEG_ITERS=${SEG_ITERS:-2500}; NGPU=${NGPU:-2}
TASKS=${TASKS-"wmh_seg cmb_seg vs_seg fetal_landmarks"}
R=/leonardo_scratch/large/userexternal/fcorrent/sfc_runs
MV=/leonardo_work/IscrC_SFMRI/fcorrent/datasets/manifests/bench_v2; TV=configs/bench/probe_tasks_v2.yaml
mkdir -p $D/logs; cd ~/mri-sfc/repo
step() {   # step <name> <command...>: its log and its exit code in <run dir>/logs (steps run in background subshells)
  local n=$1; shift
  echo "[arm] $n start $(date +%H:%M)"
  "$@" > $D/logs/$n.log 2>&1; local rc=$?; echo $rc > $D/logs/$n.rc
  echo "[arm] $n exit $rc $(date +%H:%M)"
  return $rc
}
# 1. pretraining: the recipe of the screens (p15/p16), 96 scans per step on 2 GPUs
BASE="train.lr=0.0004 train.steps=$STEPS train.decay_steps=$DECAY train.warmup_steps=150 train.save_every=1000 train.ckpt_every=250 train.sync_bn=true model.d_model=1024 model.depth=16 model.num_heads=16 model.attn_every=4 model.embed.basis=\"bspline\" objective.max_packed_tokens=262144 objective.checkpoint_tokens=0 data.patches_per_batch=131072 train.batch_size=48"
if ls $D/lejepa-*/encoder_step$(printf %06d $STEPS).pt >/dev/null 2>&1; then echo "[arm] pretrain done, skipped"
else step pretrain torchrun --standalone --nproc_per_node=$NGPU -m sfc_gdn2.cli pretrain configs/pretrain_lejepa.yaml --override seed=$SEED $BASE 'objective.mask_cells=[1, 2]' "$@" output_root=$D
fi
ls $D/lejepa-*/encoder_step$(printf %06d $STEPS).pt >/dev/null 2>&1 || { echo "[arm] no final checkpoint, evaluation skipped"; exit 1; }
# 2. evaluation, two steps at a time (GPU 0 | GPU 1)
Q="dataset in ['crossMoDA', 'TopCoW', 'UCSF-PDGM']"
probe() {
  [ "${PROBE:-1}" = 1 ] || return 0
  [ -f $D/tune/tune.csv ] && { echo "[arm] probe done, skipped"; return 0; }
  step probe bash -c "sfc bench-extract configs/bench/bench.yaml --group sfc_tune --tune --fm-set checkpoint=$D --query \"$Q\" --out $D/features --override out_dir=$MV && sfc bench-probe configs/bench/bench.yaml --models sfc_gdn2 --tune --tasks idh koos pcom_any --out $D/tune --override fm.out_root=$D/features out_dir=$MV probe.tasks=$TV"
}
cache() {
  [ "${CACHE:-1}" = 1 ] || return 0
  [ -d $D/tok ] && [ -f $D/logs/cache.done ] && { echo "[arm] cache done, skipped"; return 0; }
  step cache python $R/tune/p15_screen/tokjob.py $D && touch $D/logs/cache.done
}
F="args.taps=3 args.refine=2 args.image=true args.erase=head args.fine_curves=[raster] args.fine_encoder=true args.matryoshka=true args.head_w=0.2 args.decoder_c=true args.c_local=true args.c_wpow=0.5"
seg() {
  [ -f $D/segdec/$1/s0/sfc_gdn2__pretrained.json ] && { echo "[arm] seg $1 done, skipped"; return 0; }
  step seg_$1 sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2 --init pretrained --tasks $1 --iters $SEG_ITERS --val-only --fm-set checkpoint=$D $F args.fine_pca=0 args.token_cache=$D/tok --override segdec.out=$D/segdec out_dir=$MV probe.tasks=$TV
}
pair() {   # pair <cmd for GPU 0> <cmd for GPU 1> (either may be empty); NGPU=1: one after the other
  if [ $NGPU = 1 ]; then for c in "$1" "$2"; do [ -n "$c" ] && eval "$c"; done; return; fi
  [ -n "$1" ] && { CUDA_VISIBLE_DEVICES=0 eval "$1" & }
  [ -n "$2" ] && { CUDA_VISIBLE_DEVICES=1 eval "$2" & }
  wait
}
pair cache probe
set -- $TASKS
while [ $# -gt 0 ]; do pair "seg $1" "${2:+seg $2}"; shift; [ $# -gt 0 ] && shift; done
echo "[arm] summary (exit codes):"; for f in $D/logs/*.rc; do [ -f $f ] && echo "  $(basename $f .rc) $(cat $f)"; done
