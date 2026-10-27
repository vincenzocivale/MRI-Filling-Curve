#!/bin/bash
# Full benchmark of one encoder checkpoint in ONE 4-GPU job (user, 24/10: no chain of ~150 small jobs; as sb2_bench.sh
# for SB epoch 2, with the Dmix cascade of night_dmix/group.sh for segmentation, full 12500 iterations + test):
#   [SEG_SEEDS="0 1 2"] [LANES=4] bench_full.sh <out dir> <checkpoint>
#   python slurm/submit.py slurm/presets/1gpu.yaml --set gres=gpu:4 --set cpus_per_task=32 --set mem=480G \
#     --set time=<estimate> -- bash slurm/bench_full.sh <out dir> <checkpoint>
# Phases (slurm/par.sh, one command per GPU), each skipped / resumed where its outputs exist, so a rerun continues:
#   A. the extraction (4 shards): 16 mm + the multigrain levels 8 / 4 / 2 mm and the exact covariances (fm/sfc.multigrain,
#      user 27/10: our only reading under a head; no 12 / 8 mm extractions, no covariance bases)
#   B. segdec (Dmix cascade, 16 tasks x SEG_SEEDS, the longest first) in LANES lanes (one per GPU), from here on
#   C. probe readings (caches), beside the segdec
#   D. probes (every non-seg task, 5 seeds; the externals keep their earlier probes, user 24/10) in 4 lanes of their own
# Segmentation data on $FAST (user 24/10): the inputs in $FAST/fcorrent/segdec_inputs, the token cache live (each run
# reads and writes the 16 / 8 mm tokens its volumes lack: no cache job) in $FAST/fcorrent/tok/<out dir name>/<task>, so
# the seeds after the first only read. Blocks: the depth quartiles (taps 3 + last) concatenated at every level, as the
# probe (user 27/10: no block_plan).
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
SEG="totalseg tumor_ucsf tumor_upenn stroke spine_seg fetal_landmarks vs_seg prostate_zones prostate_lesion wmh_seg cmb_seg heart_seg ms_seg bladder_seg cow_seg hie_seg"
OLD="age sex dementia autism nihss mrs90 idh grade survival"   # the first 9 tasks: on the volumes every model has (--common)
EXTM="brainfm brainiac bsf_atlas_pretrain bsf_ukb_pretrain fomo26 medicalnet_resnet50_23dataset nnfoundation_cnn nnfoundation_vit nnunet_totalseg_mr_870 openmind_primusm_mae openmind_resencl_mae openmind_resencl_simclr"
NEW=$(python -c "import yaml; print(' '.join(k for k, v in yaml.safe_load(open('$TV')).items() if v['type'] != 'seg' and k not in '$OLD'.split()))")
X="sfc bench-extract configs/bench/bench.yaml --group sfc_tune --fm-set checkpoint=$CK"

echo "[bench] $(date +%H:%M) A: extraction"
for g in 0 1 2 3; do echo "$X --shard $g/4 --out $S/fm --override out_dir=$MX"; done > $S/cmds/A.txt
$PAR $S/cmds/A.txt

for t in $SEG; do mkdir -p $(tokdir $t); done   # one token folder per task

PIDS=""
lanes() {   # lanes <file> <n> <tag> [env]: n lanes (lane k on GPU k % 4), each taking the file's next line when it frees
  echo 0 > $S/cmds/$3.next   # (a queue, not a fixed share: no GPU idle at the end while another lane still has 3 tasks)
  for k in $(seq 0 $(($2 - 1))); do
    ( while n=$(flock $S/cmds/$3.next sh -c 'n=$(($(cat "$0") + 1)); echo $n > "$0"; echo $n' $S/cmds/$3.next) \
            && l=$(sed -n "${n}p" $1) && [ -n "$l" ]; do   # a failure (e.g. GPU OOM beside an extraction) retried once:
        { env CUDA_VISIBLE_DEVICES=$((k % 4)) ${4:-} bash -c "$l" || \
          env CUDA_VISIBLE_DEVICES=$((k % 4)) ${4:-} bash -c "$l"; } < /dev/null > $S/cmds/$3.$n.log 2>&1   # both resume
        echo "[bench] $3 #$n lane $k exit $? $(date +%H:%M)"
      done ) &
    PIDS="$PIDS $!"
  done
}
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 60 > $S/cmds/gpu.csv 2>&1 &
G=$!   # GPU use every minute: is any GPU idle, could LANES=8 (2 segdec per GPU) fill it

echo "[bench] $(date +%H:%M) B: segdec (runs beside C and D)"
DM="args.taps=3 args.image=true args.erase=head args.fine_curves=[raster] args.fine_encoder=true args.matryoshka=true args.head_w=0.2 args.decoder_c=true args.c_local=true args.refine=3 args.cascade=true args.fine_live4=true args.fine_split=true args.fine_head=mix args.fine_pca=0"
P="env SFC_IO_WORKERS=8 sfc bench-probe configs/bench/bench.yaml --seeds 0,1,2,3,4"; O="--override out_dir=$MV probe.tasks=$TV fm.common_root=$R/fm_bench"
{ for sd in $SEG_SEEDS; do for t in $SEG; do   # the finest level per task (summary 24/10): 2 mm for small lesions / landmarks, 8 mm for stroke
    case $t in vs_seg|wmh_seg|cmb_seg|fetal_landmarks) gmin=2;; stroke) gmin=8;; *) gmin=4;; esac
    echo "sfc bench-segdec configs/bench/bench.yaml --stage train --models sfc_gdn2 --init pretrained --seeds $sd --resume --tasks $t --fm-set checkpoint=$CK $DM args.cascade_g0=16 args.cascade_gmin=$gmin args.token_cache=live:$(tokdir $t) --override segdec.patience=2 segdec.out=$S/seg segdec.cache=$F/segdec_inputs out_dir=$MV probe.tasks=$TV"
  done; done
} > $S/cmds/Dseg.txt
# segdec: LANES lanes, one per GPU (its training is CPU-bound: 24/10, two fetal runs on 8 CPUs 2.25 s/it each, ~1 alone)
lanes $S/cmds/Dseg.txt $LANES seg OMP_NUM_THREADS=4

echo "[bench] $(date +%H:%M) C: readings"
echo "SFC_IO_WORKERS=8 python -c \"import pandas as pd; from pathlib import Path; from sfc_gdn2.bench import probe as bp; v, s = pd.read_csv('$MV/volumes.csv', low_memory=False), pd.read_csv('$MV/splits.csv'); bp.readings('sfc_gdn2', Path('$S/fm/sfc_gdn2'), set(v.loc[v['subject'].isin(bp.tune_subjects(s)), 'id']))\"" > $S/cmds/C.txt
$PAR $S/cmds/C.txt

echo "[bench] $(date +%H:%M) D: probes"
{ for t in $OLD; do echo "$P --models sfc_gdn2 --common $EXTM --tasks $t $O fm.out_root=$S/fm probe.out=$S/probe"; done
  for t in $NEW; do echo "$P --models sfc_gdn2 --tasks $t $O fm.out_root=$S/fm probe.out=$S/probe"; done
} > $S/cmds/Dprobe.txt
# probes beside the segdec in their own lanes on the same GPUs (L-BFGS: one Python thread, a few GB), 2 threads each
lanes $S/cmds/Dprobe.txt 4 probe OMP_NUM_THREADS=2
wait $PIDS; kill $G
echo "[bench] $(date +%H:%M) done"
