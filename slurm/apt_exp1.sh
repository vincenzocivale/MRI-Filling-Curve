#!/bin/bash
# APT experiment 1 (19/10): p11_res #3 (B-spline, patch 12/16/20 mm, d=512) read as an octree of 16 -> 8 mm patches.
# Extraction (2 GPUs, arms of configs/fm/sfc_gdn2_apt.yaml) on the tuning subjects of the six probe tasks' datasets
# (no test split is read), then the tuning probe against the uniform 16 and 12 mm readings of the same checkpoint.
#   python slurm/submit.py slurm/presets/1gpu.yaml --job-name apt_exp1 --set gres=gpu:2 --set cpus_per_task=16 \
#       --set mem=240G --set time=06:00:00 -- bash slurm/apt_exp1.sh
set -eo pipefail
P=/leonardo_scratch/large/userexternal/fcorrent/sfc_runs/tune/p11_res/03
RUN=$P/lejepa-8bbd5b3eca
OUT=${APT_OUT:-/leonardo_scratch/large/userexternal/vcivale0/apt/exp1}
Q="dataset in ['AOMIC-ID1000','Calgary','DLBS','IXI','Long579','Pixar','OASIS-1','UCSF-PDGM','UPENN-GBM','SOOP']"
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g python -m sfc_gdn2.cli bench-extract configs/bench/bench.yaml --group sfc_apt --tune \
    --query "$Q" --fm-set checkpoint=$RUN --shard $g/2 --out $OUT &
done
wait
python scripts/apt_eval.py $OUT/sfc_gdn2_apt $OUT/eval.csv --gpus 2 \
  --uniform 16=$P/features/sfc_gdn2 12=$P/eval12/features/sfc_gdn2
