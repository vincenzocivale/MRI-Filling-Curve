#!/bin/bash
# Several independent 1-GPU commands in ONE job (user, 23/10: decoders of different tasks / models side by side, the
# protocol unchanged): par.sh <file of commands, one per line>. Each line runs as its own job step on one GPU of the
# allocation (any node), with its own log <file>.<n>.log; lines beyond the GPUs wait for a free one (srun --exact).
# Submit with the GPUs wanted, e.g.
#   python slurm/submit.py slurm/presets/1gpu.yaml --set gres=gpu:4 --set nodes=N --set cpus_per_task=32 ... -- bash slurm/par.sh cmds.txt
# A summary of the exit codes ends the job.
set -u
F=$1
CPG=$(( ${SLURM_CPUS_PER_TASK:-8} / ${SLURM_GPUS_ON_NODE:-1} ))
MEM=$(( ${SLURM_MEM_PER_NODE:-120000} / ${SLURM_GPUS_ON_NODE:-1} ))
n=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  n=$((n + 1))
  echo "[par] $n start $(date +%H:%M): $line" | cut -c1-200
  ( srun --exact -N1 -n1 --gres=gpu:1 --cpus-per-task=$CPG --mem=${MEM}M bash -c "$line" > $F.$n.log 2>&1
    echo $? > $F.$n.rc; echo "[par] $n exit $(cat $F.$n.rc) $(date +%H:%M)" ) &
done < "$F"
wait
echo "[par] summary:"; for i in $(seq 1 $n); do echo "  $i $(cat $F.$i.rc 2>/dev/null)"; done
