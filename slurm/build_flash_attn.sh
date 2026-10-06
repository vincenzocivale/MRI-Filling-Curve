#!/usr/bin/env bash
# Build flash-attn from source for Leonardo (glibc 2.28: the upstream wheels need GLIBC_2.32).
# Compute nodes are offline: fetch the sdist on a login node first, e.g.
#   curl -sSLO https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz
#   tar xzf flash_attn-2.8.3.post1.tar.gz
# then, from the repo root (A100 = sm_80 only, ~30-60 min on a booster node):
#   python slurm/submit.py slurm/presets/1gpu.yaml --job-name flash_attn_build \
#       --set cpus_per_task=32 --set time=02:00:00 -- bash slurm/build_flash_attn.sh <src_dir> <wheel_dir>
# and install the wheel into the env: pip install --no-deps --force-reinstall <wheel_dir>/flash_attn-*.whl
set -euo pipefail
src=$(realpath "$1"); out=$(realpath -m "$2")
module load cuda/12.6 gcc/12.2.0
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=${MAX_JOBS:-16} NVCC_THREADS=${NVCC_THREADS:-2}
mkdir -p "$out"
nvcc --version | tail -2
pip wheel --no-build-isolation --no-deps -v "$src" -w "$out"
