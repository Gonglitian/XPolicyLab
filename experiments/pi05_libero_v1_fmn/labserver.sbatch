#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use submit.sh or sbatch}"
: "${V1_CODE:?Use submit.sh, which exports the checkout path}"
: "${V1_MACHINE_CONFIG:?Use submit.sh, which exports the machine config}"
source "$V1_MACHINE_CONFIG"
v1_prepare_dirs
# Conda initialization is required before activation in noninteractive Slurm shells.
if [ -n "${V1_CONDA_ENV:-}" ]; then
    if [ -n "${V1_CONDA_EXE:-}" ]; then export PATH="$(dirname "$V1_CONDA_EXE"):$PATH"; fi
    eval "$(conda shell.bash hook)"
    conda activate "$V1_CONDA_ENV"
fi
export PYTHONDONTWRITEBYTECODE=1
# Do not assign CUDA_VISIBLE_DEVICES: srun sets it from the allocation.
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) code=$V1_CODE"
exec srun --ntasks=1 --gres=gpu:1 --kill-on-bad-exit=1 \
    "$V1_PI_PY" "$V1_CODE/lane.py" "$@"
