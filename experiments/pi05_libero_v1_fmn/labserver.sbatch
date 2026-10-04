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
# Preserve the batch/srun return code even after Slurm purges its in-memory history.
# SIGKILL/node failure can prevent this trap; missing evidence never means success.
record_exit() {
    local rc=$?
    trap - EXIT
    V1_JOB_EXIT_CODE="$rc" "$V1_PI_PY" - <<'PY' || echo "Failed to persist batch exit record" >&2
import json, os, time
from pathlib import Path
path = Path(os.environ['V1_RUN']) / 'slurm' / ('exit_' + os.environ['SLURM_JOB_ID'] + '.json')
record = dict(job_id=os.environ['SLURM_JOB_ID'], batch_exit_code=int(os.environ['V1_JOB_EXIT_CODE']),
              code_commit=os.environ.get('V1_CODE_COMMIT'), timestamp=time.time(),
              source='batch EXIT trap after srun; not scheduler terminal state')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(path)
PY
    exit "$rc"
}
trap record_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Do not assign CUDA_VISIBLE_DEVICES: srun sets it from the allocation.
echo "SLURM_JOB_ID=$SLURM_JOB_ID host=$(hostname) code=$V1_CODE commit=${V1_CODE_COMMIT:-unknown} predecessor=${V1_PREDECESSOR_JOB_ID:-none}"
srun --ntasks=1 --gres=gpu:1 --kill-on-bad-exit=1 \
    "$V1_PI_PY" "$V1_CODE/lane.py" "$@"
