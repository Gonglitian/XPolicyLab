#!/bin/bash
# Original multi-lane workflow; the one-GPU Slurm entry is a separate integration step.
# Source labserver_env.sh or bcc_env.sh before invoking within a Slurm allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?GPU execution requires a Slurm allocation}"
: "${V1_RUN:?Source a machine configuration first}"
: "${V1_PI_PY:?Source a machine configuration first}"
: "${V1_DOWNLOAD_LOG:?Source a machine configuration first}"
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$CODE/env_defaults.sh"
v1_prepare_dirs
RUN=$V1_RUN
DL=$V1_DOWNLOAD_LOG
GPU=${GPU:-1}
FIRST_STREAMS=${FIRST_STREAMS:-er:libero_spatial,er:libero_goal}
cd "$CODE"
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q DOWNLOAD_DONE "$DL"; do sleep 60; done; log download done
[ -f "$RUN/bench_tasks.json" ] || { "$V1_PI_PY" envrun.py "$GPU" sim -- PY prepare_tasks.py "$RUN/bench_tasks.json"; log tasks ok; }
[ -f "$RUN/manifest.json" ] || { "$V1_PI_PY" envrun.py "$GPU" pi -- PY prepare_data.py "$RUN/bench_tasks.json"; log data ok; }
[ -f "$RUN/orientation_check.json" ] || { "$V1_PI_PY" envrun.py "$GPU" pi -- PY orient_dump.py; "$V1_PI_PY" envrun.py "$GPU" sim -- PY orient_check.py; log orientation ok; }
if ! grep -q preflight_passed "$RUN/status_gpu$GPU.json" 2>/dev/null; then
  "$V1_PI_PY" lane.py --gpu "$GPU" --streams er:libero_spatial --preflight; log preflight passed
fi
log starting formal lane "gpu$GPU" "$FIRST_STREAMS"
exec "$V1_PI_PY" lane.py --gpu "$GPU" --streams "$FIRST_STREAMS"
