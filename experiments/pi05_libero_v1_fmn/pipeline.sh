#!/bin/bash
# v1 dependency chain (run with nohup). Each step is skipped if its output exists.
#   download -> prepare_tasks (sim env) -> prepare_data (openpi env) -> orient_check (sim env)
#   -> preflight on GPU $GPU (5-step ER tasks 0/1 + 2-episode eval) -> first formal lane on GPU $GPU.
# Other lanes are started only after the first lane's task-0 diagonal clears the gate (>=0.8).
set -euo pipefail
RUN=/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/pi05_libero_v1_fmn
DL=/data2/vla-reasoning/proj/XPolicyLab-assets/datasets/dl_libero256.log
GPU=${GPU:-1}
FIRST_STREAMS=${FIRST_STREAMS:-er:libero_spatial,er:libero_goal}
cd $RUN/code
log(){ echo "[$(date '+%F %T')] $*"; }
until grep -q DOWNLOAD_DONE $DL; do sleep 60; done; log download done
[ -f $RUN/bench_tasks.json ] || { python3 envrun.py $GPU sim -- PY prepare_tasks.py $RUN/bench_tasks.json; log tasks ok; }
[ -f $RUN/manifest.json ] || { python3 envrun.py $GPU pi -- PY prepare_data.py $RUN/bench_tasks.json; log data ok; }
[ -f $RUN/orientation_check.json ] || { python3 envrun.py $GPU pi -- PY orient_dump.py; python3 envrun.py $GPU sim -- PY orient_check.py; log orientation ok; }
if ! grep -q preflight_passed $RUN/status_gpu$GPU.json 2>/dev/null; then
  python3 lane.py --gpu $GPU --streams er:libero_spatial --preflight; log preflight passed
fi
log starting formal lane gpu$GPU $FIRST_STREAMS
exec python3 lane.py --gpu $GPU --streams $FIRST_STREAMS
