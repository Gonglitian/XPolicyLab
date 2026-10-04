#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 32 --mem=300G -t 3-00:00:00 -J v1-lanes
#SBATCH -o /data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc/slurm-%j.log
# Four independent single-GPU lanes (global batch 8 each, same as tasl-labserver), not one FSDP job.
# Resumable: a follow-up job with the same name (--dependency=singleton) continues where this one stopped.
set -euo pipefail
source /data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc/code/bcc_env.sh
cd $V1_RUN/code
log(){ echo "[$(date '+%F %T')] $*"; }
log "node $(hostname) SLURM CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv,noheader
STREAMS=(${V1_STREAMS:-sf:libero_goal er:libero_goal sf:libero_10 er:libero_10})
# lane.py addresses GPUs by nvidia-smi index (PCI bus order); use the indices Slurm granted.
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })
[ ${#GPUS[@]} -eq 4 ] || { log "expected 4 GPUs, got ${CUDA_VISIBLE_DEVICES:-none}"; exit 1; }
G0=${GPUS[0]}
[ -f $V1_RUN/bench_tasks.json ] || { python3 envrun.py $G0 sim -- PY prepare_tasks.py $V1_RUN/bench_tasks.json; log tasks ok; }
[ -f $V1_RUN/manifest.json ] || { python3 envrun.py $G0 pi -- PY prepare_data.py $V1_RUN/bench_tasks.json; log data ok; }
[ -f $V1_RUN/orientation_check.json ] || { python3 envrun.py $G0 pi -- PY orient_dump.py; python3 envrun.py $G0 sim -- PY orient_check.py; log orientation ok; }
if [ ! -f $V1_RUN/preflight_passed ]; then
  python3 lane.py --gpu $G0 --streams er:libero_spatial --preflight && touch $V1_RUN/preflight_passed; log preflight passed
fi
for i in 0 1 2 3; do
  log "lane gpu${GPUS[$i]} ${STREAMS[$i]}"
  python3 lane.py --gpu ${GPUS[$i]} --streams ${STREAMS[$i]} >> $V1_RUN/lane_${STREAMS[$i]/:/_}.log 2>&1 &
done
wait
log "all lanes exited"
