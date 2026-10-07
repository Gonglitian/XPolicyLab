#!/bin/bash
# Run TWO probe lanes inside ONE step of an existing allocation (two separate overlapping steps cannot both hold the GPUs).
#   launch_probe2.sh JOBID ROOT GPU_A JOBS_A GPU_B JOBS_B
PROBE_CODE=/data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc/code_probe
source $PROBE_CODE/bcc_env.sh
export PROBE_ROOT=$2
cd $PROBE_CODE
exec srun --jobid=$1 --overlap --gres=gpu:4 -c 24 --mem=0 -n1 bash -c "python3 probe_lane.py --gpu $3 --jobs $4 > $2/lane_gpu$3.log 2>&1 & python3 probe_lane.py --gpu $5 --jobs $6 > $2/lane_gpu$5.log 2>&1 & wait"
