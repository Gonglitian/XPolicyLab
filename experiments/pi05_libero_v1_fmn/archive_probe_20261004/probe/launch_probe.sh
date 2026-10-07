#!/bin/bash
# Run probe_lane.py on one GPU of an existing Slurm allocation.  launch_probe.sh JOBID GPU JOBS ROOT [STEPS] [EPISODES]
PROBE_CODE=/data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc/code_probe
source $PROBE_CODE/bcc_env.sh      # sets the V1_* paths (and its own B / A variables)
export PROBE_ROOT=$4
cd $PROBE_CODE
exec srun --jobid=$1 --overlap --gres=gpu:4 -c 16 --mem=0 -n1 python3 probe_lane.py --gpu $2 --jobs $3 --steps ${5:-10000} --episodes ${6:-50}
