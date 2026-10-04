#!/bin/bash
# Run one extra lane inside the existing 4-GPU allocation (job $1) on an idle GPU ($2) for streams ($3).
source /data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc/code/bcc_env.sh
cd $V1_RUN/code
exec srun --jobid=$1 --overlap --gres=gpu:4 -c 16 --mem=0 -n1 python3 lane.py --gpu $2 --streams $3
