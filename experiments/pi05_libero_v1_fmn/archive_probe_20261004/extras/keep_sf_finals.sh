#!/bin/bash
# Keep each sequential-fine-tuning task-final checkpoint by hard-linking it before lane.py prunes it.
# Needed for the component-swap diagnostic (swap SigLIP / Gemma LoRA / action expert between task checkpoints).
# Hard links cost no extra space until the original is pruned; delete $B/kept_sf_finals to free ~9 GB per task.
B=/data/JiachenLi/litian/xpl-runs/pi05_libero_v1_fmn_bcc
K=$B/kept_sf_finals
while true; do
  for s in libero_spatial libero_object libero_goal libero_10; do
    for t in $B/sf/$s/task*; do
      [ -f $t/trained.json ] || continue
      n=$(basename $t); step=$(( (10#${n#task} + 1) * 10000 ))
      src=$t/checkpoints/pi05_libero_cl/stage/$step; dst=$K/$s/$n/$step
      [ -d $src/params ] && [ ! -d $dst ] && mkdir -p $K/$s/$n && cp -al $src $dst && echo "$(date "+%F %T") kept $s/$n/$step"
    done
  done
  [ -f $B/sf/libero_spatial/task09/evaluated.json ] && [ -f $B/sf/libero_object/task09/evaluated.json ] && { echo "$(date "+%F %T") both SF streams finished; exiting"; exit 0; }
  sleep 300
done
