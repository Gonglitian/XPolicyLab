#!/bin/bash
# Gate: wait for ER Spatial task-0 evaluation on lane 1; if its diagonal >= 0.8, start lanes 2-4
# (each lane waits by itself until its GPU is idle, so a GPU used by someone else is never taken).
R=/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/pi05_libero_v1_fmn
E=$R/er/libero_spatial/task00/evaluated.json
log(){ echo "[$(date '+%F %T')] $*"; }
until [ -f $E ]; do
  grep -q '"failed"' $R/status_gpu1.json 2>/dev/null && { log "lane 1 failed before the gate; not launching"; exit 1; }
  sleep 120
done
D=$(python3 -c "import json;print(json.load(open('$E'))['row']['0'])")
log "gate: ER Spatial task0 diagonal = $D"
if python3 -c "import sys;sys.exit(0 if float('$D') >= 0.8 else 1)"; then
  cd $R/code
  for spec in "2 sf:libero_spatial,er:libero_goal" "3 er:libero_object,sf:libero_10" "4 sf:libero_object,er:libero_10"; do
    set -- $spec
    setsid nohup python3 lane.py --gpu $1 --streams $2 > $R/lane_gpu$1.log 2>&1 < /dev/null &
    log "launched lane gpu$1 $2 pid $!"
  done
  echo "{\"gate\": \"passed\", \"diagonal\": $D}" > $R/gate.json
else
  echo "{\"gate\": \"failed\", \"diagonal\": $D}" > $R/gate.json
  cat > $R/GATE_FAILED.md <<MD
ER Spatial task0 diagonal = $D < 0.8. Lanes 2-4 were NOT launched. Lane 1 keeps running.
Investigate before scaling (see er/libero_spatial/task00/{metrics.jsonl,evaluation/}).
MD
  log "gate failed; lanes 2-4 not launched"
fi
