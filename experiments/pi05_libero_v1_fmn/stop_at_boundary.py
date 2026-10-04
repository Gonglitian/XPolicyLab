"""Stop each labserver lane right after its first stream finishes: the second streams now run on UCR BCC.
A lane waits >=60 s for an idle GPU before starting the next stage, so a 15 s poll acts before any new stage starts.
Kills only the recorded PID's process group, after checking its command line."""
import json, os, signal, time
from pathlib import Path
from paths import RUN as R
# Explicit input prevents reuse of the historical migration process IDs.
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lanes-json', type=Path, required=True,
                    help='Mapping GPU index to [method, suite, PID]; legacy multi-lane helper only')
args = parser.parse_args()
LANES = {int(gpu): tuple(spec) for gpu, spec in json.loads(args.lanes_json.read_text()).items()}
log = lambda m: print(time.strftime('[%F %T] ') + m, flush=True)
pending = dict(LANES)
log(f'watching {pending}')
while pending:
    for gpu, (method, suite, pid) in list(pending.items()):
        if not (R / method / suite / 'task09' / 'evaluated.json').exists():
            continue
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            log(f'gpu{gpu}: pid {pid} already gone'); pending.pop(gpu); continue
        if f'lane.py --gpu {gpu} ' not in cmd:
            log(f'gpu{gpu}: pid {pid} is not the lane ({cmd!r}); not killing'); pending.pop(gpu); continue
        os.killpg(pid, signal.SIGTERM)
        log(f'gpu{gpu}: {method}:{suite} finished; stopped lane pgid {pid} before its second stream')
        (R / f'status_gpu{gpu}.json').write_text(json.dumps(dict(phase='stopped_after_first_stream',
            timestamp=time.time(), gpu=str(gpu), note='second stream runs on UCR BCC')) + '\n')
        pending.pop(gpu)
    time.sleep(15)
log('all lanes handled')
