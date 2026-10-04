"""Observe competing compute processes; this is not a scheduler reservation."""
import os
import subprocess
import threading

def snapshot(gpus, allowed_root=None):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    selected = {}
    for row in rows:
        index, uuid, memory, util = [v.strip() for v in row.split(',')]
        if index in gpus:
            selected[uuid] = dict(index=index, used_mib=int(memory), utilization=int(util))
    assert len(selected) == len(gpus), (gpus, selected)
    allowed = set()
    if allowed_root is not None:
        allowed.add(allowed_root)
        pairs = [tuple(map(int, r.split())) for r in subprocess.check_output(
            ['ps', '-e', '-o', 'pid=,ppid='], text=True).splitlines()]
        while True:
            children = {pid for pid, parent in pairs if parent in allowed}
            if children <= allowed:
                break
            allowed |= children
    foreign = []
    for row in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True).splitlines():
        uuid, pid = [v.strip() for v in row.split(',')]
        if uuid in selected and int(pid) not in allowed:
            foreign.append(dict(gpu=selected[uuid]['index'], pid=int(pid)))
    return dict(gpus=list(selected.values()), foreign=foreign)

class Guard:
    def __init__(self, gpus, allowed_root=None):
        self.gpus = gpus
        self.root = os.getpid() if allowed_root is None else allowed_root
        self.conflict = None
        self.stop = threading.Event()
        self.check()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def check(self):
        try:
            result = snapshot(self.gpus, self.root)
            if result['foreign']:
                self.conflict = result
        except Exception as exc:
            self.conflict = dict(telemetry_error=repr(exc))

    def loop(self):
        while not self.stop.wait(5):
            self.check()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
