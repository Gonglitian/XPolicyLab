"""Run plasticity-probe jobs on one GPU, one after another: train a single task from pi05_base, then evaluate that task
on its 50 fixed initial states (4 policy servers + 4 simulator clients, like lane.py). Resumable: finished parts are skipped.
  PROBE_ROOT=<dir> python3 probe_lane.py --gpu 0 --jobs frozen_siglip:libero_spatial:0,trainable_siglip:libero_spatial:4"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
from pathlib import Path
from common import write_json
from lane import environment, PI_PY, SIM_PY, CODE
from gpu_guard import Guard

ROOT = Path(os.environ['PROBE_ROOT'])
N = 4

def run_logged(command, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as log:
        log.write('\nCOMMAND ' + json.dumps([str(v) for v in command]) + '\n')
        log.flush()
        subprocess.run([str(v) for v in command], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

def evaluate(gpu, stage, suite, task, checkpoint, episodes):
    eval_dir = stage / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    servers, clients, handles = [], [], []
    guard = Guard([gpu], allowed_root=os.getpid())
    base = 6600 + 10 * int(gpu)
    try:
        for s in range(N):
            with socket.socket() as probe:
                assert probe.connect_ex(('127.0.0.1', base + s)) != 0, f'Port {base + s} already occupied'
            log = (eval_dir / f'server{s}.log').open('a'); handles.append(log)
            senv = environment(gpu)
            senv['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.22'
            servers.append(subprocess.Popen([str(PI_PY), str(CODE / 'serve.py'), '--checkpoint', checkpoint,
                '--port', str(base + s)], env=senv, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 1200
        for s, server in enumerate(servers):
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server {s} exited with {server.returncode}')
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1', base + s)) == 0:
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Policy server {s} startup timeout')
                time.sleep(2)
        for w in range(N):
            log = (eval_dir / f'client{w}.log').open('a'); handles.append(log)
            clients.append(subprocess.Popen([str(SIM_PY), str(CODE / 'probe_eval.py'), '--suite', suite, '--task', str(task),
                '--worker', str(w), '--workers', str(N), '--episodes', str(episodes), '--port', str(base + w),
                '--output', str(eval_dir / f'worker{w}.json')], env=environment(gpu, sim=True), stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in clients):
            for p in clients + servers:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f'Evaluation subprocess {p.pid} failed with {p.returncode}')
            time.sleep(5)
        records = []
        for w in range(N):
            records.extend(json.loads((eval_dir / f'worker{w}.json').read_text())['episodes'])
        assert len(records) == episodes and len({r['episode'] for r in records}) == episodes, len(records)
        result = dict(success=sum(r['success'] for r in records) / episodes, episodes=episodes,
                      finished_at=time.time(), gpu_contention=guard.conflict)
        write_json(stage / 'evaluated.json', result)
        return result
    finally:
        guard.close()
        for proc in clients + servers:
            if proc.poll() is None:
                proc.terminate()
        for proc in clients + servers:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        for h in handles:
            h.close()

def job(gpu, arm, suite, task, steps, episodes, status):
    stage = ROOT / arm / suite / f'task{task:02d}'
    if not (stage / 'trained.json').exists():
        status('training', arm=arm, suite=suite, task=task)
        run_logged([PI_PY, CODE / 'probe_train.py', '--arm', arm, '--suite', suite, '--task', task, '--steps', steps,
                    '--stage', stage], stage / 'train.log', environment(gpu))
    trained = json.loads((stage / 'trained.json').read_text())
    if not (stage / 'evaluated.json').exists():
        for attempt in range(3):
            status('evaluation', arm=arm, suite=suite, task=task, attempt=attempt)
            try:
                evaluate(gpu, stage, suite, task, trained['checkpoint'], episodes)
                break
            except Exception as exc:
                bad = stage / f'evaluation_failed_{attempt}'
                shutil.rmtree(bad, ignore_errors=True)
                (stage / 'evaluation').rename(bad)
                write_json(bad / 'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
                if attempt == 2:
                    raise
    result = json.loads((stage / 'evaluated.json').read_text())
    # The optimiser state is not needed after evaluation; the params stay (they can serve as an expert later).
    shutil.rmtree(Path(trained['checkpoint']) / 'train_state', ignore_errors=True)
    print('PROBE_DONE', arm, suite, task, json.dumps(result), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--jobs', required=True, help='comma list of arm:suite:task')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--episodes', type=int, default=50)
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        record = dict(phase=phase, timestamp=time.time(), pid=os.getpid(), gpu=a.gpu, **extra)
        write_json(ROOT / f'status_gpu{a.gpu}.json', record)
        print('PROBE_STATUS', json.dumps(record), flush=True)
    try:
        for item in a.jobs.split(','):
            arm, suite, task = item.split(':')
            job(a.gpu, arm, suite, int(task), a.steps, a.episodes, status)
        status('completed', jobs=a.jobs)
    except Exception as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
