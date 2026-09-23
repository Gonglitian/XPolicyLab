"""Sequential durable job queue. Never counts technical preflight as formal evaluation."""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from common import ROOT, ASSETS, RUN, SUITES, write_json
from gpu_guard import snapshot, Guard
from settings import REPO, PI_PY, SIM_PY, GPUS, PORT, METHODS

CODE = Path(__file__).resolve().parent

def environment(sim=False, gpu=None):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES=','.join(GPUS) if gpu is None else gpu,
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.85',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TOKENIZERS_PARALLELISM='false',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg',
        LIBERO_CONFIG_PATH=str(ROOT / 'config/libero'),
        JAX_COMPILATION_CACHE_DIR=str(RUN / 'jax_cache'),
        PYTHONHASHSEED='42')
    paths = [str(CODE), str(REPO)]
    if sim:
        paths.insert(0, str(ROOT / 'upstreams/libero'))
    else:
        paths.insert(0, str(ROOT / 'upstreams/openpi/src'))
        paths.extend([str(ROOT / 'upstreams/robocasa'), str(ROOT / 'upstreams/robosuite')])
    env['PYTHONPATH'] = ':'.join(paths)
    if sim and gpu is not None:
        env['MUJOCO_EGL_DEVICE_ID'] = gpu
    return env

def status(phase, **extra):
    record = dict(phase=phase, timestamp=time.time(), pid=os.getpid(), **extra)
    write_json(RUN / 'status.json', record)
    print('QUEUE_STATUS', json.dumps(record), flush=True)

def run_logged(command, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as log:
        log.write('\nCOMMAND ' + json.dumps([str(v) for v in command]) + '\n')
        log.flush()
        subprocess.run([str(v) for v in command], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

def wait_for_gpus(required_mib=30000):
    last_busy = None
    free_since = None
    while True:
        sample = snapshot(GPUS)
        busy = [g for g in sample['gpus'] if g['used_mib'] >= 1000 or g['utilization'] > 5]
        busy += sample['foreign']
        if not busy:
            if free_since is None:
                free_since = time.monotonic()
            if time.monotonic() - free_since >= 60:
                return
            time.sleep(10)
            continue
        free_since = None
        if busy != last_busy:
            status('waiting_for_four_idle_gpus', busy=busy, gpus=GPUS,
                   rule='No foreign compute processes, used memory <1000 MiB, utilization <=5%, stable 60s')
            last_busy = busy
        time.sleep(60)

def evaluate(stage, suite, task, checkpoint, episodes=50):
    wait_for_gpus(12000)
    eval_dir = stage / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    servers, clients, handles = [], [], []
    guard = Guard(GPUS)
    try:
        for worker, gpu in enumerate(GPUS):
            port = PORT + worker
            with socket.socket() as probe:
                assert probe.connect_ex(('127.0.0.1', port)) != 0, f'Port {port} already occupied'
            log = (eval_dir / f'server{worker}.log').open('a')
            handles.append(log)
            server = subprocess.Popen([str(PI_PY), str(CODE / 'serve.py'), '--checkpoint', checkpoint,
                '--port', str(port)], env=environment(gpu=gpu), stdout=log, stderr=subprocess.STDOUT)
            servers.append(server)
        deadline = time.monotonic() + 900
        for worker, server in enumerate(servers):
            port = PORT + worker
            while True:
                if guard.conflict:
                    raise RuntimeError('GPU contention during evaluation: ' + str(guard.conflict))
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server {worker} exited with {server.returncode}')
                with socket.socket() as probe:
                    if probe.connect_ex(('127.0.0.1', port)) == 0:
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Policy server {worker} startup timeout')
                time.sleep(2)
            log = (eval_dir / f'client{worker}.log').open('a')
            handles.append(log)
            clients.append(subprocess.Popen([str(SIM_PY), str(CODE / 'evaluate.py'), '--suite', suite,
                '--stage', str(task), '--worker', str(worker), '--episodes', str(episodes),
                '--port', str(port), '--output', str(eval_dir / f'worker{worker}.json')],
                env=environment(sim=True, gpu=GPUS[worker]), stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in clients):
            if guard.conflict:
                raise RuntimeError('GPU contention during evaluation: ' + str(guard.conflict))
            for p in clients + servers:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f'Evaluation subprocess {p.pid} failed with {p.returncode}')
            time.sleep(5)
        for worker, proc in enumerate(clients):
            if proc.returncode != 0:
                raise RuntimeError(f'Evaluation client {worker} exited with {proc.returncode}; see client{worker}.log')
        guard.check()
        if guard.conflict:
            raise RuntimeError('GPU contention during evaluation: ' + str(guard.conflict))
        records = []
        for worker in range(4):
            records.extend(json.loads((eval_dir / f'worker{worker}.json').read_text())['episodes'])
        assert len(records) == (task + 1) * episodes
        assert len({(r['task'], r['episode']) for r in records}) == len(records)
        row = {str(i): sum(r['success'] for r in records if r['task'] == i) / episodes for i in range(task + 1)}
        write_json(stage / 'evaluated.json', dict(row=row, episodes=episodes,
            total_episodes=len(records), finished_at=time.time(),
            idle_gpu_timing_valid=not (stage / 'evaluation_contention.json').exists()))
        return row
    finally:
        guard.close()
        if guard.conflict:
            write_json(stage / 'evaluation_contention.json', guard.conflict)
        for proc in clients + servers:
            if proc.poll() is None:
                proc.terminate()
        for proc in clients + servers:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()

def train_and_eval(root, method, suite, task, steps=10000, preflight=False):
    stage = root / method / suite / f'task{task:02d}'
    if not (stage / 'evaluated.json').exists():
        wait_for_gpus()
    if not (stage / 'trained.json').exists():
        status('preflight_training' if preflight else 'training', method=method, suite=suite, task=task, gpus=GPUS,
               stage_dir=str(stage), target_steps=steps)
        command = [PI_PY, CODE / 'train_stage.py', '--method', method, '--suite', suite,
                   '--task', task, '--steps', steps, '--root', root]
        if preflight:
            command.append('--preflight')
        while True:
            try:
                run_logged(command, stage / 'train.log', environment())
                break
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 75:
                    raise
                wait_for_gpus()
                status('training', method=method, suite=suite, task=task, gpus=GPUS,
                       stage_dir=str(stage), target_steps=steps)
    trained = json.loads((stage / 'trained.json').read_text())
    if not (stage / 'evaluated.json').exists():
        status('preflight_evaluation' if preflight else 'evaluation', method=method, suite=suite, task=task,
               stage_dir=str(stage), checkpoint=trained['checkpoint'])
        while True:
            try:
                row = evaluate(stage, suite, task, trained['checkpoint'], 4 if preflight else 50)
                break
            except RuntimeError as exc:
                if not str(exc).startswith('GPU contention during evaluation:'):
                    raise
                wait_for_gpus()
    else:
        row = json.loads((stage / 'evaluated.json').read_text())['row']
    if not preflight:
        path = root / method / suite / 'matrix.json'
        matrix = json.loads(path.read_text()) if path.exists() else {}
        matrix[str(task)] = row
        write_json(path, matrix)

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'runner.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Do not replace the live owner's status with a duplicate-launch failure.
        raise SystemExit('This output already has a running queue.')
    (RUN / 'runner.pid').write_text(str(os.getpid()))
    if not (RUN / 'manifest.json').exists():
        status('preparing_data_manifest')
        run_logged([SIM_PY, CODE / 'prepare.py'], RUN / 'prepare.log', environment(sim=True, gpu=GPUS[0]))
    if not (RUN / 'preflight_passed.json').exists():
        # Both batch shapes, parameter carry-over, optimizer reset, and LoRA checkpoint rollout.
        for task in (0, 1):
            train_and_eval(RUN / 'preflight', 'er', SUITES[0], task, steps=2, preflight=True)
        write_json(RUN / 'preflight_passed.json', dict(timestamp=time.time(),
            checks=['four GPU training', 'LoRA-only parameters', '8+8 replay batch',
                    'checkpoint carry-over with fresh optimizer', 'matching-norm checkpoint rollout']))
    for suite in SUITES:
        for method in METHODS:
            for task in range(10):
                train_and_eval(RUN, method, suite, task)
    streams = len(SUITES) * len(METHODS)
    status('completed', training_stages=10*streams, matrix_cells=55*streams, rollout_episodes=2750*streams)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise
