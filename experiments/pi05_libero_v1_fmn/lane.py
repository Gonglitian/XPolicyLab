"""One GPU lane: runs its assigned streams task by task (train, then evaluate on the same GPU).
Durable and resumable: finished stages are skipped. Never uses a GPU that another user occupies."""
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

CODE = Path(__file__).resolve().parent
PI_PY = Path(os.environ.get('V1_PI_PY', str(ROOT / 'XPolicyLab-envs/pi05-robocasa/bin/python')))
SIM_PY = Path(os.environ.get('V1_SIM_PY', str(ASSETS / 'envs/xvla-sanity/bin/python')))
# Extra PYTHONPATH entries for the policy and simulator processes (labserver defaults below).
PI_PATHS = os.environ.get('V1_PI_PATHS', ':'.join([str(ROOT / 'XPolicyLab-upstreams/openpi-robocasa/src'),
    '/data1/vla-reasoning/proj/EvoMoE/eval/robocasa/deps/robocasa', '/data1/vla-reasoning/proj/EvoMoE/eval/robocasa/deps/robosuite']))
SIM_PATHS = os.environ.get('V1_SIM_PATHS', '/home/vla-reasoning/proj/autofocus_3d/baselines/libero')
# Delete a task's checkpoint once the next task is trained and evaluated (only the carry-over checkpoint is needed).
PRUNE = os.environ.get('V1_PRUNE_CHECKPOINTS') == '1'
# One policy server per client: the served Model keeps a single observation window, so clients must not share it.
SERVERS = WORKERS = 4

def environment(gpu, sim=False):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu,
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.92',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TOKENIZERS_PARALLELISM='false',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        LIBERO_CONFIG_PATH=os.environ.get('V1_LIBERO_CONFIG', str(ASSETS / 'sanity_checks/20260922/xvla_libero/libero_config')),
        JAX_COMPILATION_CACHE_DIR=os.environ.get('V1_JAX_CACHE', str(RUN / 'jax_cache')), PYTHONHASHSEED='42')
    if Path('/usr/bin/ffmpeg').exists():
        env['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'
    paths = [str(CODE), str(ASSETS / 'envs/xvla-ws-deps'), str(ROOT), str(ROOT / 'XPolicyLab')]
    if sim:
        paths[:0] = SIM_PATHS.split(':')
        env['MUJOCO_EGL_DEVICE_ID'] = gpu
    else:
        paths.insert(0, PI_PATHS.split(':')[0])
        paths.extend(PI_PATHS.split(':')[1:])
    env['PYTHONPATH'] = ':'.join(paths)
    return env

class Lane:
    def __init__(self, gpu):
        self.gpu = gpu
        self.status_path = RUN / f'status_gpu{gpu}.json'

    def status(self, phase, **extra):
        record = dict(phase=phase, timestamp=time.time(), pid=os.getpid(), gpu=self.gpu, **extra)
        write_json(self.status_path, record)
        print('LANE_STATUS', json.dumps(record), flush=True)

    def wait_idle(self):
        free_since, last = None, None
        while True:
            sample = snapshot([self.gpu])
            busy = [g for g in sample['gpus'] if g['used_mib'] >= 1000 or g['utilization'] > 5] + sample['foreign']
            if not busy:
                free_since = free_since or time.monotonic()
                if time.monotonic() - free_since >= 60:
                    return
                time.sleep(10)
                continue
            free_since = None
            if busy != last:
                self.status('waiting_for_idle_gpu', busy=busy)
                last = busy
            time.sleep(60)

    def run_logged(self, command, path, env):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as log:
            log.write('\nCOMMAND ' + json.dumps([str(v) for v in command]) + '\n')
            log.flush()
            subprocess.run([str(v) for v in command], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def evaluate(self, stage, suite, task, checkpoint, episodes):
        self.wait_idle()
        eval_dir = stage / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        servers, clients, handles = [], [], []
        guard = Guard([self.gpu], allowed_root=os.getpid())
        base = 6500 + 10 * int(self.gpu)
        try:
            for s in range(SERVERS):
                port = base + s
                with socket.socket() as probe:
                    assert probe.connect_ex(('127.0.0.1', port)) != 0, f'Port {port} already occupied'
                log = (eval_dir / f'server{s}.log').open('a'); handles.append(log)
                senv = environment(self.gpu)
                senv['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.22'
                servers.append(subprocess.Popen([str(PI_PY), str(CODE / 'serve.py'), '--checkpoint', checkpoint,
                    '--port', str(port)], env=senv, stdout=log, stderr=subprocess.STDOUT))
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
            for w in range(WORKERS):
                log = (eval_dir / f'client{w}.log').open('a'); handles.append(log)
                clients.append(subprocess.Popen([str(SIM_PY), str(CODE / 'evaluate.py'), '--suite', suite,
                    '--stage', str(task), '--worker', str(w), '--workers', str(WORKERS), '--episodes', str(episodes),
                    '--port', str(base + w), '--output', str(eval_dir / f'worker{w}.json')],
                    env=environment(self.gpu, sim=True), stdout=log, stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in clients):
                for p in clients + servers:
                    if p.poll() not in (None, 0):
                        raise RuntimeError(f'Evaluation subprocess {p.pid} failed with {p.returncode}')
                time.sleep(5)
            records = []
            for w in range(WORKERS):
                records.extend(json.loads((eval_dir / f'worker{w}.json').read_text())['episodes'])
            assert len(records) == (task + 1) * episodes, (len(records), task, episodes)
            assert len({(r['task'], r['episode']) for r in records}) == len(records)
            row = {str(i): sum(r['success'] for r in records if r['task'] == i) / episodes for i in range(task + 1)}
            write_json(stage / 'evaluated.json', dict(row=row, episodes=episodes, total_episodes=len(records),
                finished_at=time.time(), gpu_contention=guard.conflict))
            return row
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

    def stage(self, root, method, suite, task, steps_per_task, episodes, preflight=False):
        stage = root / method / suite / f'task{task:02d}'
        if not (stage / 'trained.json').exists():
            self.wait_idle()
            self.status('training', method=method, suite=suite, task=task, stage_dir=str(stage), preflight=preflight)
            command = [PI_PY, CODE / 'train_stage.py', '--method', method, '--suite', suite, '--task', task,
                       '--steps-per-task', steps_per_task, '--root', root]
            if preflight:
                command.append('--preflight')
            while True:
                try:
                    self.run_logged(command, stage / 'train.log', environment(self.gpu))
                    break
                except subprocess.CalledProcessError as exc:
                    if exc.returncode != 75:
                        raise
                    self.status('paused_gpu_contention', method=method, suite=suite, task=task)
                    self.wait_idle()
        trained = json.loads((stage / 'trained.json').read_text())
        if (stage / 'evaluated.json').exists():
            row = json.loads((stage / 'evaluated.json').read_text())['row']
        else:
            for attempt in range(3):
                self.status('evaluation', method=method, suite=suite, task=task, checkpoint=trained['checkpoint'], attempt=attempt)
                try:
                    row = self.evaluate(stage, suite, task, trained['checkpoint'], episodes)
                    break
                except Exception as exc:
                    # A failed evaluation is discarded whole and rerun from scratch; training is never redone.
                    import shutil
                    bad = stage / f'evaluation_failed_{attempt}'
                    shutil.rmtree(bad, ignore_errors=True)
                    (stage / 'evaluation').rename(bad)
                    write_json(bad / 'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
                    if attempt == 2:
                        raise
        path = root / method / suite / 'matrix.json'
        matrix = json.loads(path.read_text()) if path.exists() else {}
        matrix[str(task)] = row
        write_json(path, matrix)
        previous = root / method / suite / f'task{task - 1:02d}'
        if PRUNE and task > 0 and (previous / 'evaluated.json').exists() and (previous / 'checkpoints').exists():
            import shutil
            shutil.rmtree(previous / 'checkpoints')
            print('PRUNED', previous / 'checkpoints', flush=True)
        print('STAGE_DONE', method, suite, task, json.dumps(row), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', required=True)
    p.add_argument('--streams', required=True, help='comma list of method:suite')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    lane = Lane(a.gpu)
    lock = (RUN / f'lane_gpu{a.gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (RUN / f'lane_gpu{a.gpu}.pid').write_text(str(os.getpid()))
    try:
        if a.preflight:
            for task in (0, 1):
                lane.stage(RUN / 'preflight', 'er', a.streams.split(',')[0].split(':')[1], task, 5, 2, preflight=True)
            lane.status('preflight_passed')
            return
        for item in a.streams.split(','):
            method, suite = item.split(':')
            assert method in ('sf', 'er') and suite in SUITES
            for task in range(10):
                lane.stage(RUN, method, suite, task, 10000, 50)
        lane.status('completed', streams=a.streams)
    except Exception as exc:
        lane.status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
