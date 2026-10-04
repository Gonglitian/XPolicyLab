"""One Slurm allocation, one GPU, one stream: train and evaluate sequentially."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from common import RUN, SUITES, write_json
from paths import CODE, REPO, OPENPI, configured_path, resolve_checkpoint
from slurm_runtime import require_slurm, stream_lock, install_signal_handlers, stop_processes, wait_ready

PI_PY = configured_path('V1_PI_PY')
SIM_PY = configured_path('V1_SIM_PY')
PI_PATHS = os.environ.get('V1_PI_PATHS', str(OPENPI / 'src'))
SIM_PATHS = os.environ['V1_SIM_PATHS']
LIBERO_CONFIG = configured_path('V1_LIBERO_CONFIG')
WS_PATHS = os.environ.get('V1_WS_PATHS', '')
SERVERS = WORKERS = 4

def environment(sim=False):
    require_slurm()
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', WANDB_MODE='disabled',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='0.92',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TOKENIZERS_PARALLELISM='false',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        LIBERO_CONFIG_PATH=str(LIBERO_CONFIG),
        JAX_COMPILATION_CACHE_DIR=str(configured_path('V1_JAX_CACHE')), PYTHONHASHSEED='42')
    ffmpeg = os.environ.get('V1_FFMPEG') or shutil.which('ffmpeg')
    if ffmpeg:
        env['IMAGEIO_FFMPEG_EXE'] = ffmpeg
    paths = (SIM_PATHS if sim else PI_PATHS).split(os.pathsep)
    paths += [str(CODE), *WS_PATHS.split(os.pathsep), str(REPO.parent), str(REPO)]
    env.pop('MUJOCO_EGL_DEVICE_ID', None)  # resolved inside the simulator allocation
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    for key in ('TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE',
                'TRANSFORMERS_CACHE', 'OPENPI_DATA_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME',
                'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'JAX_COMPILATION_CACHE_DIR'):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env

class Lane:
    def __init__(self, method, suite, mode='formal'):
        self.method, self.suite, self.mode = method, suite, mode
        self.root = RUN if mode == 'formal' else RUN / mode
        self.stream = self.root / method / suite
        self.job_id = os.environ.get('SLURM_JOB_ID', 'unsubmitted')
        self.status_path = self.stream / f'status_job_{self.job_id}.json'

    def status(self, phase, **extra):
        record = dict(phase=phase, timestamp=time.time(), pid=os.getpid(), job_id=self.job_id,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), mode=self.mode, **extra)
        write_json(self.status_path, record)
        print('LANE_STATUS', json.dumps(record), flush=True)

    def run_logged(self, command, path, env):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as log:
            log.write('\nCOMMAND ' + json.dumps([str(v) for v in command]) + '\n')
            log.flush()
            proc = subprocess.Popen([str(v) for v in command], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait()
                if code: raise subprocess.CalledProcessError(code, command)
            finally:
                stop_processes([proc])

    def prepare(self):
        # Different streams may share preparation outputs. Serialize writers.
        with (RUN / 'prepare.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            log = RUN / f'prepare_job_{self.job_id}.log'
            if not (RUN / 'bench_tasks.json').exists():
                self.run_logged([SIM_PY, CODE / 'prepare_tasks.py', RUN / 'bench_tasks.json'], log, environment(sim=True))
            if not (RUN / 'manifest.json').exists():
                self.run_logged([PI_PY, CODE / 'prepare_data.py', RUN / 'bench_tasks.json'], log, environment())
            if not (RUN / 'orientation_check.json').exists():
                self.run_logged([PI_PY, CODE / 'orient_dump.py'], log, environment())
                self.run_logged([SIM_PY, CODE / 'orient_check.py'], log, environment(sim=True))
            report = json.loads((RUN / 'orientation_check.json').read_text())
            if not all(report[s]['matches_rotated'] for s in SUITES):
                raise RuntimeError('Dataset orientation check failed')

    def evaluate(self, stage, suite, task, checkpoint, episodes):
        eval_dir = stage / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        servers, clients, handles, ports = [], [], [], []
        # New readiness records per attempt; a restarted job never trusts old ports.
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f'ports_{self.job_id}_', dir=eval_dir) as ready_dir:
            try:
                for index in range(SERVERS):
                    log = (eval_dir / f'server{index}.log').open('a'); handles.append(log)
                    env = environment()
                    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.22'
                    servers.append(subprocess.Popen([str(PI_PY), str(CODE / 'serve.py'), '--checkpoint', str(checkpoint),
                        '--port', '0', '--ready-file', str(Path(ready_dir) / f'{index}.json')],
                        env=env, stdout=log, stderr=subprocess.STDOUT))
                for index, server in enumerate(servers):
                    ports.append(wait_ready(server, Path(ready_dir) / f'{index}.json'))
                if len(set(ports)) != SERVERS:
                    raise RuntimeError(f'Duplicate policy ports: {ports}')
                self.status('evaluation_ready', suite=suite, task=task, ports=ports)
                for worker, port in enumerate(ports):
                    log = (eval_dir / f'client{worker}.log').open('a'); handles.append(log)
                    clients.append(subprocess.Popen([str(SIM_PY), str(CODE / 'evaluate.py'), '--suite', suite,
                        '--stage', str(task), '--worker', str(worker), '--workers', str(WORKERS), '--episodes', str(episodes),
                        '--port', str(port), '--output', str(eval_dir / f'worker{worker}.json')],
                        env=environment(sim=True), stdout=log, stderr=subprocess.STDOUT))
                while any(p.poll() is None for p in clients):
                    for proc in clients:
                        if proc.poll() not in (None, 0):
                            raise RuntimeError(f'Evaluation client failed: {proc.returncode}')
                    for proc in servers:
                        if proc.poll() is not None:
                            raise RuntimeError(f'Policy server exited unexpectedly: {proc.returncode}')
                    time.sleep(1)
                if any(p.returncode for p in clients):
                    raise RuntimeError('Evaluation client failed')
                records = []
                for worker in range(WORKERS):
                    records.extend(json.loads((eval_dir / f'worker{worker}.json').read_text())['episodes'])
                expected = {(i, e) for i in range(task + 1) for e in range(episodes)}
                if len(records) != len(expected) or {(r['task'], r['episode']) for r in records} != expected:
                    raise RuntimeError('Incomplete or duplicate evaluation episodes')
                row = {str(i): sum(r['success'] for r in records if r['task'] == i) / episodes for i in range(task + 1)}
                write_json(stage / 'evaluated.json', dict(row=row, episodes=episodes, total_episodes=len(records),
                    finished_at=time.time(), job_id=self.job_id))
                return row
            finally:
                stop_processes(clients + servers)
                for handle in handles: handle.close()

    def stage(self, root, method, suite, task, steps_per_task, episodes, preflight=False):
        stage = root / method / suite / f'task{task:02d}'
        if not (stage / 'trained.json').exists():
            self.status('training', method=method, suite=suite, task=task, stage_dir=str(stage), preflight=preflight)
            command = [PI_PY, CODE / 'train_stage.py', '--method', method, '--suite', suite, '--task', task,
                       '--steps-per-task', steps_per_task, '--root', root]
            if preflight: command.append('--preflight')
            self.run_logged(command, stage / 'train.log', environment())
        trained = json.loads((stage / 'trained.json').read_text())
        if (stage / 'evaluated.json').exists():
            row = json.loads((stage / 'evaluated.json').read_text())['row']
        else:
            checkpoint = str(resolve_checkpoint(stage.parent, trained['checkpoint'], task))
            # Preserve episode-level files on failure; the worker skips completed episodes.
            self.status('evaluation', method=method, suite=suite, task=task, checkpoint=trained['checkpoint'])
            row = self.evaluate(stage, suite, task, checkpoint, episodes)
        path = root / method / suite / 'matrix.json'
        matrix = json.loads(path.read_text()) if path.exists() else {}
        matrix[str(task)] = row
        write_json(path, matrix)
        print('STAGE_DONE', method, suite, task, json.dumps(row), flush=True)

    def entry_check(self):
        self.status('entry_check')
        for role, python, sim in [('policy', PI_PY, False), ('sim', SIM_PY, True)]:
            self.run_logged([python, CODE / 'slurm_probe.py', '--role', role, '--suite', self.suite,
                             '--output', self.stream / f'{role}_job_{self.job_id}.json'],
                            self.stream / f'{role}_job_{self.job_id}.log', environment(sim=sim))
        self.status('entry_check_passed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['sf', 'er'], required=True)
    p.add_argument('--suite', choices=SUITES, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true', help='ER: two tasks, five steps/task, two episodes/cell')
    modes.add_argument('--entry-check', action='store_true', help='Short GPU/EGL/port check without training')
    p.add_argument('--tasks', type=int, default=10, choices=range(1, 11))
    a = p.parse_args()
    if a.preflight and a.method != 'er': p.error('--preflight requires --method er')
    require_slurm()
    install_signal_handlers()
    RUN.mkdir(parents=True, exist_ok=True)
    mode = 'entry_checks' if a.entry_check else 'preflight' if a.preflight else 'formal'
    lane = Lane(a.method, a.suite, mode)
    with stream_lock(lane.stream):
        try:
            lane.status('starting', method=a.method, suite=a.suite)
            if a.entry_check:
                lane.entry_check()
                return
            lane.prepare()
            for task in range(2 if a.preflight else a.tasks):
                lane.stage(lane.root, a.method, a.suite, task, 5 if a.preflight else 10000,
                           2 if a.preflight else 50, preflight=a.preflight)
            lane.status('preflight_passed' if a.preflight else 'completed', tasks=2 if a.preflight else a.tasks)
        except SystemExit as exc:
            lane.status('interrupted', exit_code=exc.code)
            raise
        except Exception as exc:
            lane.status('failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
