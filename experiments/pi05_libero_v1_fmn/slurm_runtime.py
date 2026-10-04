"""Slurm allocation checks, per-stream locking and process lifecycle helpers."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def require_slurm():
    job = os.environ.get('SLURM_JOB_ID')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not job or not job.isdigit():
        raise RuntimeError('GPU execution requires sbatch/srun (SLURM_JOB_ID missing)')
    devices = [v.strip() for v in visible.split(',') if v.strip()]
    if len(devices) != 1 or devices[0] in ('-1', 'NoDevFiles'):
        raise RuntimeError(f'Expected exactly one Slurm GPU, got CUDA_VISIBLE_DEVICES={visible!r}')
    return job


@contextlib.contextmanager
def stream_lock(stream):
    stream = Path(stream)
    stream.mkdir(parents=True, exist_ok=True)
    with (stream / 'stream.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another job is writing this stream: {stream}') from None
        lock.seek(0); lock.truncate()
        json.dump({'job_id': os.environ.get('SLURM_JOB_ID'), 'pid': os.getpid()}, lock)
        lock.flush()
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    # Never unlink: a new job might already hold this inode's lock.


def install_signal_handlers():
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def stop_processes(processes):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 15
    for proc in processes:
        try:
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()


def wait_ready(process, path, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Policy server exited with {process.returncode}')
        if Path(path).is_file():
            ready = json.loads(Path(path).read_text())
            if ready.get('pid') != process.pid or ready.get('job_id') != os.environ.get('SLURM_JOB_ID'):
                raise RuntimeError(f'Stale policy server readiness record: {path}')
            port = ready['port']
            if not isinstance(port, int) or not 0 < port < 65536:
                raise ValueError(f'Invalid policy port: {port}')
            return port
        time.sleep(0.2)
    raise TimeoutError(f'Policy server startup timed out: {path}')
