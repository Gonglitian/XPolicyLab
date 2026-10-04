"""CPU-only checks of Slurm boundaries, locking and failure propagation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slurm_runtime as runtime
import lane


class SlurmRuntimeTest(unittest.TestCase):
    def test_requires_single_allocated_gpu(self):
        for env in ({}, {'SLURM_JOB_ID': '12'}, {'SLURM_JOB_ID': '12', 'CUDA_VISIBLE_DEVICES': '0,1'},
                    {'SLURM_JOB_ID': '12', 'CUDA_VISIBLE_DEVICES': '-1'}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises(RuntimeError): runtime.require_slurm()
        for device in ('0', '5', 'GPU-allocated-uuid'):
            with patch.dict(os.environ, {'SLURM_JOB_ID': '12', 'CUDA_VISIBLE_DEVICES': device}, clear=True):
                self.assertEqual(runtime.require_slurm(), '12')

    def test_environment_preserves_slurm_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {'SLURM_JOB_ID': '12', 'CUDA_VISIBLE_DEVICES': 'GPU-allocated-uuid',
                   'CUDA_DEVICE_ORDER': 'FASTEST_FIRST', 'MUJOCO_EGL_DEVICE_ID': '7'}
            for key in ('TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE',
                        'OPENPI_DATA_HOME', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'V1_JAX_CACHE'):
                env[key] = str(Path(tmp) / key)
            with patch.dict(os.environ, env):
                result = lane.environment(sim=True)
            self.assertEqual(result['CUDA_VISIBLE_DEVICES'], 'GPU-allocated-uuid')
            self.assertEqual(result['CUDA_DEVICE_ORDER'], 'FASTEST_FIRST')
            self.assertNotIn('MUJOCO_EGL_DEVICE_ID', result)

    def test_stream_lock_blocks_other_process_and_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'er/libero_spatial'
            script = 'import sys; from slurm_runtime import stream_lock;\nwith stream_lock(sys.argv[1]): pass'
            command = [sys.executable, '-c', script, str(path)]
            env = dict(os.environ, PYTHONPATH=str(lane.CODE))
            with runtime.stream_lock(path):
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Another job', result.stderr)
                with runtime.stream_lock(Path(tmp) / 'sf/libero_spatial'): pass
            self.assertEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)

    def test_status_is_scoped_to_stream_and_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(lane, 'RUN', Path(tmp)):
            with patch.dict(os.environ, {'SLURM_JOB_ID': '12'}): a = lane.Lane('er', 'libero_spatial')
            with patch.dict(os.environ, {'SLURM_JOB_ID': '13'}): b = lane.Lane('er', 'libero_spatial')
            c = lane.Lane('sf', 'libero_spatial')
            self.assertEqual(a.stream, b.stream)
            self.assertNotEqual(a.status_path, b.status_path)
            self.assertNotEqual(a.stream, c.stream)
            self.assertNotEqual(a.stream, lane.Lane('er', 'libero_spatial', 'preflight').stream)

    def test_training_failure_is_not_retried_as_gpu_contention(self):
        with tempfile.TemporaryDirectory() as tmp:
            flow = lane.Lane('er', 'libero_spatial')
            with patch.object(flow, 'status'), patch.object(lane, 'environment', return_value={}), \
                 patch.object(flow, 'run_logged', side_effect=subprocess.CalledProcessError(75, 'trainer')) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    flow.stage(Path(tmp), 'er', 'libero_spatial', 0, 5, 2, True)
                self.assertEqual(run.call_count, 1)

    def test_submit_rejects_invalid_options_before_sbatch(self):
        for args in [('sf', 'libero_spatial', '--preflight'), ('er', 'libero_spatial', '--gpu', '7'),
                     ('sf', 'libero_spatial', '--tasks', '11')]:
            result = subprocess.run(['bash', str(lane.CODE / 'submit.sh'), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)

    def test_retired_launcher_cannot_start_background_lanes(self):
        result = subprocess.run(['bash', str(lane.CODE / 'gate_then_launch.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('retired', result.stderr)

    def test_readiness_rejects_stale_pid(self):
        class Process:
            pid = 123
            def poll(self): return None
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'ready.json'
            p.write_text(json.dumps({'pid': 122, 'job_id': '12', 'port': 12345}))
            with patch.dict(os.environ, {'SLURM_JOB_ID': '12'}), self.assertRaises(RuntimeError):
                runtime.wait_ready(Process(), p, timeout=1)
            p.write_text(json.dumps({'pid': 123, 'job_id': '12', 'port': 12345}))
            with patch.dict(os.environ, {'SLURM_JOB_ID': '12'}):
                self.assertEqual(runtime.wait_ready(Process(), p, timeout=1), 12345)


if __name__ == '__main__': unittest.main()
