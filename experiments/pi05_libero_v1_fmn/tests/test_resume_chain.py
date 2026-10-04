"""CPU checks of resume decisions, immutable source snapshots and dependency submission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resume import stage_action, latest_complete_step, check_predecessor
from submit_jobs import submit_pair, snapshot


class ResumeChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def test_stage_resume_decisions_and_protocol_validation(self):
        stage = self.root / 'task00'
        self.assertEqual(stage_action(stage, 0, 10000, 50), 'train')
        self.write(stage / 'trained.json', {'end_step': 10000})
        self.assertEqual(stage_action(stage, 0, 10000, 50), 'evaluate')
        self.write(stage / 'evaluated.json', {'episodes': 50, 'row': {'0': 1.0}})
        self.assertEqual(stage_action(stage, 0, 10000, 50), 'skip')
        with self.assertRaises(RuntimeError): stage_action(stage, 0, 5, 2)
        with self.assertRaises(RuntimeError): stage_action(stage, 0, 10000, 2)
        (stage / 'trained.json').unlink()
        with self.assertRaises(RuntimeError): stage_action(stage, 0, 10000, 50)

    def test_only_latest_finalized_checkpoint_is_selected(self):
        for step in (21000, 22000):
            for part in ('params', 'train_state'): (self.root / str(step) / part).mkdir(parents=True)
        (self.root / '23000.orbax-checkpoint-tmp').mkdir()
        self.assertEqual(latest_complete_step([21000, 22000], self.root, 20000, 30000), 22000)
        self.assertIsNone(latest_complete_step([], self.root, 20000, 30000))
        with self.assertRaises(RuntimeError): latest_complete_step([31000], self.root, 20000, 30000)
        (self.root / '22000/train_state').rmdir()
        with self.assertRaises(RuntimeError): latest_complete_step([21000, 22000], self.root, 20000, 30000)

    def test_predecessor_interrupt_allowed_but_error_blocked(self):
        p = self.root / 'status_job_123.json'
        with self.assertRaises(RuntimeError): check_predecessor(self.root, '123')
        self.write(p, {'job_id': '123', 'phase': 'interrupted', 'exit_code': 143})
        self.assertEqual(check_predecessor(self.root, '123')['phase'], 'interrupted')
        self.write(p, {'job_id': '123', 'phase': 'failed', 'error': 'bad dataset'})
        with self.assertRaises(RuntimeError): check_predecessor(self.root, '123')
        self.assertIsNone(check_predecessor(self.root, None))

    def test_dependency_and_shared_configuration(self):
        calls = []
        def submit(command, env):
            calls.append((command, env)); return str(100 + len(calls))
        command = ['sbatch', '--parsable', 'job.sbatch', '--method', 'er']
        env = {'V1_RUN': str(self.root), 'V1_CODE_COMMIT': 'abc', 'V1_PREDECESSOR_JOB_ID': 'old'}
        self.assertEqual(submit_pair(command, env, True, submit), ('101', '102'))
        self.assertIn('--dependency=afterany:101', calls[1][0])
        self.assertNotIn('V1_PREDECESSOR_JOB_ID', calls[0][1])
        self.assertEqual(calls[1][1]['V1_PREDECESSOR_JOB_ID'], '101')
        self.assertEqual(calls[0][1]['V1_CODE_COMMIT'], calls[1][1]['V1_CODE_COMMIT'])
        self.assertEqual(calls[0][1]['V1_RUN'], calls[1][1]['V1_RUN'])

    def test_partial_submission_failure_preserves_first_job_id(self):
        calls = []
        def submit(command, env):
            calls.append(command)
            if len(calls) == 1: return '101'
            raise RuntimeError('scheduler unavailable')
        with self.assertRaisesRegex(RuntimeError, 'First job 101 is submitted'):
            submit_pair(['sbatch', 'job.sbatch'], {}, True, submit)
        self.assertEqual(len(calls), 2)

    def test_snapshot_uses_committed_repo_and_rejects_dirty_experiment(self):
        repo = self.root / 'repo'; repo.mkdir()
        def git(*args):
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        git('init')
        exp = repo / 'experiments/pi05_libero_v1_fmn'; exp.mkdir(parents=True)
        config = exp / 'machine.sh'; config.write_text('export V1_DATA=/data/test\n')
        (repo / 'adapter.py').write_text('version = 1\n')
        git('add', '.')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'fixture')
        (repo / 'adapter.py').write_text('version = 2\n')
        commit, frozen, _ = snapshot(repo, self.root / 'run', config)
        self.assertEqual((frozen / 'adapter.py').read_text(), 'version = 1\n')
        git('add', 'adapter.py')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'new adapter')
        self.assertNotEqual(commit, git('rev-parse', 'HEAD'))
        self.assertEqual((frozen / 'adapter.py').read_text(), 'version = 1\n')
        config.write_text('changed\n')
        with self.assertRaises(RuntimeError): snapshot(repo, self.root / 'run', config)


if __name__ == '__main__': unittest.main()
