"""Retention ordering, interrupted deletion, snapshot filtering and portability."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import retention as r
from resume import stage_action

class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stream = Path(self.tmp.name) / 'sf/libero_spatial'
        self.stream.mkdir(parents=True)
        r.ensure_policy(self.stream, False)

    def trained(self, task, evaluated=False):
        stage = self.stream / f'task{task:02d}'
        end = (task + 1) * 10000
        for step in (end-1000, end):
            for part in ('params', 'train_state'):
                d = stage / r.LAYOUT / str(step) / part
                d.mkdir(parents=True)
                (d / 'payload').write_bytes(b'full checkpoint test fixture')
        r.write_json(stage / 'trained.json', dict(end_step=end, checkpoint_complete=True,
            checkpoint=f'task{task:02d}/{r.LAYOUT}/{end}'))
        if evaluated:
            r.write_json(stage / 'evaluated.json', dict(episodes=50, row={str(i): 1.0 for i in range(task+1)}))
        return stage

    def test_cleanup_timing_and_final_retained(self):
        first = self.trained(0)
        with self.assertRaises(RuntimeError): r.after_evaluation(self.stream, 0, 10000, 50)
        self.assertTrue((first / r.LAYOUT / '9000').exists())
        r.write_json(first / 'evaluated.json', dict(episodes=50, row={'0': 1.0}))
        r.after_evaluation(self.stream, 0, 10000, 50)
        self.assertFalse((first / r.LAYOUT / '9000').exists())
        self.assertTrue((first / r.LAYOUT / '10000').exists())
        second = self.trained(1)
        r.after_training(self.stream, 1, 10000, 50)
        self.assertFalse((first / r.LAYOUT / '10000').exists())
        self.assertTrue((second / r.LAYOUT / '19000').exists())
        self.assertTrue((second / r.LAYOUT / '20000').exists())
        self.assertEqual(stage_action(first, 0, 10000, 50), 'skip')
        self.assertFalse(r.verify_artifacts(self.stream, 0, 10000))
        self.assertTrue(r.verify_artifacts(self.stream, 1, 10000))
        self.assertTrue((first / 'evaluated.json').exists())

    def test_incomplete_new_save_cannot_delete_previous(self):
        first = self.trained(0, True)
        second = self.trained(1)
        shutil.rmtree(second / r.LAYOUT / '20000/train_state')
        with self.assertRaises(RuntimeError): r.after_training(self.stream, 1, 10000, 50)
        self.assertTrue((first / r.LAYOUT / '10000').exists())

    def test_previous_evaluation_and_snapshot_required(self):
        first = self.trained(0)
        self.trained(1)
        with self.assertRaises(RuntimeError): r.after_training(self.stream, 1, 10000, 50)
        r.write_json(first / 'evaluated.json', dict(episodes=50, row={'0': 1.0}))
        r.write_json(self.stream / r.POLICY, dict(version=1, save_trainable_snapshots=True))
        with self.assertRaises(FileNotFoundError): r.after_training(self.stream, 1, 10000, 50)
        self.assertTrue((first / r.LAYOUT / '10000').exists())

    def test_crash_mid_delete_recovers_and_missing_tail_fails(self):
        first = self.trained(0, True); second = self.trained(1)
        original = shutil.rmtree
        calls = []
        def interrupted(path):
            calls.append(path)
            if len(calls) == 2: raise OSError('simulated job interruption')
            return original(path)
        with patch('retention.shutil.rmtree', interrupted):
            with self.assertRaises(OSError): r.after_training(self.stream, 1, 10000, 50)
        self.assertFalse(r.verify_artifacts(self.stream, 0, 10000))
        self.assertEqual(r.checkpoint_entries(self.stream, 0), [])
        shutil.rmtree(second / r.LAYOUT / '20000')
        with self.assertRaises(RuntimeError): r.verify_artifacts(self.stream, 0, 10000)

    def test_legacy_and_option_changes_are_rejected(self):
        with self.assertRaises(RuntimeError): r.ensure_policy(self.stream, True)
        (self.stream / r.POLICY).unlink()
        self.trained(0, True)
        self.assertIsNone(r.ensure_policy(self.stream, False))
        r.after_training(self.stream, 0, 10000, 50)
        r.after_evaluation(self.stream, 0, 10000, 50)
        self.assertTrue((self.stream / 'task00' / r.LAYOUT / '10000').exists())

    def test_symlinks_cannot_be_cleaned(self):
        stage = self.trained(0, True)
        outside = Path(self.tmp.name) / 'outside'; outside.mkdir()
        sentinel = outside / 'precious'; sentinel.write_text('keep')
        (stage / r.LAYOUT / '8000').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(RuntimeError): r.after_evaluation(self.stream, 0, 10000, 50)
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_ten_task_flow_retains_only_last_and_relocation_works(self):
        for task in range(10):
            self.trained(task, True)
            r.after_training(self.stream, task, 10000, 50)
            r.after_evaluation(self.stream, task, 10000, 50)
        finals = list(self.stream.glob('task*/checkpoints/pi05_libero_cl/stage/[0-9]*'))
        self.assertEqual([p.name for p in finals], ['100000'])
        moved = Path(self.tmp.name) / 'moved/sf/libero_spatial'
        moved.parent.mkdir(parents=True)
        shutil.move(self.stream, moved)
        for task in range(10):
            self.assertEqual(stage_action(moved / f'task{task:02d}', task, 10000, 50), 'skip')
            r.verify_artifacts(moved, task, 10000)
        self.assertTrue((moved / 'task09' / r.LAYOUT / '100000').is_dir())

class SnapshotTest(unittest.TestCase):
    def test_real_orbax_roundtrip_filter_bfloat16_and_atomic_failure(self):
        # Explicit CPU backend: no GPU process is started by this test.
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import jax.numpy as jnp
        import numpy as np
        import flax.nnx as nnx
        import orbax.checkpoint as ocp
        from trainable_snapshot import save_snapshot, validate_snapshot
        params = nnx.State(dict(img={'kernel': nnx.VariableState(nnx.Param, jnp.arange(6, dtype=jnp.float32))},
                                llm={'lora': nnx.VariableState(nnx.Param, jnp.ones(3, dtype=jnp.bfloat16)),
                                     'frozen': nnx.VariableState(nnx.Param, jnp.zeros(7))}))
        filt = nnx.Not(nnx.PathContains('frozen'))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'task00/trainable_snapshot'
            with patch.object(ocp.PyTreeCheckpointer, 'save', side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    save_snapshot(dest, params, filt, task=0, step=10000, metadata={})
            self.assertFalse(dest.exists())
            record = save_snapshot(dest, params, filt, task=0, step=10000, metadata={'base_name':'fixture'})
            self.assertEqual(set(record['parameters']), {'img/kernel', 'llm/lora'})
            self.assertEqual(record['parameter_count'], 9)
            with ocp.PyTreeCheckpointer() as checkpointer:
                value = checkpointer.restore(dest / 'params')['params']
            np.testing.assert_array_equal(value['img']['kernel'], np.arange(6))
            self.assertEqual(str(value['llm']['lora'].dtype), 'bfloat16')
            self.assertNotIn('frozen', value['llm'])
            again = save_snapshot(dest, params, filt, task=0, step=10000, metadata={})
            self.assertEqual(record, again)
            damaged = dest / 'params' / next(iter(record['files']))
            damaged.write_bytes(b'broken')
            with self.assertRaises(RuntimeError): validate_snapshot(dest, task=0, step=10000)


class IntegrationTest(unittest.TestCase):
    setUp = RetentionTest.setUp
    trained = RetentionTest.trained
    def test_lane_keeps_current_intermediates_on_evaluation_failure(self):
        from lane import Lane
        self.trained(0, True)
        second = self.trained(1)
        lane = Lane('sf', 'libero_spatial')
        with patch.object(lane, 'evaluate', side_effect=RuntimeError('simulator failed')):
            with patch.object(lane, 'status'):
                with self.assertRaisesRegex(RuntimeError, 'simulator failed'):
                    lane.stage(self.stream.parents[1], 'sf', 'libero_spatial', 1, 10000, 50)
        self.assertFalse((self.stream / 'task00' / r.LAYOUT / '10000').exists())
        self.assertTrue((second / r.LAYOUT / '19000').exists())
        self.assertTrue((second / r.LAYOUT / '20000').exists())

    def test_snapshots_enabled_keep_ten_snapshots_and_one_full_checkpoint(self):
        os.environ['JAX_PLATFORMS'] = 'cpu'
        import flax.nnx as nnx
        import jax.numpy as jnp
        from trainable_snapshot import save_snapshot, validate_snapshot
        r.write_json(self.stream / r.POLICY, dict(version=1, save_trainable_snapshots=True))
        for task in range(10):
            stage = self.trained(task, True)
            params = nnx.State({'weight': nnx.VariableState(nnx.Param, jnp.array([float(task)]))})
            save_snapshot(stage / 'trainable_snapshot', params, nnx.Param,
                          task=task, step=(task+1)*10000, metadata={})
            r.after_training(self.stream, task, 10000, 50)
            r.after_evaluation(self.stream, task, 10000, 50)
        self.assertEqual(len(list(self.stream.glob('task*/trainable_snapshot/manifest.json'))), 10)
        self.assertEqual(len(list(self.stream.glob('task*/checkpoints/pi05_libero_cl/stage/[0-9]*'))), 1)
        for task in range(10):
            validate_snapshot(self.stream / f'task{task:02d}' / 'trainable_snapshot', task=task, step=(task+1)*10000)

    def test_submission_flag_reaches_both_jobs(self):
        import submit_jobs
        calls = []
        def submit(command, env):
            calls.append((command, env))
            return str(500 + len(calls))
        repo = Path(submit_jobs.__file__).resolve().parents[2]
        env = dict(V1_RUN=str(self.stream / 'submission'), V1_MACHINE_CONFIG=str(repo/'machine.sh'))
        with patch.dict(os.environ, env), patch.object(sys, 'argv', ['submit_jobs.py', 'er', 'libero_spatial',
                '--chain-next', '--save-trainable-snapshots']):
            with patch.object(submit_jobs, 'snapshot', return_value=('testcommit', repo, repo/'machine.sh')):
                with patch.object(submit_jobs.subprocess, 'check_output', side_effect=lambda cmd, **kw: submit(cmd, kw['env'])):
                    submit_jobs.main()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('--save-trainable-snapshots' in command for command, env in calls))

if __name__ == '__main__': unittest.main()
