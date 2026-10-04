"""CPU-only relocation checks; no JAX, simulator, or model loading."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
import common


class PortablePathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def checkpoint(self, stream):
        p = stream / 'task00/checkpoints/pi05_libero_cl/stage/10000'
        (p / 'params').mkdir(parents=True)
        (p / 'train_state').mkdir()
        return p

    def test_relative_checkpoint_survives_move(self):
        old = self.root / 'old/sf/libero_spatial'
        checkpoint = self.checkpoint(old)
        value = paths.relative_path(checkpoint, old)
        new = self.root / 'new/sf/libero_spatial'
        new.parent.mkdir(parents=True)
        shutil.move(old, new)
        self.assertEqual(paths.resolve_checkpoint(new, value, 0), new / value)

    def test_legacy_checkpoint_prefers_new_stream_even_if_old_exists(self):
        old = self.checkpoint(self.root / 'old/sf/libero_spatial')
        new = self.root / 'new/sf/libero_spatial'
        expected = self.checkpoint(new)
        self.assertEqual(paths.resolve_checkpoint(new, str(old), 0), expected)
        shutil.rmtree(expected / 'train_state')
        with self.assertRaises(FileNotFoundError):
            paths.resolve_checkpoint(new, str(old), 0)

    def test_checkpoint_rejects_wrong_stream_task_and_escape(self):
        stream = self.root / 'sf/libero_spatial'
        self.checkpoint(stream)
        for value in ['/old/er/libero_spatial/task00/checkpoints/stage/10000',
                      'task01/checkpoints/stage/10000', '../outside',
                      'task00/checkpoints/../../../outside']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                paths.resolve_checkpoint(stream, value, 0)

    def test_data_relative_and_legacy_paths(self):
        data = self.root / 'new-data'
        file = data / 'data/chunk-001/episode_001272.parquet'
        file.parent.mkdir(parents=True)
        file.touch()
        with patch.object(paths, 'DATA', data):
            self.assertEqual(paths.resolve_data_path('data/chunk-001/episode_001272.parquet'), file)
            self.assertEqual(paths.resolve_data_path('/old/libero/data/chunk-001/episode_001272.parquet'), file)
            self.assertEqual(paths.resolve_data_path(str(file)), file)
            with self.assertRaises(ValueError): paths.resolve_data_path('../outside.parquet')
            with self.assertRaises(FileNotFoundError): paths.resolve_data_path('missing.parquet')
            with self.assertRaises(ValueError): paths.resolve_data_path('/unknown/layout.parquet')

    def test_explicit_legacy_data_root(self):
        file = self.root / 'custom.parquet'
        file.touch()
        with patch.object(paths, 'DATA', self.root), patch.dict(os.environ, {'V1_LEGACY_DATA_ROOT': '/legacy'}):
            self.assertEqual(paths.resolve_data_path('/legacy/custom.parquet'), file)

    def test_symlink_escape_is_rejected(self):
        data = self.root / 'data'
        data.mkdir()
        (data / 'escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): paths.under(data, 'escape/file')

    def test_python_executable_symlink_is_preserved(self):
        executable = self.root / 'python'
        executable.symlink_to(sys.executable)
        with patch.dict(os.environ, {'V1_TEST_PY': str(executable)}):
            self.assertEqual(paths.configured_path('V1_TEST_PY'), executable)

    def test_stream_carries_metadata_without_old_run(self):
        run, data = self.root / 'run', self.root / 'data'
        file = data / 'data/chunk-000/episode_000000.parquet'
        file.parent.mkdir(parents=True); file.touch()
        norm = run / 'norm/libero_spatial/norm_stats.json'
        norm.parent.mkdir(parents=True); norm.write_text('{"preserved": true}')
        common.write_json(run / 'manifest.json', {'libero_spatial': [{'episode_files': [str(file)]}]})
        stream = run / 'er/libero_spatial'
        with patch.object(common, 'RUN', run), patch.object(common, 'DATA', data), patch.object(paths, 'DATA', data):
            common.ensure_stream_metadata(stream, 'libero_spatial')
            new = self.root / 'moved/er/libero_spatial'
            new.parent.mkdir(parents=True); shutil.move(stream, new)
            shutil.rmtree(run)
            common.ensure_stream_metadata(new, 'libero_spatial')
            self.assertEqual(common.load_manifest(new)['libero_spatial'][0]['episode_files'],
                             ['data/chunk-000/episode_000000.parquet'])
            self.assertEqual((new / 'metadata/norm/libero_spatial/norm_stats.json').read_text(), '{"preserved": true}')

    def test_lane_evaluation_resolves_checkpoint_in_current_stream(self):
        from lane import Lane
        root = self.root / 'runs'
        stream = root / 'sf/libero_spatial'
        expected = self.checkpoint(stream)
        stage = stream / 'task00'
        common.write_json(stage / 'trained.json', {'checkpoint': '/old/sf/libero_spatial/task00/checkpoints/pi05_libero_cl/stage/10000'})
        lane = Lane('sf', 'libero_spatial')
        with patch.object(lane, 'status'), patch.object(lane, 'evaluate', return_value={'0': 1.0}) as evaluate:
            lane.stage(root, 'sf', 'libero_spatial', 0, 10000, 50)
        self.assertEqual(evaluate.call_args.args[3], str(expected))
        self.assertEqual(json.loads((stream / 'matrix.json').read_text()), {'0': {'0': 1.0}})

    def test_evaluated_stage_skips_missing_pruned_checkpoint(self):
        from lane import Lane
        root = self.root / 'runs'
        stage = root / 'sf/libero_spatial/task00'
        common.write_json(stage / 'trained.json', {'checkpoint': 'task00/checkpoints/pruned/10000'})
        common.write_json(stage / 'evaluated.json', {'row': {'0': 0.98}})
        lane = Lane('sf', 'libero_spatial')
        with patch.object(lane, 'evaluate') as evaluate, patch.object(lane, 'run_logged') as train:
            lane.stage(root, 'sf', 'libero_spatial', 0, 10000, 50)
        evaluate.assert_not_called()
        train.assert_not_called()

    def test_configs_are_side_effect_free_and_honor_overrides(self):
        for profile in ('labserver_env.sh', 'bcc_env.sh'):
            with self.subTest(profile=profile):
                env = {k: v for k, v in os.environ.items() if not k.startswith('V1_')}
                env.update(V1_ASSETS=str(self.root / 'assets with spaces'), V1_RUN=str(self.root / 'run with spaces'),
                           V1_OPENPI=str(self.root / 'custom openpi'))
                result = subprocess.check_output(['bash', '-c',
                    'source "$1"; printf "%s\n" "$V1_RUN" "$TMPDIR" "$V1_PI_PATHS" "$V1_REPO"',
                    'test', str(paths.CODE / profile)], env=env, text=True).splitlines()
                self.assertEqual(result[0], env['V1_RUN'])
                self.assertTrue(result[1].startswith(env['V1_ASSETS']))
                self.assertTrue(result[2].startswith(env['V1_OPENPI'] + '/src:'))
                self.assertEqual(Path(result[3]), paths.CODE.parents[1])
                self.assertFalse((self.root / 'run with spaces').exists())
                self.assertFalse((self.root / 'assets with spaces').exists())


if __name__ == '__main__':
    unittest.main()
