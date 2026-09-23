"""CPU-only control-plane tests; no Conda, downloads, GPU or simulator required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

deploy, assets = load('deploy'), load('assets')

class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'workspace with spaces'
        self.config = Path(self.temp.name) / 'deployment.json'
        self.config.write_text(json.dumps(dict(root=str(self.root), models=list(deploy.MODELS), benchmarks=list(deploy.BENCHMARKS))))

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args):
        return subprocess.run([sys.executable, str(HERE/'deploy.py'), *map(str,args)], text=True, capture_output=True)

    def test_build_dry_run_is_side_effect_free(self):
        result = self.cli('build','--root',self.root,'--models','all','--benchmarks','all','--dry-run')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('conda create',result.stdout)
        self.assertIn('flash-attn==2.8.3',result.stdout)
        self.assertIn('checkpoints',result.stdout)
        self.assertFalse(self.root.exists())

    def test_selection_validation(self):
        for value in ('sf,sf','sf,unknown',''):
            with self.assertRaises(ValueError):
                deploy.selection(value,('sf','er'))

    def test_output_identity_protects_resume(self):
        out = self.root/'run'
        deploy.preserve_launch(out,{'method':'sf'},False)
        deploy.preserve_launch(out,{'method':'sf'},False)
        with self.assertRaises(ValueError):
            deploy.preserve_launch(out,{'method':'er'},False)
        self.assertEqual(json.loads((out/'launch.json').read_text()),{'method':'sf'})

    def test_baseline_requires_four_distinct_gpus(self):
        for gpus in ('0,1,2','0,1,2,2','0,1,2,x'):
            result=self.cli('run','--config',self.config,'--mode','baseline','--output',self.root/'out','--gpus',gpus,'--dry-run')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('four distinct',result.stderr)

    def test_distributed_suite_assignment(self):
        result=self.cli('run','--config',self.config,'--mode','baseline','--output',self.root/'out',
                        '--suites','libero_goal,libero_10','--methods','er','--dry-run')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('runner.py',result.stdout)
        self.assertFalse(self.root.exists())

    def test_eval_supported_combinations(self):
        for model,bench in [('pi05','libero'),('xvla','libero'),('gr00t','libero'),('pi05','robocasa'),('gr00t','robocasa')]:
            result=self.cli('run','--config',self.config,'--model',model,'--benchmark',bench,
                            '--output',self.root/'out','--dry-run')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('setup_policy_server.py',result.stdout)
            self.assertIn('client',result.stdout)

    def test_unsupported_pair_fails_before_output(self):
        result=self.cli('run','--config',self.config,'--model','xvla','--benchmark','robocasa',
                        '--output',self.root/'out','--dry-run')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.root.exists())

    def test_status_does_not_create_output(self):
        result=self.cli('run','--config',self.config,'--mode','baseline','--status','--output',self.root/'out')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.root.exists())

    def test_argument_boundaries(self):
        with patch.object(deploy.subprocess,'run') as run:
            with contextlib.redirect_stdout(io.StringIO()):
                deploy.Commands(False).run('python','path with spaces.py','literal;$(not-a-command)')
            self.assertEqual(run.call_args.args[0],['python','path with spaces.py','literal;$(not-a-command)'])
            self.assertNotIn('shell',run.call_args.kwargs)

    def test_archive_path_traversal_rejected(self):
        archive=Path(self.temp.name)/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('../outside','bad')
        with self.assertRaises(ValueError):
            assets.unpack(archive,self.root)
        self.assertFalse((self.root.parent/'outside').exists())

    def test_tar_links_rejected(self):
        archive=Path(self.temp.name)/'bad.tar'
        with tarfile.open(archive,'w') as tar:
            member=tarfile.TarInfo('link')
            member.type=tarfile.SYMTYPE
            member.linkname='/etc/passwd'
            tar.addfile(member)
        with self.assertRaises(ValueError):
            assets.unpack(archive,self.root)

    def test_baseline_sources_have_no_original_server_paths(self):
        for path in (HERE.parents[1]/'experiments/pi05_libero').glob('*.py'):
            text=path.read_text()
            self.assertNotIn('/data2/vla-reasoning',text,path)
            self.assertNotIn('/data1/vla-reasoning',text,path)
            self.assertNotIn('/home/vla-reasoning',text,path)

    def test_eta_uses_selected_streams_and_only_idle_timing(self):
        self.root.mkdir()
        (self.root/'status.json').write_text('{}')
        stage=self.root/'er/libero_goal/task00'
        stage.mkdir(parents=True)
        (stage/'metrics.jsonl').write_text('\n'.join(json.dumps(row) for row in [
            dict(step=100,seconds_per_step=1.0,idle_gpu_timing_valid=True),
            dict(step=150,seconds_per_step=99.0,idle_gpu_timing_valid=False)]))
        (stage/'checkpoints/pi05_libero_cl/stage/168').mkdir(parents=True)
        spec=importlib.util.spec_from_file_location('estimate',HERE.parents[1]/'experiments/pi05_libero/estimate.py')
        module=importlib.util.module_from_spec(spec)
        output=io.StringIO()
        with patch.dict(sys.modules,{
                'common':SimpleNamespace(RUN=self.root,SUITES=['libero_goal'],HORIZONS={'libero_goal':300}),
                'settings':SimpleNamespace(METHODS=['er'])}):
            spec.loader.exec_module(module)
            with contextlib.redirect_stdout(output):
                module.main()
        value=json.loads(output.getvalue())
        self.assertEqual(value['target_updates'],100000)
        self.assertEqual(value['target_cells'],55)
        self.assertEqual(value['completed_updates'],168)
        self.assertEqual(value['batch8_seconds_per_step'],1.0)

if __name__=='__main__':
    unittest.main()
