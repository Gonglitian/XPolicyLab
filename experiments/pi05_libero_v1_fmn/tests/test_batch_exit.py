import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path

class BatchExitTest(unittest.TestCase):
    def test_exit_codes(self):
        script = Path(__file__).resolve().parents[1] / 'labserver.sbatch'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'slurm').mkdir()
            binary=root/'bin'; binary.mkdir()
            fake=binary/'srun'; fake.write_text('#!/bin/bash\nexit "$TEST_SRUN_EXIT"\n'); fake.chmod(0o755)
            config=root/'config.sh'; config.write_text('v1_prepare_dirs() { :; }\nexport V1_CONDA_ENV=""\n')
            for code in (0,99,143):
                job=str(91000+code)
                env=dict(os.environ,V1_RUN=str(root),V1_CODE=str(script.parent),V1_MACHINE_CONFIG=str(config),
                         V1_PI_PY=sys.executable,V1_CODE_COMMIT='fixture',SLURM_JOB_ID=job,TEST_SRUN_EXIT=str(code),
                         PATH=str(binary)+os.pathsep+os.environ['PATH'])
                result=subprocess.run(['bash',str(script)],env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,code,result.stderr)
                record=json.loads((root/'slurm'/f'exit_{job}.json').read_text())
                self.assertEqual(record['batch_exit_code'],code)
                self.assertEqual(record['job_id'],job)
