"""Portable Linux deployment and experiment dispatch (stdlib-only bootstrap)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCES = json.loads((HERE / 'sources.json').read_text())
MODELS = ('pi05', 'xvla', 'gr00t')
BENCHMARKS = ('libero', 'robocasa')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')

def selection(raw, choices):
    values = list(choices) if raw == 'all' else raw.split(',')
    if not values or len(set(values)) != len(values) or not set(values) <= set(choices):
        raise ValueError(f'Choose a comma-separated subset of {choices}, or all: {raw}')
    return values

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

class Commands:
    def __init__(self, dry):
        self.dry = dry

    def run(self, *args, env=None, cwd=None):
        command = [str(a) for a in args]
        print('+ ' + shlex.join(command), flush=True)
        if not self.dry:
            subprocess.run(command, check=True, env=env, cwd=cwd)

def environment(root, model=None, benchmark=None):
    env = dict(os.environ)
    paths = [str(REPO)]
    if model == 'pi05':
        paths += [str(root / 'upstreams/openpi/src'), str(root / 'upstreams/robocasa'),
                  str(root / 'upstreams/robosuite')]
    if model == 'gr00t':
        paths.insert(0, str(root / ('upstreams/gr00t-robocasa' if benchmark == 'robocasa'
                                    else 'upstreams/gr00t-libero')))
    if benchmark == 'libero' and model is None:
        paths.insert(0, str(root / 'upstreams/libero'))
    if benchmark == 'robocasa' and model is None:
        paths = [str(root / 'upstreams/robocasa'), str(root / 'upstreams/robosuite'), *paths]
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONNOUSERSITE='1',
               PYTHONUNBUFFERED='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', LIBERO_CONFIG_PATH=str(root / 'config/libero'),
               HF_HOME=str(root / 'cache/huggingface'), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_MODE='disabled')
    return env

def python(root, name):
    return root / 'envs' / name / 'bin/python'

def clone(root, name, commands):
    target = root / 'upstreams' / name
    source = SOURCES[name]
    if not target.exists():
        commands.run('git', 'clone', '--filter=blob:none', '--no-checkout', source['url'], target)
        commands.run('git', '-C', target, 'checkout', '--detach', source['commit'])
    elif not commands.dry:
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != source['commit']:
            raise ValueError(f'{target} has revision {actual}; expected {source["commit"]}. Use a fresh --root.')
        # Preserve user modifications; do not silently reset source trees.
        dirty = subprocess.check_output(['git', '-C', str(target), 'diff', '--name-only', 'HEAD'], text=True)
        if dirty.strip():
            raise ValueError(f'Tracked source changes at {target}; use an unmodified pinned checkout.')

def build(a):
    models = selection(a.models, MODELS)
    benchmarks = selection(a.benchmarks, BENCHMARKS)
    datasets = [] if a.datasets == 'none' else selection(a.datasets, BENCHMARKS)
    if not set(datasets) <= set(benchmarks):
        raise ValueError('--datasets must be a subset of --benchmarks')
    root = a.root.expanduser().resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError('--root must be outside the Git checkout (large assets and environments).')
    commands = Commands(a.dry_run)
    if not a.dry_run:
        if sys.platform != 'linux':
            raise RuntimeError('Deployment targets Linux NVIDIA servers. --dry-run works on other hosts.')
        for tool in (a.conda, 'git', 'nvidia-smi'):
            if not shutil.which(tool):
                raise RuntimeError(f'Missing prerequisite: {tool}; see docs/CODEBASE_DEPLOYMENT.md')
        root.mkdir(parents=True, exist_ok=True)
    needed = set(benchmarks)
    if 'robocasa' in benchmarks or 'pi05' in models:
        needed |= {'robocasa', 'robosuite'}  # OpenPI fork imports RoboCasa at config import.
    if 'pi05' in models:
        needed.add('openpi')
    if 'gr00t' in models:
        needed |= {'gr00t-' + b for b in benchmarks}
    for source in sorted(needed):
        clone(root, source, commands)
    profiles = ['bootstrap', *models, *benchmarks]
    for profile in dict.fromkeys(profiles):
        prefix = root / 'envs' / profile
        version = '3.11' if profile in {'pi05', 'bootstrap'} else '3.10'
        if not (prefix / 'bin/python').exists():
            commands.run(a.conda, 'create', '--yes', '--prefix', prefix, f'python={version}', 'pip')
    boot = python(root, 'bootstrap')
    commands.run(boot, '-m', 'pip', 'install', 'uv==0.8.22', 'huggingface-hub>=0.30,<1', 'google-crc32c==1.7.1', 'pyyaml==6.0.2')

    def pip(profile, *packages, no_deps=False):
        args = [boot, '-m', 'uv', 'pip', 'install', '--python', python(root, profile)]
        commands.run(*args, *(['--no-deps'] if no_deps else []), *packages)

    for profile in dict.fromkeys([*models, *benchmarks]):
        if profile == 'pi05':
            env = environment(root, 'pi05')
            env['UV_PROJECT_ENVIRONMENT'] = str(root / 'envs/pi05')
            commands.run(boot, '-m', 'uv', 'sync', '--frozen', '--no-dev', '--project',
                         root / 'upstreams/openpi', '--python', python(root, 'pi05'), env=env)
            pip(profile, '-r', HERE / 'requirements-robocasa-imports.txt')
        elif profile == 'xvla':
            pip(profile, 'torch==2.1.2', 'torchvision==0.16.2', '--index-url', 'https://download.pytorch.org/whl/cu121')
            pip(profile, '-r', REPO / 'policy/X_VLA/xvla/requirements.txt', 'transformers==4.51.3', 'opencv-python-headless==4.11.0.86')
        elif profile == 'gr00t':
            pip(profile, 'torch==2.5.1', 'torchvision==0.20.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
            # The two policy source forks are selected via PYTHONPATH, never installed over each other.
            pip(profile, '-r', HERE / 'requirements-gr00t.txt')
            pip(profile, 'setuptools', 'wheel', 'packaging', 'ninja')
            pip(profile, '--no-build-isolation', 'flash-attn==2.8.3')
            pip(profile, '--no-build-isolation', 'git+https://github.com/facebookresearch/pytorch3d.git@89653419d0973396f3eff1a381ba09a07fffc2ed')
        elif profile == 'libero':
            pip(profile, '-r', HERE / 'requirements-libero.txt')
            pip(profile, '-e', str(root / 'upstreams/libero'), no_deps=True)
        elif profile == 'robocasa':
            pip(profile, '-r', HERE / 'requirements-robocasa-imports.txt', 'torch==2.7.1', 'torchvision==0.22.1',
                'lerobot==0.3.3', 'pygame', 'pynput', 'hidapi', 'tianshou==0.4.10')
            pip(profile, '-e', str(root / 'upstreams/robosuite'), '-e', str(root / 'upstreams/robocasa'), no_deps=True)
        pip(profile, '-r', HERE / 'requirements-common.txt')
        pip(profile, '-e', str(REPO), no_deps=True)
        if not commands.dry:
            lock = subprocess.check_output([str(boot), '-m', 'uv', 'pip', 'freeze', '--python', str(python(root, profile))], text=True)
            (root / 'config').mkdir(exist_ok=True)
            (root / 'config' / f'{profile}-installed.txt').write_text(lock)
    config = dict(schema=1, root=str(root), repo=str(REPO), models=models, benchmarks=benchmarks,
                  source_revisions=SOURCES, built_at=time.time())
    if not commands.dry:
        commands.run(boot, HERE / 'deploy.py', 'configure', '--root', root, '--benchmarks', ','.join(benchmarks))
        old = root / 'deployment.json'
        if old.exists():
            prior = json.loads(old.read_text())
            config['models'] = sorted(set(prior['models']) | set(models))
            config['benchmarks'] = sorted(set(prior['benchmarks']) | set(benchmarks))
        # Install completion is recorded separately from assets completion.
        config['assets_ready'] = False
        save(old, config)
    for model in models:
        if a.checkpoints != 'none':
            commands.run(boot, HERE / 'assets.py', 'checkpoints', '--root', root, '--model', model,
                         '--benchmarks', ','.join(benchmarks), '--kind', a.checkpoints)
    if 'robocasa' in benchmarks:
        commands.run(python(root, 'robocasa'), HERE / 'assets.py', 'kitchen', '--root', root,
                     env=environment(root, benchmark='robocasa'))
    for benchmark in datasets:
        args = ['--root', root]
        if benchmark == 'robocasa':
            args += ['--tasks', a.robocasa_tasks, '--split', a.split, '--source', a.source]
        commands.run(python(root, benchmark), HERE / 'assets.py', benchmark, *args,
                     env=environment(root, benchmark=benchmark))
    for profile in models:
        benchmark = benchmarks[0]
        module = {'pi05': 'openpi.training.config', 'xvla': 'XPolicyLab.policy.X_VLA.model',
                  'gr00t': 'gr00t.model.policy'}[profile]
        env = environment(root, model=profile, benchmark=benchmark)
        env['CUDA_VISIBLE_DEVICES'] = ''  # Import check must not reserve a GPU.
        commands.run(python(root, profile), '-c', f'import importlib; importlib.import_module({module!r}); print("IMPORT_OK")', env=env)
    if not commands.dry:
        config['assets_ready'] = True
        save(root / 'deployment.json', config)
    print(f'Deployment config: {root / "deployment.json"}', flush=True)

def configure(root, benchmarks):
    # The runtime helper owns env_cfg in the checkout's parent workspace.
    sys.path.insert(0, str(REPO))
    from XPolicyLab.policy.GR00T_N15.setup_workspace import register
    for benchmark in benchmarks:
        register(REPO.parent, 'libero_franka' if benchmark == 'libero' else 'robocasa_panda_omron')
    if 'libero' in benchmarks:
        source = root / 'upstreams/libero/libero/libero'
        config = dict(benchmark_root=str(source), bddl_files=str(source / 'bddl_files'),
                      init_states=str(source / 'init_files'), datasets=str(root / 'datasets/libero'),
                      assets=str(source / 'assets'))
        # JSON is valid YAML, avoiding a bootstrap dependency on PyYAML.
        save(root / 'config/libero/config.yaml', config)
    robocasa = root / 'upstreams/robocasa/robocasa'
    if robocasa.exists():
        private = robocasa / 'macros_private.py'
        content = f'DATASET_BASE_PATH = {str(root / "datasets/robocasa")!r}\n'
        if private.exists() and private.read_text() != content:
            raise ValueError(f'Refusing to replace existing {private}')
        private.write_text(content)

def checkpoint(root, model, benchmark):
    suffix = {('pi05','libero'): 'pi05-libero', ('pi05','robocasa'): 'robocasa/pi05_pretrain_human300/multitask_learning/75000',
              ('xvla','libero'): 'xvla-libero', ('gr00t','libero'): 'gr00t-libero-combined',
              ('gr00t','robocasa'): 'robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000'}
    if (model, benchmark) not in suffix:
        raise ValueError('X-VLA + RoboCasa365 has no validated checkpoint/adapter recipe in this codebase.')
    return root / 'checkpoints' / suffix[(model, benchmark)]

def run(a):
    config_path = a.config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    root = Path(config['root'])
    commands = Commands(a.dry_run)
    if a.mode == 'baseline':
        if a.model != 'pi05' or a.benchmark != 'libero':
            raise ValueError('Formal SF/ER queue currently supports pi05 + libero only; other combinations use eval.')
        suites = selection(a.suites, SUITES)
        methods = selection(a.methods, ('sf', 'er'))
        gpus = a.gpus.split(',')
        if len(gpus) != 4 or len(set(gpus)) != 4 or not all(v.isdigit() for v in gpus):
            raise ValueError('Baseline requires exactly four distinct GPU indices, e.g. --gpus 0,1,2,3')
        env = environment(root, 'pi05')
        env.update(XPL_DEPLOY_CONFIG=str(config_path), XPL_RUN_DIR=str(a.output.resolve()),
                   XPL_GPUS=a.gpus, XPL_SUITES=','.join(suites), XPL_METHODS=','.join(methods),
                   XPL_PORT_BASE=str(a.port))
        code = REPO / 'experiments/pi05_libero'
        launch = dict(mode='baseline', suites=suites, methods=methods, seed=42, steps=10000,
                      config=str(config_path), root=str(root))
        if a.status:
            require_paths([a.output / 'launch.json', a.output / 'status.json'])
            if json.loads((a.output / 'launch.json').read_text()) != launch:
                raise ValueError('Status selection must match the original launch --suites/--methods.')
        else:
            preserve_launch(a.output, launch, a.dry_run)
        if not a.dry_run:
            require_paths([python(root, 'pi05'), python(root, 'libero'), root / 'checkpoints/cl_base/pi05_base/params/_METADATA',
                           root / 'datasets/libero', root / 'config/libero/config.yaml'])
        commands.run(python(root, 'pi05'), '-u', code / ('estimate.py' if a.status else 'runner.py'), env=env)
        return
    if a.model not in config['models'] or a.benchmark not in config['benchmarks']:
        raise ValueError('This model/benchmark was not installed; rerun build_env.sh with the selection.')
    ckpt = a.checkpoint.resolve() if a.checkpoint else checkpoint(root, a.model, a.benchmark)
    if a.model == 'xvla' and a.benchmark == 'robocasa':
        raise ValueError('X-VLA RoboCasa365 evaluation is not validated/supported by this entry point.')
    policy = {'pi05':'Pi_05', 'xvla':'X_VLA', 'gr00t':'GR00T_N15'}[a.model]
    robot = 'libero_franka' if a.benchmark == 'libero' else 'robocasa_panda_omron'
    server_env = environment(root, a.model, a.benchmark)
    client_env = environment(root, benchmark=a.benchmark)
    server_env['CUDA_VISIBLE_DEVICES'] = str(a.policy_gpu)
    client_env.update(CUDA_VISIBLE_DEVICES=str(a.env_gpu), MUJOCO_EGL_DEVICE_ID=str(a.env_gpu))
    chunk = (dict(pi05=5, xvla=30, gr00t=1) if a.benchmark == 'libero' else dict(pi05=5, gr00t=16))[a.model]
    if a.model == 'gr00t' and a.benchmark == 'libero':
        chunk = 1
    cfg = dict(policy_name=policy, protocol='ws', host='127.0.0.1', port=a.port,
               bench_name='LIBERO' if a.benchmark == 'libero' else 'RoboCasa365',
               task_name=a.suite if a.benchmark == 'libero' else a.task, ckpt_name=str(ckpt),
               model_path=str(ckpt), env_cfg_type=robot, action_type='ee', seed=a.seed,
               device='cuda', steps=10, denoising_steps=4, ws_ping_timeout_s=600)
    if a.model == 'gr00t':
        cfg['gr00t_root'] = str(root / f'upstreams/gr00t-{a.benchmark}')
        if a.benchmark == 'libero':
            cfg.update(libero_suite=a.suite, data_config='XPolicyLab.policy.GR00T_N15.libero_combined:LiberoCombinedDataConfig')
    output = a.output.resolve()
    preserve_launch(output, dict(mode='eval', model=a.model, benchmark=a.benchmark, checkpoint=str(ckpt),
                                suite=a.suite, task=a.task, task_id=a.task_id, episodes=a.episodes,
                                seed=a.seed, chunk=chunk, split=a.split, source=a.source), a.dry_run)
    config_file = output / 'policy.json'
    server = [python(root, a.model), REPO / 'setup_policy_server.py', '--config_path', config_file]
    client = [python(root, a.benchmark), '-m', 'XPolicyLab.benchmarks.' +
              ('libero' if a.benchmark == 'libero' else 'robocasa365') + '.client',
              '--port', a.port, '--episodes', a.episodes, '--seed', a.seed, '--output', output / 'result.json']
    if a.benchmark == 'libero':
        client += ['--suite', a.suite, '--task-id', a.task_id, '--action-chunk-steps', chunk]
    else:
        client += ['--task', a.task, '--split', a.split, '--source-kind', a.source, '--action-steps', chunk]
    if a.video:
        client += ['--video-dir', output / 'videos']
    if a.dry_run:
        print(json.dumps(cfg, indent=2))
        commands.run(*server)
        commands.run(*client)
        return
    if (output / 'result.json').exists():
        raise ValueError(f'{output}/result.json exists; choose a new output directory.')
    require_paths([python(root,a.model), python(root,a.benchmark), ckpt])
    save(config_file, cfg)
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', a.port)) == 0:
            raise RuntimeError(f'Port {a.port} is occupied; choose --port.')
    proc = None
    try:
        with (output / 'server.log').open('a') as log:
            proc = subprocess.Popen([str(v) for v in server], env=server_env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'Policy server failed; see {output / "server.log"}')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', a.port)) == 0:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Policy server did not become ready in 15 minutes')
                time.sleep(2)
            commands.run(*client, env=client_env, cwd=REPO)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

def preserve_launch(output, launch, dry):
    path = output / 'launch.json'
    if path.exists() and json.loads(path.read_text()) != launch:
        raise ValueError(f'Output belongs to a different experiment: {output}. Choose a new --output.')
    if not dry and not path.exists():
        save(path, launch)

def require_paths(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing deployment assets: ' + ', '.join(missing))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('configure', help=argparse.SUPPRESS)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--benchmarks', required=True)
    p.set_defaults(func=lambda a: configure(a.root.resolve(), selection(a.benchmarks, BENCHMARKS)))
    p = sub.add_parser('build', help='Conda + pinned sources + checkpoints + datasets')
    p.add_argument('--root', type=Path, required=True, help='External persistent workspace; no spaces restriction')
    p.add_argument('--conda', default='conda', help='Conda executable or absolute path')
    p.add_argument('--models', default='all')
    p.add_argument('--benchmarks', default='all')
    p.add_argument('--datasets', default='libero', help='none, libero, robocasa, or all')
    p.add_argument('--checkpoints', choices=('base','eval','both','none'), default='both')
    p.add_argument('--robocasa-tasks', default='OpenDrawer', help='Comma-separated tasks; never downloads all 365 implicitly')
    p.add_argument('--split', choices=('pretrain','target'), default='pretrain')
    p.add_argument('--source', choices=('human','mimicgen'), default='human')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=build)
    p = sub.add_parser('run', help='Checkpoint evaluation or formal pi05 LIBERO SF/ER')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--mode', choices=('eval','baseline'), default='eval')
    p.add_argument('--model', choices=MODELS, default='pi05')
    p.add_argument('--benchmark', choices=BENCHMARKS, default='libero')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--suite', choices=SUITES, default='libero_spatial')
    p.add_argument('--task-id', type=int, default=0)
    p.add_argument('--task', default='OpenDrawer')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--policy-gpu', type=int, default=0)
    p.add_argument('--env-gpu', type=int, default=0)
    p.add_argument('--port', type=int, default=6321)
    p.add_argument('--split', choices=('pretrain','target'), default='pretrain')
    p.add_argument('--source', choices=('human','mimicgen'), default='human')
    p.add_argument('--video', action='store_true')
    p.add_argument('--suites', default='all', help='Baseline suites, each starts from base independently')
    p.add_argument('--methods', default='all', help='Baseline sf,er or a single method')
    p.add_argument('--gpus', default='0,1,2,3', help='Baseline: four physical GPU indices')
    p.add_argument('--status', action='store_true', help='Baseline read-only timing summary')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=run)
    a = parser.parse_args()
    if a.command == 'run' and (a.episodes < 1 or not 0 <= a.task_id <= 9 or not 1024 <= a.port <= 65532):
        parser.error('Require episodes>0, task-id in 0..9 and port in 1024..65532')
    a.func(a)

if __name__ == '__main__':
    main()
