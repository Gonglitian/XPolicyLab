"""Submit a stream and optionally one afterany continuation using one Git snapshot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def snapshot(repo, run, config):
    repo = Path(repo).resolve()
    config_relative = Path(config).resolve().relative_to(repo)
    experiment = 'experiments/pi05_libero_v1_fmn'
    if git(repo, 'status', '--porcelain', '--', experiment, str(config_relative)):
        raise RuntimeError('Commit experiment/config changes before submitting; jobs use a committed snapshot')
    commit = git(repo, 'rev-parse', 'HEAD')
    subprocess.run(['git', '-C', str(repo), 'cat-file', '-e', f'{commit}:{config_relative}'], check=True)
    root = Path(run) / 'code_snapshots'
    root.mkdir(parents=True, exist_ok=True)
    final = root / commit
    with (root / 'snapshot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not final.exists():
            with tempfile.TemporaryDirectory(prefix='snapshot-', dir=root) as tmp:
                staging = Path(tmp) / commit
                checkout = staging / 'XPolicyLab'
                checkout.mkdir(parents=True)
                archive = subprocess.Popen(['git', '-C', str(repo), 'archive', commit], stdout=subprocess.PIPE)
                try:
                    subprocess.run(['tar', '-x', '-C', str(checkout)], stdin=archive.stdout, check=True)
                    archive.stdout.close()
                    if archive.wait(): raise RuntimeError('git archive failed')
                finally:
                    if archive.poll() is None: archive.terminate(); archive.wait()
                (staging / 'commit.txt').write_text(commit + '\n')
                staging.rename(final)
        if (final / 'commit.txt').read_text().strip() != commit:
            raise RuntimeError(f'Invalid code snapshot: {final}')
    checkout = final / 'XPolicyLab'
    return commit, checkout, checkout / config_relative


def submit_pair(command, env, chain, submit=None):
    """On partial failure, report the already-live first job instead of hiding it."""
    if submit is None:
        submit = lambda args, environment: subprocess.check_output(args, env=environment, text=True).strip()
    first_env = dict(env)
    first_env.pop('V1_PREDECESSOR_JOB_ID', None)
    first = submit(command, first_env).split(';')[0]
    if not first.isdigit(): raise RuntimeError(f'Unexpected sbatch response: {first}')
    print(f'SUBMITTED job={first}', flush=True)
    second = None
    if chain:
        following_env = dict(first_env, V1_PREDECESSOR_JOB_ID=first)
        try:
            second = submit([command[0], f'--dependency=afterany:{first}', *command[1:]], following_env).split(';')[0]
            if not second.isdigit(): raise RuntimeError(f'Unexpected continuation response: {second}')
        except Exception as exc:
            raise RuntimeError(f'First job {first} is submitted, but continuation submission failed; inspect squeue. {exc}') from exc
        print(f'SUBMITTED continuation={second} dependency=afterany:{first}', flush=True)
    return first, second


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['sf', 'er'])
    parser.add_argument('suite', choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--entry-check', action='store_true')
    mode.add_argument('--chain-check', action='store_true', help='Synthetic handoff only; no model training')
    parser.add_argument('--chain-next', action='store_true', help='Submit exactly one afterany continuation')
    parser.add_argument('--save-trainable-snapshots', action='store_true', help='Keep per-task trainable parameters for probes; default off')
    parser.add_argument('--tasks', type=int, default=10, choices=range(1, 11))
    args = parser.parse_args()
    if args.preflight and args.method != 'er': parser.error('--preflight requires er')
    if args.chain_check and not args.chain_next: parser.error('--chain-check requires --chain-next')
    run = Path(os.environ['V1_RUN']).resolve()
    repo = Path(__file__).resolve().parents[2]
    commit, frozen_repo, config = snapshot(repo, run, os.environ['V1_MACHINE_CONFIG'])
    env = dict(os.environ, V1_REPO=str(frozen_repo), V1_CODE=str(frozen_repo / 'experiments/pi05_libero_v1_fmn'),
               V1_MACHINE_CONFIG=str(config), V1_CODE_COMMIT=commit, V1_RUN=str(run))
    logs = run / 'slurm'; logs.mkdir(parents=True, exist_ok=True)
    command = ['sbatch', '--parsable', '--partition=' + env.get('V1_SLURM_PARTITION', 'gpu'),
               '--time=' + env.get('V1_SLURM_TIME', '3-00:00:00'), '--nodes=1', '--ntasks=1', '--gres=gpu:1',
               '--cpus-per-task=' + env.get('V1_SLURM_CPUS', '12'), '--mem=' + env.get('V1_SLURM_MEM', '60G'),
               f'--job-name=v1_{args.method}_{args.suite}', '--output=' + str(logs / '%x-%j.log'),
               '--chdir=' + env['V1_CODE'], '--export=ALL', str(Path(env['V1_CODE']) / 'labserver.sbatch'),
               '--method', args.method, '--suite', args.suite, '--tasks', str(args.tasks)]
    if args.preflight: command.append('--preflight')
    if args.entry_check: command.append('--entry-check')
    if args.chain_check: command.append('--chain-check')
    if args.save_trainable_snapshots: command.append('--save-trainable-snapshots')
    first, second = submit_pair(command, env, args.chain_next)
    record = dict(first_job=first, continuation_job=second, dependency=f'afterany:{first}' if second else None,
                  commit=commit, code=env['V1_CODE'], run=str(run), method=args.method, suite=args.suite,
                  tasks=args.tasks, preflight=args.preflight, entry_check=args.entry_check,
                  chain_check=args.chain_check, save_trainable_snapshots=args.save_trainable_snapshots, submitted_at=time.time())
    (logs / f'submission_{first}.json').write_text(json.dumps(record, indent=2) + '\n')
    print('SUBMISSION', json.dumps(record), flush=True)


if __name__ == '__main__': main()
