"""CPU-only validation of task completion and finalized checkpoint selection."""
import json
from pathlib import Path


def stage_action(stage, task, steps_per_task, episodes):
    stage = Path(stage)
    trained_path, evaluated_path = stage / 'trained.json', stage / 'evaluated.json'
    if not trained_path.exists():
        if evaluated_path.exists():
            raise RuntimeError(f'Evaluated task has no training record: {stage}')
        return 'train'
    trained = json.loads(trained_path.read_text())
    if trained.get('end_step') != (task + 1) * steps_per_task:
        raise RuntimeError(f'Training length/config mismatch: {trained_path}')
    if not evaluated_path.exists(): return 'evaluate'
    evaluated = json.loads(evaluated_path.read_text())
    if evaluated.get('episodes') != episodes or set(evaluated.get('row', {})) != {str(i) for i in range(task + 1)}:
        raise RuntimeError(f'Evaluation protocol/incomplete row: {evaluated_path}')
    if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in evaluated['row'].values()):
        raise RuntimeError(f'Invalid success rates: {evaluated_path}')
    return 'skip'


def latest_complete_step(finalized_steps, directory, begin_step, end_step):
    """Use only Orbax manager's finalized steps; never scan temporary step folders."""
    steps = sorted(int(s) for s in finalized_steps)
    if not steps: return None
    if any(s < begin_step or s > end_step for s in steps):
        raise RuntimeError(f'Checkpoint steps conflict with this task: {steps}, expected {begin_step}..{end_step}')
    latest = steps[-1]
    checkpoint = Path(directory) / str(latest)
    if not all((checkpoint / part).is_dir() for part in ('params', 'train_state')):
        raise RuntimeError(f'Finalized checkpoint is damaged: {checkpoint}; refusing to restart from scratch')
    return latest


def check_predecessor(stream, job_id):
    if not job_id: return None
    if not str(job_id).isdigit(): raise ValueError('Invalid predecessor job ID')
    path = Path(stream) / f'status_job_{job_id}.json'
    if not path.is_file():
        raise RuntimeError(f'Predecessor never recorded a resumable state: {path}')
    record = json.loads(path.read_text())
    if record.get('job_id') != str(job_id): raise RuntimeError(f'Incorrect predecessor record: {path}')
    if record.get('phase') in ('failed', 'blocked_previous_failure'):
        raise RuntimeError(f'Predecessor failed; inspect its logs before manual resubmission: {record.get("error")}')
    allowed = {'starting', 'training', 'evaluation', 'evaluation_ready', 'interrupted',
               'completed', 'preflight_passed', 'entry_check', 'entry_check_passed', 'chain_probe_saved', 'chain_probe_passed'}
    if record.get('phase') not in allowed: raise RuntimeError(f'Unknown predecessor state: {record}')
    return record
