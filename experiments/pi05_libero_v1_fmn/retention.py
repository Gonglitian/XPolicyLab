"""Stream-owned checkpoint retention. Called only while holding stream.lock."""
import json
import os
from pathlib import Path
import re
import shutil
import time
from resume import stage_action

POLICY = 'retention_policy.json'
LAYOUT = Path('checkpoints/pi05_libero_cl/stage')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def contained(stream, path):
    """Reject every symlink component, even links that currently point inside."""
    stream, path = Path(stream).absolute(), Path(path).absolute()
    relative = path.relative_to(stream)
    if '..' in relative.parts or stream.is_symlink():
        raise RuntimeError(f'Unsafe retention path: {path}')
    current = stream
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f'Symlink in retention path: {current}')
    path.resolve().relative_to(stream.resolve())
    return path

def ensure_policy(stream, snapshots):
    stream = Path(stream)
    path = contained(stream, stream / POLICY)
    expected = dict(version=1, save_trainable_snapshots=bool(snapshots))
    if path.exists():
        if read(path) != expected:
            raise RuntimeError('Keep --save-trainable-snapshots unchanged when resubmitting this stream')
    else:
        # Never opt archived/legacy streams into deletion by merely pointing V1_RUN at them.
        if any(stream.glob('task[0-9][0-9]')):
            if snapshots:
                raise RuntimeError('Cannot backfill snapshots of a legacy stream; use a new V1_RUN')
            print('LEGACY_STREAM: retention disabled; existing checkpoints preserved', flush=True)
            return None
        write_json(path, expected)
    return expected

def policy(stream):
    path = contained(stream, Path(stream) / POLICY)
    if not path.exists(): return None
    value = read(path)
    if value.get('version') != 1 or not isinstance(value.get('save_trainable_snapshots'), bool):
        raise RuntimeError('Unknown retention policy')
    return value

def final_checkpoint(stream, task, steps):
    stage = Path(stream) / f'task{task:02d}'
    record = read(contained(stream, stage / 'trained.json'))
    checkpoint = contained(stream, stage / LAYOUT / str((task + 1) * steps))
    if (record.get('end_step') != (task + 1) * steps
        or record.get('checkpoint') != checkpoint.relative_to(stream).as_posix()
        or record.get('checkpoint_complete') is not True):
        raise RuntimeError(f'No confirmed final save for task {task}')
    for name in ('params', 'train_state'):
        if not contained(stream, checkpoint / name).is_dir():
            raise RuntimeError(f'Final checkpoint is missing {name}: {checkpoint}')
    return checkpoint

def require_snapshot(stream, task, steps):
    from trainable_snapshot import validate_snapshot
    return validate_snapshot(contained(stream, Path(stream) / f'task{task:02d}' / 'trainable_snapshot'),
                             task=task, step=(task + 1) * steps)

def usage_bytes(directory):
    directory = Path(directory)
    if not directory.exists(): return 0
    total = 0
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in files:
            p = Path(root) / name
            if not p.is_symlink():
                total += p.stat().st_blocks * 512
    return total

def audit(stream, event, **extra):
    with (Path(stream) / 'retention.jsonl').open('a') as log:
        log.write(json.dumps(dict(event=event, timestamp=time.time(),
                   job_id=os.environ.get('SLURM_JOB_ID'), **extra)) + '\n')
        log.flush()
        os.fsync(log.fileno())

def sample_disk(stream, event):
    allocated = usage_bytes(stream)
    path = Path(stream) / 'disk_usage.json'
    old = read(path) if path.exists() else {}
    write_json(path, dict(scope='stream only; excludes shared caches, data, base weights and code',
        measurement='allocated bytes sampled at save/cleanup boundaries; not continuous peak',
        sampled_peak_bytes=max(allocated, old.get('sampled_peak_bytes', 0)),
        current_bytes=allocated, event=event, timestamp=time.time()))
    return allocated

def remove_step(stream, candidate, reason):
    candidate = contained(stream, candidate)
    if not candidate.exists(): return
    if not candidate.is_dir(): raise RuntimeError(f'Unexpected checkpoint file: {candidate}')
    # Do not recurse through any links, including malformed checkpoint internals.
    for root, dirs, files in os.walk(candidate, followlinks=False):
        for name in dirs + files:
            contained(stream, Path(root) / name)
    size = usage_bytes(candidate)
    rel = candidate.relative_to(stream).as_posix()
    audit(stream, 'delete_started', path=rel, reason=reason, allocated_bytes=size)
    shutil.rmtree(candidate)
    audit(stream, 'delete_completed', path=rel, reason=reason, allocated_bytes=size)

def checkpoint_entries(stream, task):
    directory = contained(stream, Path(stream) / f'task{task:02d}' / LAYOUT)
    if not directory.exists(): return []
    entries = []
    for item in directory.iterdir():
        # Delete only known step folders, never assets, logs, buffers or arbitrary files.
        if re.fullmatch(r'[0-9]+(?:\.orbax-checkpoint-tmp(?:-.*)?)?', item.name):
            entries.append(contained(stream, item))
    return entries

def after_training(stream, task, steps, episodes):
    """New final save supersedes only its evaluated predecessor."""
    settings = policy(stream)
    if settings is None: return
    final_checkpoint(stream, task, steps)
    if settings['save_trainable_snapshots']:
        require_snapshot(stream, task, steps)
        for temporary in (Path(stream) / f'task{task:02d}').glob('.trainable-snapshot-*'):
            remove_step(stream, temporary, 'completed snapshot supersedes interrupted temporary save')
    if task == 0: return
    previous = Path(stream) / f'task{task-1:02d}'
    if stage_action(previous, task-1, steps, episodes) != 'skip':
        raise RuntimeError('Cannot retire a predecessor before its evaluation completes')
    if settings['save_trainable_snapshots']: require_snapshot(stream, task-1, steps)
    write_json(previous / 'checkpoints_retired.json', dict(superseded_by=task, steps_per_task=steps))
    for item in checkpoint_entries(stream, task-1):
        remove_step(stream, item, f'task{task:02d} final checkpoint confirmed')
    sample_disk(stream, 'after_previous_task_cleanup')

def after_evaluation(stream, task, steps, episodes):
    settings = policy(stream)
    if settings is None: return
    stage = Path(stream) / f'task{task:02d}'
    if stage_action(stage, task, steps, episodes) != 'skip':
        raise RuntimeError('Intermediate cleanup requires completed evaluation')
    final = final_checkpoint(stream, task, steps)
    if settings['save_trainable_snapshots']: require_snapshot(stream, task, steps)
    for item in checkpoint_entries(stream, task):
        if item != final:
            remove_step(stream, item, f'task{task:02d} evaluation completed')
    sample_disk(stream, 'after_evaluation_cleanup')


def verify_artifacts(stream, task, steps):
    """Follow retirement records to a surviving complete checkpoint, never silently trust deletion."""
    settings = policy(stream)
    if settings is None:
        stage = Path(stream) / f'task{task:02d}'
        if (stage / 'evaluated.json').is_file(): return False
        from paths import resolve_checkpoint
        resolve_checkpoint(stream, read(stage / 'trained.json')['checkpoint'], task)
        return True
    if settings['save_trainable_snapshots']: require_snapshot(stream, task, steps)
    marker = contained(stream, Path(stream) / f'task{task:02d}' / 'checkpoints_retired.json')
    if marker.exists():
        value = read(marker)
        if value != dict(superseded_by=task+1, steps_per_task=steps) or task >= 9:
            raise RuntimeError('Invalid checkpoint retirement chain')
        verify_artifacts(stream, task+1, steps)
        # Retry a deletion interrupted after its write-ahead retirement marker.
        for item in checkpoint_entries(stream, task):
            remove_step(stream, item, 'finish interrupted retirement')
        return False
    final_checkpoint(stream, task, steps)
    return True
