"""Atomic Orbax snapshots of the exact trainable filter; no optimizer or frozen weights."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

def payload_files(directory):
    if Path(directory).is_symlink(): raise RuntimeError('Snapshot payload is a symlink')
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink(): raise RuntimeError(f'Snapshot contains symlink: {path}')
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.stat().st_size
    return result

def validate_snapshot(destination, *, task, step):
    destination = Path(destination)
    if destination.is_symlink(): raise RuntimeError('Snapshot must not be a symlink')
    manifest = json.loads((destination / 'manifest.json').read_text())
    if (manifest.get('format') != 'v1-trainable-orbax-1' or manifest.get('task') != task
        or manifest.get('step') != step or manifest.get('complete') is not True):
        raise RuntimeError(f'Invalid trainable snapshot: {destination}')
    if not manifest.get('parameters') or not manifest.get('files'):
        raise RuntimeError(f'Empty trainable snapshot: {destination}')
    if payload_files(destination / 'params') != manifest['files']:
        raise RuntimeError(f'Missing or truncated snapshot files: {destination}')
    return manifest

def save_snapshot(destination, params, trainable_filter, *, task, step, metadata):
    # Lazy imports keep lifecycle decisions usable without importing JAX or using a GPU.
    import flax.traverse_util
    import orbax.checkpoint as ocp
    destination = Path(destination)
    tree = params.filter(trainable_filter).to_pure_dict()
    flat = flax.traverse_util.flatten_dict(tree, sep='/')
    if not flat: raise RuntimeError('Trainable parameter filter is empty')
    schema = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in sorted(flat.items())}
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    if destination.exists():
        existing = validate_snapshot(destination, task=task, step=step)
        if existing.get('schema_sha256') != schema_hash:
            raise RuntimeError('Existing snapshot uses a different trainable parameter schema')
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.trainable-snapshot-', dir=destination.parent) as tmp:
        temporary = Path(tmp)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(temporary / 'params', {'params': tree})
        files = payload_files(temporary / 'params')
        manifest = dict(metadata, format='v1-trainable-orbax-1', complete=True,
            task=task, step=step, schema_sha256=schema_hash, parameters=schema,
            parameter_count=sum(math.prod(v.shape) for v in flat.values()),
            parameter_bytes=sum(math.prod(v.shape) * v.dtype.itemsize for v in flat.values()),
            files=files, created_at=time.time(), purpose='parameter probes; not a resumable training checkpoint')
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        validate_snapshot(temporary, task=task, step=step)
        temporary.rename(destination)
    return manifest
