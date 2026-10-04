"""Train one task of a stream. FMN-aligned: the full train state (params, optimizer state, global step) is
carried from the previous task, so one LR schedule spans the stream. Native openpi init/train_step/save."""
import argparse
import functools
import importlib.util
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from common import (OPENPI, RUN, STEPS_PER_TASK, CURRENT_BATCH, REPLAY_PER_TASK, LeRobotTask,
                    make_config, load_norm, write_json, load_manifest, ensure_stream_metadata)
from slurm_runtime import require_slurm
from paths import resolve_checkpoint, relative_path
from resume import latest_complete_step
from retention import ensure_policy, sample_disk
from trainable_snapshot import save_snapshot

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['sf', 'er'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--steps-per-task', type=int, default=STEPS_PER_TASK)
    p.add_argument('--root', type=Path, default=RUN)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--save-trainable-snapshots', action='store_true')
    args = p.parse_args()
    require_slurm()
    import jax
    import jax.numpy as jnp
    import flax.traverse_util
    from torch.utils.data import Subset, ConcatDataset
    from openpi.training import data_loader as dl, sharding, checkpoints
    from openpi.models.model import Observation
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    assert jax.device_count() == 1 and jax.devices()[0].platform == 'gpu', jax.devices()
    spt = args.steps_per_task
    stream = args.root / args.method / args.suite
    retention_settings = ensure_policy(stream, args.save_trainable_snapshots)
    stage = stream / f'task{args.task:02d}'
    stage.mkdir(parents=True, exist_ok=True)
    previous = stream / f'task{args.task-1:02d}'
    begin_step, end_step = spt * args.task, spt * (args.task + 1)
    ensure_stream_metadata(stream, args.suite)
    manifest = load_manifest(stream)[args.suite]
    norms = load_norm(args.suite, stream)
    current = LeRobotTask(manifest[args.task])
    effective_batch = 2 * CURRENT_BATCH if args.method == 'er' and args.task else CURRENT_BATCH
    config = make_config(norms, stage, effective_batch, end_step)
    data_config = config.data.create(config.assets_dirs, config.model)
    transformed = dl.transform_dataset(current, data_config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = dl.TorchDataLoader(transformed, local_batch_size=CURRENT_BATCH, sharding=data_sharding,
                                shuffle=True, num_workers=4, seed=42 + args.task)
    replay = None
    if effective_batch > CURRENT_BATCH:
        subsets = []
        for i in range(args.task):
            memory = json.loads((stream / f'task{i:02d}' / 'buffer.json').read_text())
            old = LeRobotTask(manifest[i])
            assert memory['frames'] == len(old)
            subsets.append(Subset(dl.transform_dataset(old, data_config), memory['indices']))
        replay = dl.TorchDataLoader(ConcatDataset(subsets), local_batch_size=CURRENT_BATCH,
                                    sharding=data_sharding, shuffle=True, num_workers=4, seed=4242 + args.task)
    class SaveLoader:
        def data_config(self):
            return data_config
    saved_loader = SaveLoader()
    spec = importlib.util.spec_from_file_location('openpi_native_train', OPENPI / 'scripts/train.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
        keep_period=1 if retention_settings else None, overwrite=False, resume=config.checkpoint_dir.exists())
    train_rng, init_rng = jax.random.split(jax.random.key(42))
    resume_step = latest_complete_step(manager.all_steps(), config.checkpoint_dir, begin_step, end_step)
    source_checkpoint = None
    if resume_step is not None:
        source = 'resume_same_task'
        state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=True)
        state = checkpoints.restore_state(manager, state, saved_loader, step=resume_step)
        assert int(state.step) == resume_step
        source_checkpoint = relative_path(config.checkpoint_dir / str(resume_step), stream)
    elif args.task:
        source = 'carry_from_previous_task'
        prev_trained = json.loads((previous / 'trained.json').read_text())
        assert prev_trained['end_step'] == begin_step, prev_trained
        source_checkpoint = relative_path(resolve_checkpoint(stream, prev_trained['checkpoint'], args.task - 1), stream)
        prev_manager, ok = checkpoints.initialize_checkpoint_dir(resolve_checkpoint(stream, prev_trained['checkpoint'], args.task - 1).parent,
            keep_period=1 if retention_settings else None, overwrite=False, resume=True)
        assert ok, 'previous task checkpoint missing'
        assert begin_step in prev_manager.all_steps(), (begin_step, prev_manager.all_steps())
        state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=True)
        state = checkpoints.restore_state(prev_manager, state, saved_loader, step=begin_step)
        prev_manager.close()
        # Prove the carry-over: global step continues and Adam moments are the previous task's, not fresh zeros.
        assert int(state.step) == begin_step, (int(state.step), begin_step)
        moment_sq = sum(float(jnp.sum(jnp.square(x.astype(jnp.float32)))) for x in jax.tree.leaves(state.opt_state)
                        if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating))
        assert moment_sq > 0, 'optimizer state was reset instead of carried over'
    else:
        source = 'pi05_base'
        state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    moment_norm = float(sum(float(jnp.sum(jnp.square(x.astype(jnp.float32)))) for x in jax.tree.leaves(state.opt_state)
                             if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating)) ** 0.5)
    start = int(state.step)
    assert begin_step <= start <= end_step, (start, begin_step, end_step)
    if start > 0:
        assert moment_norm > 0, 'resumed/carried Adam moments unexpectedly zero'
    resume_record = dict(job_id=os.environ['SLURM_JOB_ID'], code_commit=os.environ.get('V1_CODE_COMMIT'),
                         timestamp=time.time(), source=source, checkpoint=source_checkpoint,
                         start_step=start, task_step=start - begin_step,
                         lr_at_start=float(config.lr_schedule.create()(start)), optimizer_moment_norm=moment_norm)
    with (stage / 'resume_history.jsonl').open('a') as log:
        log.write(json.dumps(resume_record) + '\n')
    print('RESUME_STATE', json.dumps(resume_record), flush=True)
    leaves = flax.traverse_util.flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict(), sep='/')
    keys = list(leaves)
    assert any('lora' in k for k in keys), 'LoRA must train'
    assert any('img' in k for k in keys), 'FMN default: SigLIP vision tower must train'
    assert not any('llm' in k and 'lora' not in k for k in keys), 'Gemma/action-expert base weights must stay frozen'
    frozen = flax.traverse_util.flatten_dict(state.params.filter(config.freeze_filter).to_pure_dict(), sep='/')
    run_config = dict(method=args.method, suite=args.suite, task=args.task, seed=42,
        steps_per_task=spt, begin_step=begin_step, end_step=end_step, start_step=start, init_source=source, optimizer_moment_norm_at_start=moment_norm,
        current_batch=CURRENT_BATCH, replay_batch=effective_batch - CURRENT_BATCH,
        replay_samples_per_old_task=REPLAY_PER_TASK, devices=[str(d) for d in jax.devices()],
        fsdp_devices=config.fsdp_devices, model=str(config.model), optimizer=str(config.optimizer),
        lr_schedule=str(config.lr_schedule), lr_at_begin=float(config.lr_schedule.create()(begin_step)),
        lr_at_end=float(config.lr_schedule.create()(end_step)),
        lr_at_start=float(config.lr_schedule.create()(start)), resume_checkpoint=source_checkpoint,
        job_id=os.environ['SLURM_JOB_ID'], code_commit=os.environ.get('V1_CODE_COMMIT'),
        normalization='suite-level mean/std, computed once over all 10 tasks (FMN-style)',
        data='physical-intelligence/libero 256px no-noops, no extra flip',
        trainable_parameters=sum(int(np.prod(v.shape)) for v in leaves.values()),
        frozen_parameters=sum(int(np.prod(v.shape)) for v in frozen.values()),
        trainable_groups=sorted({k.split('/')[0] + '/' + k.split('/')[1] for k in keys}),
        preflight=args.preflight, save_trainable_snapshots=args.save_trainable_snapshots)
    write_json(stage / 'run_config.json', run_config)
    print('CONFIG_READY', json.dumps(run_config), flush=True)
    step_fn = jax.jit(functools.partial(native.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    current_iter = iter(loader)
    replay_iter = iter(replay) if replay else None
    # Deterministic samplers recover their position when a task is resumed mid-way.
    for _ in range(start - begin_step):
        next(current_iter)
        if replay_iter is not None:
            next(replay_iter)
    began = interval = time.monotonic()
    losses, grads = [], []
    for step in range(start, end_step):
        batch = next(current_iter)
        if replay_iter is not None:
            batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), batch, next(replay_iter))
        data = (Observation.from_dict(batch), batch['actions'])
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, data)
        info = jax.device_get(info)
        assert all(np.isfinite(float(v)) for v in info.values()), info
        losses.append(float(info['loss']))
        grads.append(float(info['grad_norm']))
        completed = step + 1
        if completed % 50 == 0 or completed == end_step or completed == start + 1:
            now = time.monotonic()
            record = dict(step=completed, task_step=completed - begin_step, loss=float(np.mean(losses)),
                grad_norm=float(np.mean(grads)), lr=float(config.lr_schedule.create()(completed)),
                seconds_per_step=(now - interval) / len(losses), elapsed_seconds=now - began,
                gpu_mib_in_use=int(jax.devices()[0].memory_stats().get('bytes_in_use', 0) // 2**20),
                gpu_mib_peak=int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0) // 2**20),
                timestamp=time.time(), method=args.method, suite=args.suite, task=args.task, job_id=os.environ['SLURM_JOB_ID'])
            with (stage / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            write_json(stage / 'progress.json', record)
            print('TRAIN_PROGRESS', json.dumps(record), flush=True)
            interval = time.monotonic()
            losses, grads = [], []
        if completed % 1000 == 0 or completed == end_step:
            checkpoints.save_state(manager, state, saved_loader, completed)
            manager.wait_until_finished()
            sample_disk(stream, 'checkpoint_saved')
    assert int(state.step) == end_step
    manager.wait_until_finished()
    checkpoint = config.checkpoint_dir / str(end_step)
    assert (checkpoint / 'params').is_dir() and (checkpoint / 'train_state').is_dir()
    if args.save_trainable_snapshots:
        from paths import BASE
        save_snapshot(stage / 'trainable_snapshot', state.params, config.trainable_filter,
            task=args.task, step=end_step, metadata=dict(method=args.method, suite=args.suite,
                code_commit=os.environ.get('V1_CODE_COMMIT'), base_name=BASE.name,
                base_path_provenance=str(BASE), model=str(config.model),
                trainable_filter=str(config.trainable_filter),
                normalization='../../metadata/norm/' + args.suite))
        sample_disk(stream, 'trainable_snapshot_saved')
    if args.method == 'er':
        # Fixed random subset reused for the rest of the stream (Continual-VLAs create_deterministic_buffer).
        indices = np.random.RandomState(42 + args.task).choice(len(current), min(REPLAY_PER_TASK, len(current)), replace=False)
        write_json(stage / 'buffer.json', dict(indices=sorted(indices.tolist()), frames=len(current), task=args.task))
    write_json(stage / 'trained.json', dict(checkpoint=relative_path(checkpoint, stream), checkpoint_path_base='stream', checkpoint_complete=True, begin_step=begin_step, end_step=end_step,
        seconds=time.monotonic() - began, finished_at=time.time(), preflight=args.preflight))
    manager.close()
    print('STAGE_TRAINED', checkpoint, flush=True)

if __name__ == '__main__':
    main()
