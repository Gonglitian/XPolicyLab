"""Train one stage with native OpenPI init/train/save; optimizer resets between tasks."""
import argparse
import functools
import importlib.util
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from common import ASSETS, OPENPI, RUN, H5Task, make_config, update_norm, write_json
from gpu_guard import Guard

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['sf', 'er'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--root', type=Path, default=RUN)
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    import jax
    import jax.numpy as jnp
    import flax.traverse_util
    from torch.utils.data import Subset, ConcatDataset
    from openpi.training import data_loader as dl, sharding, checkpoints
    from openpi.shared import normalize
    from openpi.models.model import Observation
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    assert jax.device_count() == 4, jax.devices()
    manifest = json.loads((RUN / 'manifest.json').read_text())[args.suite]
    stream = args.root / args.method / args.suite
    stage = stream / f'task{args.task:02d}'
    stage.mkdir(parents=True, exist_ok=True)
    previous = stream / f'task{args.task-1:02d}'
    initial = ASSETS / 'checkpoints/cl_base/pi05_base'
    if args.task:
        initial = Path(json.loads((previous / 'trained.json').read_text())['checkpoint'])
    current = H5Task(manifest[args.task])
    norms = update_norm(current, previous / 'moments.json' if args.task else None, stage / 'moments.json')
    normalize.save(stage / 'norm', norms)
    effective_batch = 16 if args.method == 'er' and args.task else 8
    config = make_config(norms, stage, initial, effective_batch, args.steps)
    data_config = config.data.create(config.assets_dirs, config.model)
    transformed = dl.transform_dataset(current, data_config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = dl.TorchDataLoader(transformed, local_batch_size=8, sharding=data_sharding,
                               shuffle=True, num_workers=4, seed=42)
    replay = None
    if effective_batch == 16:
        subsets = []
        for i in range(args.task):
            memory = json.loads((stream / f'task{i:02d}' / 'buffer.json').read_text())
            old = H5Task(manifest[i])
            assert memory['frames'] == len(old)
            subsets.append(Subset(dl.transform_dataset(old, data_config), memory['indices']))
        replay = dl.TorchDataLoader(ConcatDataset(subsets), local_batch_size=8,
                                   sharding=data_sharding, shuffle=True, num_workers=4, seed=42)
    class SaveLoader:
        def data_config(self):
            return data_config
    saved_loader = SaveLoader()
    source = OPENPI / 'scripts/train.py'
    spec = importlib.util.spec_from_file_location('openpi_native_train', source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
        keep_period=None, overwrite=False, resume=config.checkpoint_dir.exists())
    train_rng, init_rng = jax.random.split(jax.random.key(42))
    state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, saved_loader)
    jax.block_until_ready(state)
    leaves = flax.traverse_util.flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict(), sep='/')
    assert leaves and all('lora' in k for k in leaves), list(leaves)
    assert any('_1' in k for k in leaves) and any('_1' not in k for k in leaves), 'Both Gemma and action-expert LoRA must train'
    params = {k: list(v.shape) for k, v in leaves.items()}
    run_config = dict(method=args.method, suite=args.suite, task=args.task, seed=42,
        steps=args.steps, current_batch=8, replay_batch=effective_batch-8,
        initial=str(initial), devices=[str(d) for d in jax.devices()], fsdp_devices=config.fsdp_devices,
        model=str(config.model), optimizer=str(config.optimizer), lr_schedule=str(config.lr_schedule),
        normalization='Cumulative unique-frame mean/std updated on each task arrival; frozen during stage/eval',
        action_labels='original delta; last action repeated at episode boundary; horizon=10',
        trainable_parameters=sum(int(np.prod(v.shape)) for v in leaves.values()),
        trainable_shapes=params, preflight=args.preflight)
    write_json(stage / 'run_config.json', run_config)
    print('CONFIG_READY', json.dumps({k: v for k, v in run_config.items() if k != 'trainable_shapes'}), flush=True)
    step_fn = jax.jit(functools.partial(native.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    current_iter = iter(loader)
    replay_iter = iter(replay) if replay else None
    start = int(state.step)
    # Deterministic samplers recover their position when a stage resumes.
    for _ in range(start):
        next(current_iter)
        if replay_iter is not None:
            next(replay_iter)
    began = interval = time.monotonic()
    losses, grads = [], []
    gpu_guard = Guard(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    for step in range(start, args.steps):
        batch = next(current_iter)
        if replay_iter is not None:
            old_batch = next(replay_iter)
            batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), batch, old_batch)
        data = (Observation.from_dict(batch), batch['actions'])
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, data)
        # Synchronize actual compute for valid time measurements and fail on nonfinite updates.
        info = jax.device_get(info)
        assert all(np.isfinite(float(v)) for v in info.values()), info
        losses.append(float(info['loss']))
        grads.append(float(info['grad_norm']))
        completed = step + 1
        if completed % 50 == 0 or completed == args.steps or completed == start + 1:
            now = time.monotonic()
            record = dict(step=completed, loss=float(np.mean(losses)), grad_norm=float(np.mean(grads)),
                seconds_per_step=(now-interval)/len(losses), elapsed_seconds=now-began,
                timestamp=time.time(), method=args.method, suite=args.suite, task=args.task)
            record.update(timing_basis='four_idle_gpus_5s_monitor',
                          idle_gpu_timing_valid=gpu_guard.conflict is None and completed > start + 50)
            with (stage / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            write_json(stage / 'progress.json', record)
            print('TRAIN_PROGRESS', json.dumps(record), flush=True)
            interval = time.monotonic()
            losses, grads = [], []
        if completed % 1000 == 0 or completed == args.steps or gpu_guard.conflict:
            checkpoints.save_state(manager, state, saved_loader, completed)
            manager.wait_until_finished()
        if gpu_guard.conflict:
            write_json(stage / 'gpu_pause.json', dict(step=completed, reason=gpu_guard.conflict))
            gpu_guard.close()
            manager.close()
            raise SystemExit(75)
    assert int(state.step) == args.steps
    manager.wait_until_finished()
    checkpoint = config.checkpoint_dir / str(args.steps)
    assert (checkpoint / 'params').is_dir()
    if args.method == 'er':
        # Same fixed-subset construction as Continual-VLAs; keep all only if fewer than 1000 exist.
        indices = np.random.RandomState(42 + args.task).choice(len(current), min(1000, len(current)), replace=False)
        write_json(stage / 'buffer.json', dict(indices=indices.tolist(), frames=len(current), task=args.task))
    write_json(stage / 'trained.json', dict(checkpoint=str(checkpoint), updates=args.steps,
        seconds=time.monotonic()-began, finished_at=time.time(), preflight=args.preflight))
    manager.close()
    gpu_guard.close()
    print('STAGE_TRAINED', checkpoint, flush=True)

if __name__ == '__main__':
    main()
