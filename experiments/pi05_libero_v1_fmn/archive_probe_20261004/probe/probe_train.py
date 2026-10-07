"""Plasticity probe: train ONE LIBERO task from pi05_base (cold start, fresh optimiser, steps 0..N of the v1 schedule),
with the SigLIP image encoder either frozen or trainable. Everything else is the v1 baseline recipe (common.make_config)."""
import argparse
import dataclasses
import functools
import importlib.util
import json
import logging
import os
import time
from pathlib import Path
import numpy as np
from common import OPENPI, RUN, CURRENT_BATCH, LeRobotTask, make_config, load_norm, write_json
from gpu_guard import Guard

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['frozen_siglip', 'trainable_siglip'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--stage', type=Path, required=True)
    args = p.parse_args()
    import jax
    import flax.nnx as nnx
    import flax.traverse_util
    from openpi.shared import nnx_utils
    from openpi.training import data_loader as dl, sharding, checkpoints
    from openpi.models.model import Observation
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    assert jax.device_count() == 1, jax.devices()
    manifest = json.loads((RUN / 'manifest.json').read_text())[args.suite]
    stage = args.stage
    stage.mkdir(parents=True, exist_ok=True)
    norms = load_norm(args.suite)
    current = LeRobotTask(manifest[args.task])
    config = make_config(norms, stage, CURRENT_BATCH, args.steps)
    if args.arm == 'frozen_siglip':
        # Freeze the image tower on top of the default filter (which already freezes the non-LoRA Gemma weights).
        config = dataclasses.replace(config, freeze_filter=nnx.Any(config.freeze_filter, nnx_utils.PathRegex('.*img.*')))
    data_config = config.data.create(config.assets_dirs, config.model)
    transformed = dl.transform_dataset(current, data_config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = dl.TorchDataLoader(transformed, local_batch_size=CURRENT_BATCH, sharding=data_sharding,
                                shuffle=True, num_workers=4, seed=42 + args.task)
    class SaveLoader:
        def data_config(self):
            return data_config
    saved_loader = SaveLoader()
    spec = importlib.util.spec_from_file_location('openpi_native_train', OPENPI / 'scripts/train.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manager, resuming = checkpoints.initialize_checkpoint_dir(config.checkpoint_dir,
        keep_period=None, overwrite=False, resume=config.checkpoint_dir.exists())
    train_rng, init_rng = jax.random.split(jax.random.key(42))
    state, state_sharding = native.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(manager, state, saved_loader)
    jax.block_until_ready(state)
    start = int(state.step)
    leaves = flax.traverse_util.flatten_dict(state.params.filter(config.trainable_filter).to_pure_dict(), sep='/')
    keys = list(leaves)
    assert any('lora' in k for k in keys), 'LoRA must train'
    assert not any('llm' in k and 'lora' not in k for k in keys), 'Gemma / action-expert base weights must stay frozen'
    assert any('img' in k for k in keys) == (args.arm == 'trainable_siglip'), 'image tower trainability does not match the arm'
    frozen = flax.traverse_util.flatten_dict(state.params.filter(config.freeze_filter).to_pure_dict(), sep='/')
    count = lambda d, f: sum(int(np.prod(v.shape)) for k, v in d.items() if f(k))
    run_config = dict(arm=args.arm, suite=args.suite, task=args.task, seed=42, steps=args.steps, start_step=start,
        init_source='resume' if resuming else 'pi05_base (cold start)', batch=CURRENT_BATCH,
        lr_schedule=str(config.lr_schedule), lr_at_end=float(config.lr_schedule.create()(args.steps)),
        trainable_parameters=count(leaves, lambda k: True), frozen_parameters=count(frozen, lambda k: True),
        trainable_image_tower=count(leaves, lambda k: 'img' in k), trainable_lora=count(leaves, lambda k: 'lora' in k),
        trainable_other=count(leaves, lambda k: 'img' not in k and 'lora' not in k),
        trainable_groups=sorted({'/'.join(k.split('/')[:2]) for k in keys}),
        devices=[str(d) for d in jax.devices()])
    write_json(stage / 'run_config.json', run_config)
    print('CONFIG_READY', json.dumps(run_config), flush=True)
    step_fn = jax.jit(functools.partial(native.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    it = iter(loader)
    for _ in range(start):
        next(it)
    began = interval = time.monotonic()
    losses = []
    guard = Guard(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    for step in range(start, args.steps):
        batch = next(it)
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, (Observation.from_dict(batch), batch['actions']))
        info = jax.device_get(info)
        assert all(np.isfinite(float(v)) for v in info.values()), info
        losses.append(float(info['loss']))
        done = step + 1
        if done % 50 == 0 or done == args.steps or done == start + 1:
            now = time.monotonic()
            record = dict(step=done, loss=float(np.mean(losses)), lr=float(config.lr_schedule.create()(done)),
                seconds_per_step=(now - interval) / len(losses), elapsed_seconds=now - began,
                gpu_mib_peak=int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0) // 2**20),
                timestamp=time.time(), gpu_contention=guard.conflict is not None)
            with (stage / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print('TRAIN_PROGRESS', json.dumps(record), flush=True)
            interval = time.monotonic()
            losses = []
        if done % 2500 == 0 or done == args.steps:
            checkpoints.save_state(manager, state, saved_loader, done)
            manager.wait_until_finished()
    checkpoint = config.checkpoint_dir / str(args.steps)
    assert (checkpoint / 'params').is_dir()
    write_json(stage / 'trained.json', dict(checkpoint=str(checkpoint), steps=args.steps,
        seconds=time.monotonic() - began, finished_at=time.time()))
    manager.close()
    guard.close()
    print('PROBE_TRAINED', checkpoint, flush=True)

if __name__ == '__main__':
    main()
