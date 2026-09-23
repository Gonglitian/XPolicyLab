"""Shared, explicit training and inference recipe for the formal baselines."""
import dataclasses
import json
from pathlib import Path
import numpy as np

from settings import ROOT, ASSETS, RUN, OPENPI, SUITES, ALL_SUITES
HORIZONS = dict(zip(ALL_SUITES, [220, 280, 300, 520]))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def model_config():
    from openpi.models.pi0_config import Pi0Config
    return Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True,
                     paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                     dtype='bfloat16')

def make_config(norm_stats, stage_dir, initial, batch, steps=10000):
    import flax.nnx as nnx
    from openpi.shared import nnx_utils
    from openpi import transforms
    from openpi.policies import libero_policy
    from openpi.training import config as cfg, optimizer, weight_loaders

    @dataclasses.dataclass(frozen=True)
    class DataFactory(cfg.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return cfg.DataConfig(repo_id='formal_libero', asset_id='formal_libero',
                norm_stats=norm_stats, use_quantile_norm=False,
                data_transforms=transforms.Group(
                    inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                    outputs=[libero_policy.LiberoOutputs()]),
                model_transforms=cfg.ModelTransformFactory()(model_config),
                action_sequence_keys=('actions',))
    return cfg.TrainConfig(name='pi05_libero_cl', exp_name='stage',
        model=model_config(), data=DataFactory(),
        freeze_filter=nnx.Not(nnx_utils.PathRegex('.*lora.*')),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(initial) / 'params')),
        optimizer=optimizer.AdamW(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5,
                                                decay_steps=10000, decay_lr=2.5e-6),
        ema_decay=None, batch_size=batch, num_train_steps=steps, seed=42,
        num_workers=4, fsdp_devices=4, wandb_enabled=False,
        checkpoint_base_dir=str(Path(stage_dir) / 'checkpoints'),
        assets_base_dir=str(Path(stage_dir) / 'assets'), log_interval=50,
        save_interval=1000, keep_period=None)

class H5Task:
    """Raw RGB HDF5 arrays: no image-byte decoding or color-channel conversion."""
    def __init__(self, task):
        import h5py
        self.task = task
        self.ends = np.cumsum(task['lengths'])
        self.handle = None
        self.states, self.actions = [], []
        with h5py.File(task['source'], 'r') as f:
            for name in task['demos']:
                demo = f['data'][name]
                obs = demo['obs']
                state = np.concatenate([obs['ee_pos'][:], obs['ee_ori'][:], obs['gripper_states'][:]], axis=1).astype(np.float32)
                action = demo['actions'][:].astype(np.float32)
                assert state.shape[1] == 8 and np.isfinite(state).all() and np.isfinite(action).all()
                self.states.append(state)
                self.actions.append(action)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        import h5py
        if self.handle is None:
            self.handle = h5py.File(self.task['source'], 'r')
        episode = int(np.searchsorted(self.ends, index, side='right'))
        frame = int(index - (self.ends[episode - 1] if episode else 0))
        obs = self.handle['data'][self.task['demos'][episode]]['obs']
        # Repeat the terminal action at episode boundaries, like LeRobot's clamped timestamps.
        indices = np.minimum(np.arange(frame, frame + 10), len(self.actions[episode]) - 1)
        return {'observation/state': self.states[episode][frame].copy(),
                'observation/image': np.ascontiguousarray(obs['agentview_rgb'][frame][::-1, ::-1]),
                'observation/wrist_image': np.ascontiguousarray(obs['eye_in_hand_rgb'][frame][::-1, ::-1]),
                'actions': self.actions[episode][indices].copy(), 'prompt': self.task['instruction']}

    def __getstate__(self):
        return {**self.__dict__, 'handle': None}

def update_norm(current, previous_moments, target):
    """Update moments once per newly arriving task; replay never counts as fresh data."""
    from openpi.shared.normalize import NormStats
    old = json.loads(Path(previous_moments).read_text()) if previous_moments else {}
    moments, norms = {}, {}
    for key, seq in [('state', current.states), ('actions', current.actions)]:
        values = np.concatenate(seq).astype(np.float64)
        n = len(values)
        total, square = values.sum(0), np.square(values).sum(0)
        if key in old:
            n += old[key]['n']
            total += np.asarray(old[key]['sum'])
            square += np.asarray(old[key]['sum_square'])
        mean = total / n
        std = np.sqrt(np.maximum(square / n - mean ** 2, 0))
        norms[key] = NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))
        moments[key] = dict(n=n, sum=total.tolist(), sum_square=square.tolist())
    write_json(target, moments)
    return norms
