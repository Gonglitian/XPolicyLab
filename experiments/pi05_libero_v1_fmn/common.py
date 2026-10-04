"""v1 formal baseline recipe, aligned with FMN (arXiv 2603.03818, github.com/Continual-VLAs/continual-openpi).

Differences from v0 (pi05_libero_20260923), each checked against the FMN source:
  * trainable set = openpi default LoRA freeze filter (FMN default): only non-LoRA Gemma/action-expert weights
    are frozen; SigLIP vision tower, projections and LoRA adapters train.
  * one learning-rate schedule and one optimizer state for the whole 10-task stream (global step carried
    across tasks): warmup 1k, peak 2.5e-5, cosine to 2.5e-6 at global step 30k, then flat.
  * data = physical-intelligence/libero (256 px, no-op frames removed, images already rotated 180 deg to the
    openpi evaluation convention, so no extra flip here).
  * normalization = mean/std over the whole suite, computed once (FMN computes suite-level assets).
Unchanged from v0/FMN: global batch 8, 10k steps per task, ER = 1000 fixed samples per old task and an extra
replay batch of the same size (8 + 8), no EMA, evaluation on the last checkpoint of each task.
"""
import dataclasses
import io
import json
import os
from pathlib import Path
import numpy as np

# Paths default to tasl-labserver; another server overrides them via V1_* variables (see bcc_env.sh).
ROOT = Path(os.environ.get('V1_ROOT', '/data2/vla-reasoning/proj'))
ASSETS = Path(os.environ.get('V1_ASSETS', str(ROOT / 'XPolicyLab-assets')))
RUN = Path(os.environ.get('V1_RUN', str(ASSETS / 'baselines/pi05_libero_v1_fmn')))
DATA = Path(os.environ.get('V1_DATA', str(ASSETS / 'datasets/libero_pi_lerobot')))
BASE = Path(os.environ.get('V1_BASE', str(ASSETS / 'checkpoints/cl_base/pi05_base')))
OPENPI = Path(os.environ.get('V1_OPENPI', str(ROOT / 'XPolicyLab-upstreams/openpi-robocasa')))
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
HORIZONS = dict(zip(SUITES, [220, 280, 300, 520]))
STEPS_PER_TASK = 10000
CURRENT_BATCH = 8
REPLAY_PER_TASK = 1000
ACTION_HORIZON = 10

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)

def model_config():
    from openpi.models.pi0_config import Pi0Config
    # Same model settings as the official pi05_libero config (continuous state input, horizon 10),
    # with LoRA variants as in FMN's pi0_libero_low_mem_finetune family.
    return Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=False,
                     paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                     dtype='bfloat16')

def make_config(norm_stats, stage_dir, batch, total_steps, fsdp_devices=1):
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
    model = model_config()
    return cfg.TrainConfig(name='pi05_libero_cl', exp_name='stage',
        model=model, data=DataFactory(),
        freeze_filter=model.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE / 'params')),
        optimizer=optimizer.AdamW(b1=.9, b2=.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5,
                                                decay_steps=30000, decay_lr=2.5e-6),
        ema_decay=None, batch_size=batch, num_train_steps=total_steps, seed=42,
        num_workers=4, fsdp_devices=fsdp_devices, wandb_enabled=False,
        checkpoint_base_dir=str(Path(stage_dir) / 'checkpoints'),
        assets_base_dir=str(Path(stage_dir) / 'assets'), log_interval=50,
        save_interval=1000, keep_period=None)

def _fixed(column, width):
    return column.combine_chunks().flatten().to_numpy().reshape(-1, width).astype(np.float32)

class LeRobotTask:
    """All episodes of one LIBERO task from physical-intelligence/libero, kept as PNG bytes in memory."""
    def __init__(self, task):
        import pyarrow.parquet as pq
        self.instruction = task['instruction']
        self.states, self.actions, self.images, self.wrists = [], [], [], []
        for path in task['episode_files']:
            t = pq.read_table(path, columns=['image', 'wrist_image', 'state', 'actions', 'frame_index', 'task_index'])
            frames = t.column('frame_index').to_numpy()
            assert (frames == np.arange(len(frames))).all(), path
            assert set(t.column('task_index').to_numpy().tolist()) == {task['task_index']}, path
            state, action = _fixed(t.column('state'), 8), _fixed(t.column('actions'), 7)
            assert np.isfinite(state).all() and np.isfinite(action).all(), path
            self.states.append(state)
            self.actions.append(action)
            self.images.append([d['bytes'] for d in t.column('image').to_pylist()])
            self.wrists.append([d['bytes'] for d in t.column('wrist_image').to_pylist()])
        self.ends = np.cumsum([len(s) for s in self.states])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        from PIL import Image
        episode = int(np.searchsorted(self.ends, index, side='right'))
        frame = int(index - (self.ends[episode - 1] if episode else 0))
        # Repeat the terminal action at episode boundaries, like LeRobot's clamped timestamps.
        indices = np.minimum(np.arange(frame, frame + ACTION_HORIZON), len(self.actions[episode]) - 1)
        decode = lambda b: np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))
        return {'observation/state': self.states[episode][frame].copy(),
                'observation/image': decode(self.images[episode][frame]),
                'observation/wrist_image': decode(self.wrists[episode][frame]),
                'actions': self.actions[episode][indices].copy(), 'prompt': self.instruction}

def load_norm(suite):
    from openpi.shared import normalize
    return normalize.load(RUN / 'norm' / suite)
