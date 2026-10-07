"""RoboCasa's released JAX pi05 Human300 inference recipe.

Reference: robocasa-benchmark/openpi examples/robocasa/main.py.
The policy uses EEF-first state/action ordering; the stored LeRobot modality
layout is not the inference interface's ordering.
"""
import numpy as np

from XPolicyLab.benchmarks.geometry import axis_angle_to_quat_wxyz


def make_config(norm_stats):
    import dataclasses
    from openpi.training.config import get_config, DataConfig
    config = get_config('pi05_pretrain_human300')
    # Inference must use the checkpoint's saved statistics, not recalculate
    # them from training datasets at machine-specific registry paths.
    base = config.data.base_config or DataConfig()
    data = dataclasses.replace(config.data, data_dirs=(),
                               base_config=dataclasses.replace(base, norm_stats=norm_stats))
    return dataclasses.replace(config, data=data)


def encode_observation(obs, resize_image=None):
    if obs.get('benchmark') != 'robocasa365':
        raise ValueError('Expected a RoboCasa365 observation')
    if resize_image is None:
        from openpi_client import image_tools
        resize_image = lambda image: image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    state = obs['state']
    eef = np.asarray(state['ee_pose'], dtype=np.float32).reshape(7)
    base = np.asarray(state['base_pose'], dtype=np.float32).reshape(7)
    gripper = np.asarray(state['gripper_qpos'], dtype=np.float32).reshape(2)
    packed = np.concatenate((eef[:3], eef[[4, 5, 6, 3]], base[:3], base[[4, 5, 6, 3]], gripper))
    if not np.isfinite(packed).all():
        raise ValueError('Non-finite RoboCasa state')
    def image(name):
        # The official gym wrapper already returns the training orientation.
        return resize_image(np.ascontiguousarray(obs['vision'][name]['color']))
    return {'observation/state': packed,
            'observation/image': image('agentview_left'),
            'observation/wrist_image': image('eye_in_hand'),
            'observation/right_image': image('agentview_right'),
            'prompt': str(obs['instruction'])}


def decode_action(actions):
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 12 or not len(value) or not np.isfinite(value).all():
        raise ValueError('OpenPI RoboCasa requires finite nonempty (T,12) actions')
    return [{'ee_pose': np.concatenate((row[:3], axis_angle_to_quat_wxyz(row[3:6]))),
             'ee_joint_state': np.clip(row[6:7], 0, 1).copy(),
             'base_motion': row[7:11].copy(),
             'control_mode': np.clip(row[11:12], 0, 1).copy(),
             'ee_pose_mode': 'delta'} for row in value]
