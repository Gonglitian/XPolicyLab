"""OpenPI's official LIBERO inference recipe at the policy boundary."""
import numpy as np
from XPolicyLab.benchmarks.geometry import axis_angle_to_quat_wxyz, quat_wxyz_to_axis_angle


def make_config():
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import TrainConfig, LeRobotLiberoDataConfig, DataConfig
    # Official pi05_libero config; independent of this fork's ALOHA registry.
    return TrainConfig(
        name='pi05_libero',
        model=Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(repo_id='physical-intelligence/libero',
                                    base_config=DataConfig(prompt_from_task=True),
                                    extra_delta_transform=False),
    )


def encode_observation(obs, resize_image=None):
    if resize_image is None:
        from openpi_client import image_tools
        resize_image = lambda image: image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    pose = np.asarray(obs['state']['ee_pose'], dtype=np.float32).reshape(7)
    gripper = np.asarray(obs['state']['gripper_qpos'], dtype=np.float32).reshape(2)
    def image(name):
        raw = np.asarray(obs['vision'][name]['color'])
        return resize_image(np.ascontiguousarray(raw[::-1, ::-1, :]))
    return {'observation/state': np.concatenate((pose[:3], quat_wxyz_to_axis_angle(pose[3:]), gripper)),
            'observation/image': image('agentview'), 'observation/wrist_image': image('wrist'),
            'prompt': str(obs['instruction'])}


def decode_action(actions):
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 7 or not len(value) or not np.isfinite(value).all():
        raise ValueError('OpenPI LIBERO requires finite nonempty (T,7) actions')
    return [{'ee_pose': np.concatenate((row[:3], axis_angle_to_quat_wxyz(row[3:6]))),
             'ee_joint_state': np.array([(np.clip(row[6], -1, 1) + 1) / 2], np.float32),
             'ee_pose_mode': 'delta'} for row in value]
