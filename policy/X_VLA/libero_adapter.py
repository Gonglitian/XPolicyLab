"""X-VLA LIBERO recipe: two cameras, column-stacked rot6d, absolute EEF."""
import numpy as np
from scipy.spatial.transform import Rotation
from XPolicyLab.benchmarks.geometry import xyzw_to_wxyz


def encode_observation(obs, previous_proprio=None):
    state = obs['state']
    pose = np.asarray(state.get('controller_ee_pose', state['ee_pose']), np.float32).reshape(7)
    matrix = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    # The official LIBERO client stacks columns, unlike the legacy RoboDojo adapter.
    proprio = np.concatenate((pose[:3], matrix[:, 0], matrix[:, 1], [0], np.zeros(10))).astype(np.float32)
    if previous_proprio is not None:
        proprio = np.asarray(previous_proprio, np.float32).copy()
    agent = np.asarray(obs['vision']['agentview']['color'])
    wrist = np.asarray(obs['vision']['wrist']['color'])
    return {'images': [np.ascontiguousarray(agent[::-1, ::-1, :]), np.ascontiguousarray(wrist)],
            'proprio': proprio, 'prompt': str(obs['instruction']), 'output_format': 'libero'}


def decode_action(actions):
    value = np.asarray(actions, np.float32)
    if value.ndim != 2 or value.shape[1] not in (10, 20) or not len(value) or not np.isfinite(value).all():
        raise ValueError('X-VLA LIBERO requires finite nonempty (T,10) or (T,20) actions')
    result = []
    for row in value:
        first, second = row[3:6].astype(float), row[6:9].astype(float)
        if np.linalg.norm(first) < 1e-8:
            raise ValueError('Degenerate X-VLA rotation')
        first /= np.linalg.norm(first)
        second -= first * np.dot(first, second)
        if np.linalg.norm(second) < 1e-8:
            raise ValueError('Degenerate X-VLA rotation')
        second /= np.linalg.norm(second)
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        quat = xyzw_to_wxyz(Rotation.from_matrix(matrix).as_quat())
        result.append({'ee_pose': np.concatenate((row[:3], quat)),
                       'ee_joint_state': np.array([float(row[9] > 0.5)], np.float32),
                       'ee_pose_mode': 'absolute'})
    return result
