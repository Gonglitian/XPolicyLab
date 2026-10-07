"""Official RoboCasa PandaOmron mapping for GR00T N1.5 inference."""
from __future__ import annotations

import numpy as np

from XPolicyLab.benchmarks.geometry import axis_angle_to_quat_wxyz


def encode_observation(obs, fallback_prompt):
    # Gym has already vertically corrected the RGB camera images. Do not rotate
    # these like the LIBERO recipe, or change their channel order.
    cameras = {
        'agentview_left': 'video.robot0_agentview_left',
        'agentview_right': 'video.robot0_agentview_right',
        'eye_in_hand': 'video.robot0_eye_in_hand',
    }
    result = {
        target: np.ascontiguousarray(obs['vision'][source]['color'])[None]
        for source, target in cameras.items()
    }
    state = obs['state']
    ee = np.asarray(state['ee_pose'], dtype=np.float32).reshape(7).copy()
    base = np.asarray(state['base_pose'], dtype=np.float32).reshape(7).copy()
    # Shared poses are wxyz; official RoboCasa GR00T states are xyzw.
    result.update({
        'state.end_effector_position_relative': ee[None, :3],
        'state.end_effector_rotation_relative': ee[None, [4, 5, 6, 3]],
        'state.gripper_qpos': np.asarray(state['gripper_qpos'], dtype=np.float32).reshape(1, 2).copy(),
        'state.base_position': base[None, :3],
        'state.base_rotation': base[None, [4, 5, 6, 3]],
        'annotation.human.task_description': [str(obs.get('instruction') or fallback_prompt)],
    })
    return result


def decode_action(action):
    dimensions = {
        'action.end_effector_position': 3,
        'action.end_effector_rotation': 3,
        'action.gripper_close': 1,
        'action.base_motion': 4,
        'action.control_mode': 1,
    }
    values = {}
    for key, dim in dimensions.items():
        value = np.asarray(action[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != dim or not np.isfinite(value).all():
            raise ValueError(f'{key}: expected finite (horizon, {dim}), got {value.shape}')
        values[key] = value
    horizon = len(values['action.base_motion'])
    if not horizon or any(len(value) != horizon for value in values.values()):
        raise ValueError('GR00T returned empty or mismatched action horizons')
    result = []
    for step in range(horizon):
        result.append({
            'ee_pose': np.concatenate((
                values['action.end_effector_position'][step],
                axis_angle_to_quat_wxyz(values['action.end_effector_rotation'][step]),
            )),
            'ee_joint_state': values['action.gripper_close'][step],
            'base_motion': values['action.base_motion'][step],
            'control_mode': values['action.control_mode'][step],
            'ee_pose_mode': 'delta',
        })
    return result
