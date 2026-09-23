"""Pure NumPy RoboCasa365 observation/action contract helpers.

This is deliberately simulator-independent so the PandaOmron key mapping and
the action-order conversion can be unit-tested before a RoboCasa365 checkpoint
or full simulator run is approved.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from XPolicyLab.benchmarks.geometry import quat_wxyz_to_axis_angle, xyzw_to_wxyz


IMAGE_KEYS = {
    "agentview_left": "video.robot0_agentview_left",
    "eye_in_hand": "video.robot0_eye_in_hand",
    "agentview_right": "video.robot0_agentview_right",
}

EEF_POSITION_KEY = "state.end_effector_position_relative"
EEF_ROTATION_KEY = "state.end_effector_rotation_relative"
GRIPPER_KEY = "state.gripper_qpos"
BASE_POSITION_CANDIDATES = (
    "state.base_position",
    "state.base_pos",
    "state.base_position_relative",
    "state.robot0_base_pos",
    "state.base_xy",
)
BASE_ROTATION_CANDIDATES = (
    "state.base_rotation",
    "state.base_quat",
    "state.base_orientation",
    "state.base_rotation_relative",
    "state.robot0_base_quat",
)


def _first(obs: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    return next((obs[key] for key in keys if key in obs), None)


def build_state(obs: dict[str, Any]) -> np.ndarray:
    """Build the 16-D PandaOmron state used by existing EvoMoE exports."""
    base_position = _first(obs, BASE_POSITION_CANDIDATES)
    if base_position is None:
        raise KeyError("Missing RoboCasa365 base position; cannot invent mobile-base state")
    base_position = np.asarray(base_position, dtype=np.float32).reshape(-1)
    if base_position.size == 2:
        base_position = np.pad(base_position, (0, 1))

    base_rotation = _first(obs, BASE_ROTATION_CANDIDATES)
    if base_rotation is None:
        raise KeyError("Missing RoboCasa365 base quaternion (xyzw)")

    state = np.concatenate(
        (
            base_position.reshape(3),
            np.asarray(base_rotation, dtype=np.float32).reshape(4),
            np.asarray(obs[EEF_POSITION_KEY], dtype=np.float32).reshape(3),
            np.asarray(obs[EEF_ROTATION_KEY], dtype=np.float32).reshape(4),
            np.asarray(obs[GRIPPER_KEY], dtype=np.float32).reshape(2),
        )
    ).astype(np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(f"Expected RoboCasa365 state shape (16,), got {state.shape}")
    return state


def make_xpl_observation(obs: dict[str, Any], instruction: str) -> dict[str, Any]:
    vision = {}
    for target, source in IMAGE_KEYS.items():
        image = np.asarray(obs[source])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{source}: expected RGB HWC image, got {image.shape}")
        vision[target] = {"color": np.ascontiguousarray(image)}
    state = build_state(obs)
    return {
        "vision": vision,
        "state": {
            "base_pose": np.concatenate((state[:3], xyzw_to_wxyz(state[3:7]))),
            "ee_pose": np.concatenate((state[7:10], xyzw_to_wxyz(state[10:14]))),
            "gripper_qpos": state[14:16].copy(),
        },
        "instruction": str(instruction),
        "benchmark": "robocasa365",
        "data_format_version": "evomoe-benchmark-v1",
    }


def xpl_action_to_gym(action: dict[str, Any]) -> dict[str, np.ndarray]:
    """Mobile extension to XPL: ee_pose + gripper + base_motion + control_mode.

    EEF pose contains a delta rotation in wxyz. Gripper/control mode retain
    the official dataset's [0, 1] semantics; the gym wrapper binarizes them.
    """
    if action.get('ee_pose_mode', 'delta') != 'delta':
        raise ValueError('RoboCasa365 gym expects delta EEF commands')
    pose = np.asarray(action['ee_pose'], dtype=np.float32).reshape(7)
    result = {
        'action.end_effector_position': pose[:3],
        'action.end_effector_rotation': quat_wxyz_to_axis_angle(pose[3:]),
        'action.gripper_close': np.asarray(action['ee_joint_state'], dtype=np.float32).reshape(1),
        'action.base_motion': np.asarray(action['base_motion'], dtype=np.float32).reshape(4),
        'action.control_mode': np.asarray(action['control_mode'], dtype=np.float32).reshape(1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Non-finite RoboCasa365 action')
    for key in ('action.gripper_close', 'action.control_mode'):
        if not 0 <= result[key][0] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    return result


def modality_to_raw_action(action: Any) -> np.ndarray:
    """Map 12-D LeRobot modality order to raw robosuite/PandaOmron order.

    Model order: ``base(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)``.
    Raw order: ``eef_pos(3), eef_rot(3), gripper(1), base(4), control_mode(1)``.
    """
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.size != 12:
        raise ValueError(f"Expected 12-D RoboCasa365 action, got {value.shape}")
    return np.concatenate((value[5:8], value[8:11], value[11:12], value[0:4], value[4:5]))


def humanize_task_name(task_name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", task_name).lower().strip()
    return text.replace("pn p", "pick and place").replace("p n p", "pick and place")
