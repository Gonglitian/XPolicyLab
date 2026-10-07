from __future__ import annotations

import math
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"

_LIBERO_IMAGE_KEYS = {
    "video.image": ("agentview", "cam_head", "agentview_image"),
    "video.wrist_image": ("wrist", "cam_wrist", "cam_left_wrist", "robot0_eye_in_hand"),
}


def _resolve_gr00t_root(model_cfg: dict[str, Any]) -> Path:
    raw = model_cfg.get("gr00t_root") or os.environ.get("GR00T_N15_ROOT") or "Isaac-GR00T"
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (_POLICY_DIR / path).resolve()
    if not (path / "gr00t" / "model" / "policy.py").is_file():
        raise FileNotFoundError(
            f"NVIDIA Isaac-GR00T n1.5 source not found at {path}. "
            "Run install.sh or set GR00T_N15_ROOT."
        )
    return path


def _checkpoint_dir(model_cfg: dict[str, Any]) -> Path:
    path = resolve_checkpoint_root(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    required = ("config.json", "experiment_cfg/metadata.json", "model.safetensors.index.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete GR00T N1.5 checkpoint at {path}; missing {missing}")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if not shards:
        raise ValueError(f"Checkpoint has no weight_map: {path}")
    for shard in shards:
        target = (path / shard).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise FileNotFoundError(f"Missing or invalid checkpoint shard: {shard}")
    return path


def _suite_name(model_cfg: dict[str, Any]) -> str:
    value = str(model_cfg.get("libero_suite") or model_cfg.get("task_name") or "").lower()
    aliases = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "10": "libero_10",
        "long": "libero_10",
    }
    value = aliases.get(value, value)
    if value not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
        raise ValueError(
            "LIBERO suite must be one of libero_spatial/libero_object/libero_goal/libero_10; "
            f"got {value!r}."
        )
    return value


def _data_config_name(suite: str) -> str:
    suffix = "LiberoDataConfigMeanStd" if suite == "libero_goal" else "LiberoDataConfig"
    return f"examples.Libero.custom_data_config:{suffix}"


def _as_rgb_hwc(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got {array.shape}")
    if array.shape[-1] not in (1, 3) and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _find_image(obs: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
    vision = obs.get("vision", {})
    for name in candidates:
        if name not in vision:
            continue
        value = vision[name]
        if isinstance(value, dict):
            value = value.get("color", value.get("rgb"))
        if value is not None:
            return _as_rgb_hwc(value)
    raise KeyError(f"None of the expected camera names are present: {candidates}")


def _instruction(obs: dict[str, Any], fallback: str) -> str:
    value = obs.get("instruction", obs.get("instructions", fallback))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    text = str(value).strip()
    return text or fallback


def _quat_wxyz_to_axis_angle(quat: Any) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    quat = quat / norm
    if quat[0] < 0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(1.0 - w * w, 0.0))
    if scale < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[1:] / scale * angle).astype(np.float32)


def _axis_angle_to_quat_wxyz(axis_angle: Any) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = value / angle
    half = angle / 2.0
    return np.concatenate(([math.cos(half)], axis * math.sin(half))).astype(np.float32)


def _libero_state(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = obs.get("state", {})
    if "ee_position" in state and "ee_axis_angle" in state:
        position = np.asarray(state["ee_position"], dtype=np.float32).reshape(3)
        rotation = np.asarray(state["ee_axis_angle"], dtype=np.float32).reshape(3)
    else:
        pose = np.asarray(state["ee_pose"], dtype=np.float32).reshape(7)
        position = pose[:3]
        rotation = _quat_wxyz_to_axis_angle(pose[3:])
    gripper = np.asarray(
        state.get("gripper_qpos", state.get("ee_joint_state")), dtype=np.float32
    ).reshape(-1)
    if gripper.size == 1:
        gripper = np.repeat(gripper, 2)
    if gripper.size != 2:
        raise ValueError(f"LIBERO gripper state must have 1 or 2 values, got {gripper.shape}")
    return position, rotation, gripper


def encode_libero_observation(obs: dict[str, Any], fallback_prompt: str) -> dict[str, Any]:
    position, rotation, gripper = _libero_state(obs)
    encoded: dict[str, Any] = {
        # NVIDIA LIBERO recipe rotates both spatial axes; RGB channels stay intact.
        key: np.ascontiguousarray(_find_image(obs, candidates)[::-1, ::-1, :])[None, ...]
        for key, candidates in _LIBERO_IMAGE_KEYS.items()
    }
    encoded.update(
        {
            "state.x": position[0:1][None, :],
            "state.y": position[1:2][None, :],
            "state.z": position[2:3][None, :],
            "state.roll": rotation[0:1][None, :],
            "state.pitch": rotation[1:2][None, :],
            "state.yaw": rotation[2:3][None, :],
            "state.gripper": gripper[None, :],
            "annotation.human.action.task_description": [_instruction(obs, fallback_prompt)],
        }
    )
    return encoded


def decode_libero_action(action: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    values = []
    for key in keys:
        array = np.asarray(action[f"action.{key}"], dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != 1 or not np.isfinite(array).all():
            raise ValueError(f"action.{key}: expected finite (horizon, 1), got {array.shape}")
        values.append(array)
    horizon = values[0].shape[0]
    if not horizon:
        raise ValueError("GR00T returned an empty action chunk")
    if any(value.shape[0] != horizon for value in values):
        raise ValueError("GR00T action components have different horizons")

    result = []
    for step in range(horizon):
        translation = np.array(
            [values[i][step].reshape(-1)[0] for i in range(3)], dtype=np.float32
        )
        axis_angle = np.array(
            [values[i][step].reshape(-1)[0] for i in range(3, 6)], dtype=np.float32
        )
        # NVIDIA predicts openness; shared benchmark actions use closure (0=open).
        raw_gripper = float(values[6][step].reshape(-1)[0])
        gripper = np.array([(np.sign(1.0 - 2.0 * raw_gripper) + 1.0) / 2.0], dtype=np.float32)
        result.append(
            {
                "ee_pose": np.concatenate((translation, _axis_angle_to_quat_wxyz(axis_angle))),
                "ee_joint_state": gripper,
            }
        )
    return result


class Model(ModelTemplate):
    """XPolicyLab wrapper for GR00T N1.5 LIBERO and RoboCasa checkpoints."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.benchmark = str(model_cfg.get('bench_name', 'libero')).lower()
        if self.benchmark not in {'libero', 'robocasa365'}:
            raise ValueError('GR00T N1.5 supports LIBERO and RoboCasa365 checkpoints')
        self.action_type = str(model_cfg.get("action_type", "ee"))
        if self.action_type != "ee":
            raise ValueError("GR00T N1.5 LIBERO checkpoints require action_type=ee")
        expected_robot = 'libero_franka' if self.benchmark == 'libero' else 'robocasa_panda_omron'
        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or expected_robot)
        if self.env_cfg_type != expected_robot:
            raise ValueError(f'{self.benchmark} checkpoints require env_cfg_type={expected_robot}')
        from XPolicyLab.utils.process_data import get_robot_action_dim_info

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        if (len(self.robot_action_dim_info["arm_dim"]) != 1
                or self.robot_action_dim_info["ee_dim"] != [1]):
            raise ValueError("LIBERO requires a single arm and one gripper action")

        gr00t_root = _resolve_gr00t_root(model_cfg)
        if str(gr00t_root) not in sys.path:
            sys.path.insert(0, str(gr00t_root))

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.model.policy import Gr00tPolicy
        import torch
        torch.manual_seed(int(model_cfg.get('seed', 0)))
        np.random.seed(int(model_cfg.get('seed', 0)))

        if self.benchmark == 'libero':
            from gr00t.experiment.data_config import load_data_config
            self.suite = _suite_name(model_cfg)
            data_config_name = str(model_cfg.get('data_config') or _data_config_name(self.suite))
            data_config = load_data_config(data_config_name)
            self._encode_observation = encode_libero_observation
            self._decode_action = decode_libero_action
        else:
            from gr00t.experiment.data_config import PandaOmronDataConfig
            from .robocasa import encode_observation, decode_action
            self.suite = 'pretrain'
            data_config_name = 'panda_omron'
            data_config = PandaOmronDataConfig()
            self._encode_observation = encode_observation
            self._decode_action = decode_action
        checkpoint = _checkpoint_dir(model_cfg)
        self.default_prompt = str(
            model_cfg.get("default_prompt") or "Perform the manipulation task."
        )

        self.policy = Gr00tPolicy(
            model_path=str(checkpoint),
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            denoising_steps=int(model_cfg.get("denoising_steps", 8)),
            device=str(model_cfg.get("device") or "cuda"),
        )
        self.model = self.policy
        self._obs_list: list[dict[str, Any]] = []
        self._env_ids: list[int] = []
        print(f"[GR00T_N15] suite={self.suite} data_config={data_config_name}")
        print(f"[GR00T_N15] checkpoint={checkpoint}")

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._env_ids = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(set(self._env_ids)) != len(self._env_ids):
            raise ValueError("Duplicate environment indices")
        self._obs_list = [
            self._encode_observation(obs, self.default_prompt) for obs in obs_list
        ]

    def get_action(self, **kwargs):
        if not self._obs_list:
            raise AssertionError("Call update_obs before get_action")
        return self._decode_action(self.policy.get_action(self._obs_list[0]))

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_list:
            raise AssertionError("Call update_obs_batch before get_action_batch")
        indices = self._env_ids if env_idx_list is None else list(env_idx_list)
        # Clients may pass explicit IDs separately from compact observations.
        if len(indices) == len(self._obs_list) and not set(indices).issubset(self._env_ids):
            self._env_ids = indices
        observations = dict(zip(self._env_ids, self._obs_list))
        if not set(indices).issubset(observations):
            raise ValueError("Requested environment has no current observation")
        return [self._decode_action(self.policy.get_action(observations[i])) for i in indices]

    def reset(self):
        self._obs_list = []
        self._env_ids = []
