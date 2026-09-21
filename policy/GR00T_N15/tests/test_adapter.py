import json
from pathlib import Path

import numpy as np

from XPolicyLab.policy.GR00T_N15.clients.libero_eval import xpl_action_to_libero
from XPolicyLab.policy.GR00T_N15.clients.robocasa365_contract import modality_to_raw_action
from XPolicyLab.policy.GR00T_N15.model import (
    _axis_angle_to_quat_wxyz,
    _data_config_name,
    _quat_wxyz_to_axis_angle,
    decode_libero_action,
    encode_libero_observation,
)


def test_goal_uses_mean_std_config():
    assert _data_config_name("libero_goal").endswith("LiberoDataConfigMeanStd")
    assert _data_config_name("libero_spatial").endswith("LiberoDataConfig")


def test_axis_angle_quaternion_round_trip():
    axis_angle = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    restored = _quat_wxyz_to_axis_angle(_axis_angle_to_quat_wxyz(axis_angle))
    np.testing.assert_allclose(restored, axis_angle, atol=1e-6)


def test_libero_observation_mapping_keeps_rgb_order():
    agent = np.zeros((4, 5, 3), dtype=np.uint8)
    agent[..., 0] = 255
    wrist = np.zeros((4, 5, 3), dtype=np.uint8)
    obs = {
        "vision": {"agentview": {"color": agent}, "wrist": {"color": wrist}},
        "state": {
            "ee_position": [1, 2, 3],
            "ee_axis_angle": [0.1, 0.2, 0.3],
            "gripper_qpos": [0.4, 0.5],
        },
        "instruction": "pick up the bowl",
    }
    encoded = encode_libero_observation(obs, "fallback")
    assert encoded["video.image"].shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(encoded["video.image"][0], agent)
    assert encoded["annotation.human.action.task_description"] == ["pick up the bowl"]


def test_libero_action_mapping_has_standard_xpl_keys():
    action = {
        f"action.{key}": np.full((2, 1), value, dtype=np.float32)
        for key, value in zip(
            ("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
            (0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.75),
        )
    }
    chunk = decode_libero_action(action)
    assert len(chunk) == 2
    assert set(chunk[0]) == {"ee_pose", "ee_joint_state"}
    assert chunk[0]["ee_pose"].shape == (7,)
    restored = xpl_action_to_libero(chunk[0])
    np.testing.assert_allclose(restored[:6], [0.1, 0.2, 0.3, 0.01, 0.02, 0.03], atol=1e-6)
    assert restored[6] == -1.0


def test_robocasa365_action_order():
    modality = np.arange(12, dtype=np.float32)
    raw = modality_to_raw_action(modality)
    np.testing.assert_array_equal(raw, [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])


def test_manifest_keeps_human_and_mimicgen_separate():
    config = Path(__file__).resolve().parent.parent / "configs" / "data_sources.json"
    sources = json.loads(config.read_text(encoding="utf-8"))
    assert set(sources["robocasa365"]) == {"human", "mimicgen"}
