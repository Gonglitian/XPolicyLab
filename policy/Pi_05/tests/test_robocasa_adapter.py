import numpy as np
import pytest

from XPolicyLab.benchmarks.robocasa365.contract import make_xpl_observation, xpl_action_to_gym
from XPolicyLab.policy.Pi_05.robocasa_adapter import encode_observation, decode_action


def test_official_eef_first_state_and_native_camera_orientation():
    native = {key: np.arange(36, dtype=np.uint8).reshape(3, 4, 3) for key in (
        'video.robot0_agentview_left', 'video.robot0_eye_in_hand', 'video.robot0_agentview_right')}
    native.update({'state.end_effector_position_relative': [.1, .2, .3],
                   'state.end_effector_rotation_relative': [0, 0, .6, .8],
                   'state.base_position': [1, 2, 3], 'state.base_rotation': [0, .6, 0, .8],
                   'state.gripper_qpos': [.04, -.04]})
    result = encode_observation(make_xpl_observation(native, 'open drawer'), resize_image=lambda x: x)
    expected = np.concatenate([native[k] for k in (
        'state.end_effector_position_relative', 'state.end_effector_rotation_relative',
        'state.base_position', 'state.base_rotation', 'state.gripper_qpos')])
    np.testing.assert_allclose(result['observation/state'], expected)
    np.testing.assert_array_equal(result['observation/image'], native['video.robot0_agentview_left'])
    np.testing.assert_array_equal(result['observation/right_image'], native['video.robot0_agentview_right'])


def test_official_action_order_roundtrip_including_mobile_fields():
    raw = np.array([[.1, -.2, .3, .05, -.08, .12, .7, .11, .22, -.33, .44, .8]], np.float32)
    restored = xpl_action_to_gym(decode_action(raw)[0])
    actual = np.concatenate([restored[k] for k in (
        'action.end_effector_position', 'action.end_effector_rotation',
        'action.gripper_close', 'action.base_motion', 'action.control_mode')])
    np.testing.assert_allclose(actual, raw[0], atol=1e-6)
    with pytest.raises(ValueError):
        decode_action(np.zeros((1, 7)))
