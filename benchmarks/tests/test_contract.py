import ast
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
from XPolicyLab.benchmarks.libero.client import make_xpl_observation, xpl_action_to_libero, run_episode
from XPolicyLab.benchmarks.geometry import quat_wxyz_to_axis_angle
from XPolicyLab.benchmarks.robocasa365.contract import make_xpl_observation as rc_observation
from XPolicyLab.policy.GR00T_N15.model import encode_libero_observation, decode_libero_action
from XPolicyLab.policy.Pi_05.libero_adapter import encode_observation as pi_observation, decode_action as pi_action
from XPolicyLab.policy.X_VLA.libero_adapter import encode_observation as xv_observation, decode_action as xv_action


def native_obs():
    return {'agentview_image': np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            'robot0_eye_in_hand_image': np.arange(36, 72, dtype=np.uint8).reshape(3, 4, 3),
            'robot0_eef_pos': [0.1, 0.2, 0.3], 'robot0_eef_quat': [0, 0, 0, 1],
            'robot0_gripper_qpos': [0.04, -0.04]}


def test_benchmark_has_no_policy_imports():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob('*.py'):
        if 'tests' in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'policy' not in (node.module or '').split('.'), path
            elif isinstance(node, ast.Import):
                assert all('policy' not in alias.name.split('.') for alias in node.names), path


def test_shared_observation_preserves_native_rgb_and_has_named_pose():
    native = native_obs()
    obs = make_xpl_observation(native, 'pick')
    np.testing.assert_array_equal(obs['vision']['agentview']['color'], native['agentview_image'])
    np.testing.assert_array_equal(obs['state']['ee_pose'][3:], [1, 0, 0, 0])
    assert obs['instruction'] == 'pick'


def test_three_policy_recipes_own_image_orientation():
    native = native_obs()
    obs = make_xpl_observation(native, 'pick')
    n15 = encode_libero_observation(obs, '')
    pi = pi_observation(obs, resize_image=lambda image: image)
    xv = xv_observation(obs)
    for actual in (n15['video.image'][0], pi['observation/image'], xv['images'][0]):
        np.testing.assert_array_equal(actual, native['agentview_image'][::-1, ::-1, :])
    for actual in (n15['video.wrist_image'][0], pi['observation/wrist_image']):
        np.testing.assert_array_equal(actual, native['robot0_eye_in_hand_image'][::-1, ::-1, :])
    np.testing.assert_array_equal(xv['images'][1], native['robot0_eye_in_hand_image'])
    np.testing.assert_array_equal(obs['vision']['agentview']['color'], native['agentview_image'])


@pytest.mark.parametrize('raw', [-1.0, -0.4, 0.0, 0.4, 1.0])
def test_pi_preserves_continuous_gripper_command(raw):
    actions = np.array([[.1, .2, .3, .04, -.05, .06, raw]], np.float32)
    restored = xpl_action_to_libero(pi_action(actions)[0])
    np.testing.assert_allclose(restored, actions[0], atol=1e-6)


@pytest.mark.parametrize('openness, expected', [(0, 1), (.25, 1), (.5, 0), (.75, -1), (1, -1)])
def test_n15_owns_gripper_binarization(openness, expected):
    native = {f'action.{key}': np.zeros((1, 1), np.float32)
              for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')}
    native['action.gripper'][0, 0] = openness
    assert xpl_action_to_libero(decode_libero_action(native)[0])[-1] == expected


def test_xvla_column_rotation_and_absolute_mode():
    # 90 degrees about z: columns are (0,1,0) and (-1,0,0).
    raw = np.array([[.1, .2, .3, 0, 1, 0, -1, 0, 0, .6] + [0]*10], np.float32)
    action = xv_action(raw)[0]
    assert action['ee_pose_mode'] == 'absolute'
    np.testing.assert_allclose(xpl_action_to_libero(action), [.1, .2, .3, 0, 0, np.pi/2, 1], atol=1e-6)
    state = xv_observation(make_xpl_observation(native_obs(), 'pick'))['proprio']
    np.testing.assert_array_equal(state[3:9], [1, 0, 0, 0, 1, 0])


def test_invalid_quaternion_and_actions_are_rejected():
    with pytest.raises(ValueError):
        quat_wxyz_to_axis_angle([0, 0, 0, 0])
    with pytest.raises(ValueError):
        xv_action(np.zeros((1, 20)))
    with pytest.raises(ValueError):
        pi_action(np.zeros((1, 12)))
    with pytest.raises(ValueError):
        xpl_action_to_libero({'ee_pose': [0, 0, 0, 1, 0, 0, 0], 'ee_joint_state': [-1]})


def test_absolute_control_starts_after_settling(monkeypatch):
    transforms = ModuleType('robosuite.utils.transform_utils')
    transforms.mat2quat = lambda matrix: np.array([0, 0, 0, 1])
    monkeypatch.setitem(sys.modules, 'robosuite.utils.transform_utils', transforms)
    # Reusing an environment after an absolute-control episode must remain safe.
    controller = SimpleNamespace(use_delta=False, ee_pos=np.zeros(3), ee_ori_mat=np.eye(3))
    modes = []
    obs = native_obs()
    def step(action):
        modes.append(controller.use_delta)
        return obs, 0, False, {}
    env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)]), step=step)
    action = {'ee_pose': [0, 0, 0, 1, 0, 0, 0], 'ee_joint_state': [1], 'ee_pose_mode': 'absolute'}
    received = []
    def call(func_name, **kwargs):
        if func_name == 'get_action':
            return [action] * 4
        received.append(kwargs['obs'])
    success, steps = run_episode(env, SimpleNamespace(call=call), obs, 'pick',
                                 max_steps=3, wait_steps=2, chunk_steps=4)
    assert not success and steps == 3
    assert modes == [True, True, False, False, False]
    assert 'controller_ee_pose' in received[0]['state']


def test_robocasa_observation_is_policy_neutral():
    native = {key: np.zeros((3, 4, 3), np.uint8) for key in (
        'video.robot0_agentview_left', 'video.robot0_agentview_right', 'video.robot0_eye_in_hand')}
    native.update({'state.base_position': [1, 2, 3], 'state.base_rotation': [0, 0, 0, 1],
                   'state.end_effector_position_relative': [.1, .2, .3],
                   'state.end_effector_rotation_relative': [0, 0, 0, 1], 'state.gripper_qpos': [.04, -.04]})
    result = rc_observation(native, 'pick')
    assert set(result['state']) == {'base_pose', 'ee_pose', 'gripper_qpos'}
    np.testing.assert_array_equal(result['state']['base_pose'], [1, 2, 3, 1, 0, 0, 0])
