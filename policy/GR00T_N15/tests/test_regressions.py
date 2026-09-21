import json
from types import SimpleNamespace

import numpy as np
import pytest

from XPolicyLab.policy.GR00T_N15.model import Model, _checkpoint_dir, _resolve_gr00t_root, decode_libero_action
from XPolicyLab.policy.GR00T_N15.clients.libero_eval import run_episode
from XPolicyLab.policy.GR00T_N15.clients.robocasa365_contract import build_state
from XPolicyLab.policy.GR00T_N15.prepare_dataset import prepare, validate_dataset
from XPolicyLab.policy.GR00T_N15.setup_workspace import register


def action(value=0.0):
    return {f'action.{key}': np.array([[value]], dtype=np.float32)
            for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')}


def test_root_environment_is_respected(tmp_path, monkeypatch):
    (tmp_path / 'gr00t/model').mkdir(parents=True)
    (tmp_path / 'gr00t/model/policy.py').touch()
    monkeypatch.setenv('GR00T_N15_ROOT', str(tmp_path))
    assert _resolve_gr00t_root({'gr00t_root': None}) == tmp_path


def test_checkpoint_index_without_shard_is_rejected(tmp_path):
    (tmp_path / 'experiment_cfg').mkdir()
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'experiment_cfg/metadata.json').write_text('{}')
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': 'missing.safetensors'}}))
    with pytest.raises(FileNotFoundError, match='shard'):
        _checkpoint_dir({'model_path': str(tmp_path)})


def test_nan_action_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        decode_libero_action(action(float('nan')))


def test_batch_reorder_and_subset():
    model = Model.__new__(Model)
    model._obs_list = [{'value': 0.1}, {'value': 0.2}]
    model._env_ids = [2, 7]
    model.policy = SimpleNamespace(get_action=lambda obs: action(obs['value']))
    results = model.get_action_batch([7, 2])
    assert results[0][0]['ee_pose'][0] == pytest.approx(0.2)
    assert len(model.get_action_batch([7])) == 1
    assert model.get_action()[0]['ee_pose'][0] == pytest.approx(0.1)


def test_chunked_rollout_respects_control_step_budget():
    obs = {'agentview_image': np.zeros((4, 4, 3), np.uint8),
           'robot0_eye_in_hand_image': np.zeros((4, 4, 3), np.uint8),
           'robot0_eef_pos': np.zeros(3), 'robot0_eef_quat': [0, 0, 0, 1],
           'robot0_gripper_qpos': [0, 0]}
    env = SimpleNamespace(calls=0)
    def step(_):
        env.calls += 1
        return obs, 0, False, {}
    env.step = step
    chunk = decode_libero_action(action()) * 4
    client = SimpleNamespace(call=lambda func_name, **kwargs: chunk if func_name == 'get_action' else None)
    success, steps = run_episode(env, client, obs, 'pick', max_steps=5, wait_steps=2, chunk_steps=4)
    assert not success and steps == 5 and env.calls == 7


def test_robocasa_missing_base_state_fails():
    with pytest.raises(KeyError, match='base position'):
        build_state({})


def dataset_fixture(tmp_path):
    source = tmp_path / 'source'
    (source / 'meta').mkdir(parents=True)
    (source / 'data').mkdir()
    info = {'codebase_version': 'v2.1', 'total_episodes': 1, 'total_frames': 1,
            'features': {'observation.state': {'shape': [8]}, 'action': {'shape': [7]},
                         'observation.images.image': {'dtype': 'image'}}}
    (source / 'meta/info.json').write_text(json.dumps(info))
    for name in ('tasks.jsonl', 'episodes.jsonl', 'stats.json'):
        (source / 'meta' / name).write_text('{}\n')
    modality = {'state': {'all': {'start': 0, 'end': 8}},
                'action': {'all': {'start': 0, 'end': 7}},
                'video': {'image': {'original_key': 'observation.images.image'}}}
    return source, modality


def test_dataset_view_preserves_source(tmp_path):
    source, modality = dataset_fixture(tmp_path)
    before = (source / 'meta/info.json').read_bytes()
    target = tmp_path / 'view'
    prepare(source, target, 'libero', modality, {'source_kind': 'libero'})
    assert (target / 'data').resolve() == source / 'data'
    assert (source / 'meta/info.json').read_bytes() == before
    assert not (source / 'meta/modality.json').exists()
    with pytest.raises(FileExistsError):
        prepare(source, target, 'libero', modality, {})


def test_v3_and_overlapping_modality_rejected(tmp_path):
    source, modality = dataset_fixture(tmp_path)
    modality['state']['overlap'] = {'start': 0, 'end': 1}
    with pytest.raises(ValueError, match='exactly once'):
        validate_dataset(source, 'libero', modality)
    info = json.loads((source / 'meta/info.json').read_text())
    info['codebase_version'] = 'v3.0'
    (source / 'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='v2'):
        validate_dataset(source, 'libero', modality)


def test_workspace_registration_preserves_other_robots(tmp_path):
    root = tmp_path / 'env_cfg/robot'
    root.mkdir(parents=True)
    registry = root / '_robot_info.json'
    registry.write_text(json.dumps({'other': {'arm_dim': [6], 'ee_dim': [2]}}))
    register(tmp_path)
    register(tmp_path)
    assert json.loads(registry.read_text())['other']['ee_dim'] == [2]
    (tmp_path / 'env_cfg/libero_franka.yml').write_text('config: {robot: other}\n')
    with pytest.raises(ValueError, match='differs'):
        register(tmp_path)


def test_robocasa_action_retains_mobile_fields_and_gripper_polarity():
    from XPolicyLab.policy.GR00T_N15.clients.robocasa365_contract import xpl_action_to_gym
    value = {'ee_pose': [0, 0, 0, 1, 0, 0, 0], 'ee_joint_state': [0.75],
             'base_motion': [0.1, 0.2, 0.3, 0.4], 'control_mode': [1]}
    mapped = xpl_action_to_gym(value)
    assert mapped['action.gripper_close'][0] == 0.75
    np.testing.assert_allclose(mapped['action.base_motion'], value['base_motion'])
    with pytest.raises(KeyError):
        xpl_action_to_gym({'ee_pose': value['ee_pose'], 'ee_joint_state': [0.75]})


def test_robocasa_truncation_is_not_success():
    from XPolicyLab.policy.GR00T_N15.clients.robocasa365_eval import run_episode as run_robocasa
    obs = {key: np.zeros((4, 4, 3), np.uint8) for key in (
        'video.robot0_agentview_left', 'video.robot0_agentview_right', 'video.robot0_eye_in_hand')}
    obs.update({'state.base_position': [0, 0, 0], 'state.base_rotation': [0, 0, 0, 1],
                'state.end_effector_position_relative': [0, 0, 0],
                'state.end_effector_rotation_relative': [0, 0, 0, 1], 'state.gripper_qpos': [0, 0],
                'annotation.human.task_description': 'pick'})
    value = {'ee_pose': [0, 0, 0, 1, 0, 0, 0], 'ee_joint_state': [0],
             'base_motion': [0, 0, 0, 0], 'control_mode': [0]}
    env = SimpleNamespace(reset=lambda seed: (obs, {}), step=lambda act: (obs, 0, False, True, {'success': False}))
    client = SimpleNamespace(call=lambda func_name, **kwargs: [value] if func_name == 'get_action' else None)
    result = run_robocasa(env, client, seed=0, max_steps=5)
    assert result == {'success': False, 'steps': 1}
