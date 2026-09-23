"""CPU checks of raw source alignment, chunk boundaries, cumulative stats and model inputs."""
import json
import tempfile
from pathlib import Path
import numpy as np
from common import RUN, ASSETS, SUITES, H5Task, update_norm, make_config
from openpi.training import data_loader

tasks = json.loads((RUN / 'manifest.json').read_text())[SUITES[0]]
first, second = H5Task(tasks[0]), H5Task(tasks[1])
with tempfile.TemporaryDirectory(dir=RUN) as tmp:
    tmp = Path(tmp)
    norm = update_norm(first, None, tmp / 'm0.json')
    config = make_config(norm, tmp, ASSETS / 'checkpoints/cl_base/pi05_base', 8, 2)
    dc = config.data.create(config.assets_dirs, config.model)
    transformed = data_loader.transform_dataset(first, dc)
    for index in [0, tasks[0]['lengths'][0]-1, tasks[0]['lengths'][0], len(first)-1]:
        raw = first[index]
        sample = transformed[index]
        assert sample['actions'].shape == (10, 32)
        assert sample['state'].shape == (32,)
        assert sample['tokenized_prompt'].shape == (200,)
        assert sample['image']['base_0_rgb'].shape == (224,224,3)
        assert sample['image_mask']['right_wrist_0_rgb'] == False
        np.testing.assert_allclose(sample['actions'][:, :7] * (norm['actions'].std + 1e-6) + norm['actions'].mean,
                                   raw['actions'], rtol=1e-5, atol=1e-5)
    tail = first[tasks[0]['lengths'][0]-1]['actions']
    np.testing.assert_array_equal(tail, np.repeat(first.actions[0][-1:],10,axis=0))
    n2 = update_norm(second, tmp / 'm0.json', tmp / 'm1.json')
    actual = np.concatenate(first.states + second.states)
    np.testing.assert_allclose(n2['state'].mean, actual.astype(np.float64).mean(0), atol=1e-6)
    np.testing.assert_allclose(n2['state'].std, actual.astype(np.float64).std(0), atol=1e-6)
    assert json.loads((tmp/'m1.json').read_text())['state']['n'] == len(first)+len(second)
    lr = config.lr_schedule.create()
    np.testing.assert_allclose(float(lr(1000)), 2.5e-5, rtol=1e-5)
    np.testing.assert_allclose(float(lr(10000)), 2.5e-6, rtol=1e-5)
print('DATA_CHECKS_PASSED: source labels, episode boundaries, normalization, model shapes, learning-rate endpoints')
