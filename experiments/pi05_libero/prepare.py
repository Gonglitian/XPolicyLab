"""Inventory the official LIBERO task order and the 50 source demonstrations."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from libero.libero import benchmark

from settings import ASSETS as ROOT, RUN as OUT, SUITES
OUT.mkdir(parents=True, exist_ok=True)
manifest = {}
for suite_name in SUITES:
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        source = ROOT / 'datasets/libero' / suite_name / (task.name + '_demo.hdf5')
        initial = np.asarray(suite.get_task_init_states(i))
        assert len(initial) >= 50, (suite_name, i, initial.shape)
        with h5py.File(source, 'r') as f:
            ids = sorted(f['data'], key=lambda x: int(x.split('_')[-1]))
            assert len(ids) >= 50, (source, len(ids))
            ids = ids[:50]
            lengths = [len(f['data'][d]['actions']) for d in ids]
            for d, n in zip(ids, lengths):
                demo = f['data'][d]
                assert n > 0 and demo['actions'].shape == (n, 7)
                for key in ['agentview_rgb', 'eye_in_hand_rgb', 'ee_pos', 'ee_ori', 'gripper_states']:
                    assert len(demo['obs'][key]) == n
            instruction = json.loads(f['data'].attrs['problem_info'])['language_instruction']
            assert str(instruction).strip() == str(task.language).strip(), (instruction, task.language)
        tasks.append(dict(task_id=i, task_name=task.name, instruction=task.language,
                          source=str(source), demos=ids, lengths=lengths, frames=sum(lengths),
                          source_size=source.stat().st_size,
                          initial_states_sha256=hashlib.sha256(initial[:50].tobytes()).hexdigest()))
    assert len(tasks) == 10
    manifest[suite_name] = tasks
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({s: {'tasks': len(ts), 'frames': [t['frames'] for t in ts]} for s, ts in manifest.items()}, indent=2))
