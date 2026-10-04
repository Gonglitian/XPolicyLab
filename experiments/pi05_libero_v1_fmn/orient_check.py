"""(simulator env, frames dumped by orient_dump.py) Empirical orientation check: dataset frames should match the evaluation input, i.e. the
simulator render rotated 180 deg (openpi LIBERO recipe), and NOT the raw render."""
from egl_device import configure_egl
configure_egl()
import io, json
import numpy as np
from PIL import Image
from libero.libero import benchmark
from common import RUN, SUITES, write_json
from XPolicyLab.benchmarks.libero.client import make_env

dumped = np.load(RUN / 'orient_frames.npz')
report = {}
for suite in SUITES:
    bench = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    task = bench.get_task(0)
    env = make_env(task, 42, 256)
    env.reset()
    obs = env.set_init_state(np.asarray(bench.get_task_init_states(0))[0])
    for _ in range(10):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
    raw = obs['agentview_image'].astype(np.float32)
    env.close()
    rotated = raw[::-1, ::-1]
    d_rot, d_raw = [], []
    for img in dumped[suite].astype(np.float32):
        d_rot.append(float(np.abs(img - rotated).mean())); d_raw.append(float(np.abs(img - raw).mean()))
    report[suite] = dict(mean_abs_diff_vs_rotated=float(np.median(d_rot)), mean_abs_diff_vs_raw=float(np.median(d_raw)))
    report[suite]['matches_rotated'] = report[suite]['mean_abs_diff_vs_rotated'] < 0.7 * report[suite]['mean_abs_diff_vs_raw']
    Image.fromarray(np.concatenate([img, rotated], 1).astype(np.uint8)).save(RUN / f'orient_{suite}.png')
write_json(RUN / 'orientation_check.json', report)
print('ORIENTATION', json.dumps(report))
assert all(v['matches_rotated'] for v in report.values()), report
