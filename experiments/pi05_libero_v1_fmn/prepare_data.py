"""(openpi env) Map benchmark tasks to physical-intelligence/libero task_index, list episode files,
and compute suite-level mean/std normalization once (FMN-style)."""
import json, os, sys
import numpy as np
from pathlib import Path
from common import DATA, RUN, SUITES, write_json, _fixed
from paths import relative_path

def norm_text(s):
    return ' '.join(str(s).lower().replace('.', ' ').split())

bench = json.loads(Path(sys.argv[1]).read_text())
info = json.loads((DATA / 'meta/info.json').read_text())
tasks = [json.loads(l) for l in (DATA / 'meta/tasks.jsonl').read_text().splitlines() if l.strip()]
episodes = [json.loads(l) for l in (DATA / 'meta/episodes.jsonl').read_text().splitlines() if l.strip()]
by_text = {}
for t in tasks:
    by_text.setdefault(norm_text(t['task']), []).append(t['task_index'])
assert all(len(v) == 1 for v in by_text.values()), 'duplicate task strings in dataset'
chunk = info['chunks_size']
pattern = info['data_path']
manifest = {}
import pyarrow.parquet as pq
from openpi.shared import normalize
for suite in SUITES:
    entries, all_state, all_action = [], [], []
    for b in bench[suite]:
        key = norm_text(b['instruction'])
        assert key in by_text, (suite, b['instruction'])
        tidx = by_text[key][0]
        eps = sorted(e['episode_index'] for e in episodes if norm_text(e['tasks'][0]) == key)
        files = [str(DATA / pattern.format(episode_chunk=e // chunk, episode_index=e)) for e in eps]
        missing = [f for f in files if not Path(f).exists()]
        if os.environ.get('SMOKE_PARTIAL'):
            files = [f for f in files if Path(f).exists()]
        else:
            assert not missing, (suite, b['instruction'], len(missing))
        frames = 0
        for f in files:
            t = pq.read_table(f, columns=['state', 'actions', 'task_index'])
            assert set(t.column('task_index').to_numpy().tolist()) == {tidx}, f
            all_state.append(_fixed(t.column('state'), 8)); all_action.append(_fixed(t.column('actions'), 7))
            frames += t.num_rows
        entries.append(dict(b, task_index=tidx, episode_files=[relative_path(f, DATA) for f in files], episode_path_base='V1_DATA', episodes=len(files), frames=frames))
    assert len({e['task_index'] for e in entries}) == 10
    manifest[suite] = entries
    if not all_state:
        continue
    state, action = np.concatenate(all_state).astype(np.float64), np.concatenate(all_action).astype(np.float64)
    stats = {}
    for name, v in [('state', state), ('actions', action)]:
        stats[name] = normalize.NormStats(mean=v.mean(0).astype(np.float32), std=v.std(0).astype(np.float32),
                                          q01=np.quantile(v, 0.01, axis=0).astype(np.float32),
                                          q99=np.quantile(v, 0.99, axis=0).astype(np.float32))
    normalize.save(RUN / 'norm' / suite, stats)
    print(suite, 'episodes', [e['episodes'] for e in entries], 'frames', sum(e['frames'] for e in entries), flush=True)
write_json(RUN / 'manifest.json', manifest)
print('DATA_PREPARED')
