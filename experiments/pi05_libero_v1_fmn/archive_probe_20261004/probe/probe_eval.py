"""Evaluate ONE task on its fixed initial states (same protocol as evaluate.py, which evaluates tasks 0..stage)."""
import argparse
import hashlib
import json
import time
import numpy as np
from pathlib import Path
from common import RUN, HORIZONS, write_json
from XPolicyLab.benchmarks.libero.client import make_env, run_episode
from client_server.ws import WsModelClient
from libero.libero import benchmark

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    suite = benchmark.get_benchmark_dict()[a.suite](task_order_index=0)
    manifest = json.loads((RUN / 'manifest.json').read_text())[a.suite]
    records = json.loads(a.output.read_text())['episodes'] if a.output.exists() else []
    done_keys = {r['episode'] for r in records}
    write_json(a.output, dict(suite=a.suite, task=a.task, worker=a.worker, episodes=records))
    client = WsModelClient(url=f'ws://127.0.0.1:{a.port}', evaluation_id=f'{a.suite}-task{a.task}-w{a.worker}',
                           trial_id='probe', request_timeout_s=600)
    try:
        task = suite.get_task(a.task)
        states = np.asarray(suite.get_task_init_states(a.task))
        assert hashlib.sha256(states[:50].tobytes()).hexdigest() == manifest[a.task]['initial_states_sha256']
        env = make_env(task, 42, 256)
        try:
            for episode in range(a.worker, a.episodes, a.workers):
                if episode in done_keys:
                    continue
                start = time.monotonic()
                seed = 42 + a.task * 1000 + episode
                env.seed(seed)
                np.random.seed(seed)
                env.reset()
                obs = env.set_init_state(states[episode])
                client.call(func_name='reset')
                client.call(func_name='set_seed', obs={'seed': seed})
                success, steps = run_episode(env, client, obs, task.language,
                    max_steps=HORIZONS[a.suite], wait_steps=10, chunk_steps=5)
                record = dict(task=a.task, task_name=task.name, episode=episode, seed=seed, success=bool(success),
                    steps=steps, seconds=time.monotonic() - start, timestamp=time.time())
                records.append(record)
                write_json(a.output, dict(suite=a.suite, task=a.task, worker=a.worker, episodes=records))
                print('EVAL_EPISODE', json.dumps(record), flush=True)
        finally:
            env.close()
    finally:
        client.close()

if __name__ == '__main__':
    main()
