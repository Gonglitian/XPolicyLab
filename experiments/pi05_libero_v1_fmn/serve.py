"""Serve a v1 checkpoint with its own saved normalization (same model config as training)."""
import argparse
import asyncio
from slurm_runtime import require_slurm
require_slurm()
import jax
from pathlib import Path
from common import make_config, write_json
import os
from openpi.shared import normalize
from openpi.policies import policy_config
from XPolicyLab.policy.Pi_05.model import Model as BaseModel
from client_server.ws.model_server import PolicyServer, PolicyServerConfig

class Model(BaseModel):
    def __init__(self, checkpoint):
        self._is_libero = True
        self._is_robocasa = False
        self.observation_window = None
        self._latest_env_idx_list = [0]
        norms = normalize.load(checkpoint / 'assets/formal_libero')
        cfg = make_config(norms, checkpoint.parent, 8, 1)
        self.policy = policy_config.create_trained_policy(cfg, checkpoint, norm_stats=norms)
        self.model = self.policy
    def reset(self, seed=42):
        super().reset()
        self.policy._rng = jax.random.key(int(seed))
    def set_seed(self, obs):
        self.policy._rng = jax.random.key(int(obs['seed']))
        return int(obs['seed'])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--ready-file', type=Path, required=True)
    a = p.parse_args()
    model = Model(a.checkpoint)
    print('POLICY_LOADED', str(a.checkpoint), flush=True)
    async def run():
        server = PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=a.port, ws_ping_timeout_s=600))
        await server.start()  # port=0 binds an OS-assigned port before publishing it
        try:
            write_json(a.ready_file, dict(port=int(server.url.rsplit(':', 1)[1]),
                       pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID']))
            await server.serve_forever()
        finally:
            await server.stop()
    asyncio.run(run())
