"""Serve a v1 checkpoint with its own saved normalization (same model config as training)."""
import argparse
import asyncio
import jax
from pathlib import Path
from common import make_config
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
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    model = Model(a.checkpoint)
    print('POLICY_LOADED', str(a.checkpoint), flush=True)
    asyncio.run(PolicyServer(model, PolicyServerConfig(host='127.0.0.1', port=a.port,
        ws_ping_timeout_s=600)).serve_forever())
