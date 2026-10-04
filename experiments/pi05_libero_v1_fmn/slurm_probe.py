"""Short Slurm GPU, actual websocket and LIBERO rendering check; no training."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from slurm_runtime import require_slurm


def policy_probe():
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != 'gpu':
        raise RuntimeError(f'Expected one GPU, got {devices}')
    result = float(jax.device_get(jnp.sum(jnp.ones((16, 16)))))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from client_server.ws import WsModelClient
    class Model:
        def reset(self): return 'probe-ok'
    async def run():
        servers, ports = [], []
        try:
            for _ in range(4):
                server = PolicyServer(Model(), PolicyServerConfig(host='127.0.0.1', port=0))
                servers.append(server)
                await server.start()
                ports.append(int(server.url.rsplit(':', 1)[1]))
            def request(port):
                client = WsModelClient(url=f'ws://127.0.0.1:{port}', evaluation_id='slurm-entry-check', trial_id='probe')
                try: return client.call(func_name='reset')
                finally: client.close()
            responses = await asyncio.gather(*(asyncio.to_thread(request, port) for port in ports))
            if len(set(ports)) != 4: raise RuntimeError('Port collision')
            return dict(ports=ports, responses=responses)
        finally:
            for server in servers: await server.stop()
    return dict(devices=[str(d) for d in devices], sum=result, **asyncio.run(run()))


def sim_probe(suite_name):
    from egl_device import configure_egl
    index = configure_egl()
    from libero.libero import benchmark
    from XPolicyLab.benchmarks.libero.client import make_env
    suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    env = make_env(suite.get_task(0), 42, 256)
    try:
        obs = env.reset()
        image = obs['agentview_image']
        if image.shape != (256, 256, 3): raise RuntimeError(f'Unexpected render shape: {image.shape}')
        return dict(egl_device=index, image_shape=list(image.shape), image_mean=float(image.mean()))
    finally:
        env.close()


def main():
    job = require_slurm()
    p = argparse.ArgumentParser()
    p.add_argument('--role', choices=['policy', 'sim'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = policy_probe() if args.role == 'policy' else sim_probe(args.suite)
    result.update(job_id=job, pid=os.getpid(), role=args.role,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  cgroup=Path('/proc/self/cgroup').read_text())
    from common import write_json
    write_json(args.output, result)
    print('SLURM_PROBE_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
