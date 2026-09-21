"""Bounded RoboCasa365 client for any compatible XPolicyLab server."""
import argparse
import json
from pathlib import Path

from .contract import make_xpl_observation, xpl_action_to_gym


def run_episode(env, client, *, seed, max_steps):
    obs, _ = env.reset(seed=seed)
    client.call(func_name='reset')
    for step in range(max_steps):
        instruction = obs['annotation.human.task_description']
        client.call(func_name='update_obs', obs=make_xpl_observation(obs, instruction))
        chunk = client.call(func_name='get_action')
        if not chunk:
            raise ValueError('Policy returned an empty action chunk')
        obs, _, terminated, truncated, info = env.step(xpl_action_to_gym(chunk[0]))
        success = bool(info.get('success', False))
        if success or terminated or truncated:
            return {'success': success, 'steps': step + 1}
    return {'success': False, 'steps': max_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', choices=('pretrain', 'target'), required=True)
    parser.add_argument('--source-kind', choices=('human', 'mimicgen'), required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--episodes', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--max-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps < 1 or (args.source_kind == 'mimicgen' and args.split != 'pretrain'):
        parser.error('Positive step budget required; MimicGen uses split=pretrain')
    import gymnasium as gym
    import robocasa  # noqa: F401 -- registers the official gym environments
    from client_server.ws import WsModelClient

    client = WsModelClient(url=f'ws://{args.host}:{args.port}',
                           evaluation_id=f'robocasa365-{args.task}', trial_id='smoke')
    episodes = []
    try:
        for i in range(args.episodes):
            env = gym.make(f'robocasa/{args.task}', split=args.split, seed=args.seed + i)
            try:
                episodes.append(run_episode(env, client, seed=args.seed + i, max_steps=args.max_steps))
            finally:
                env.close()
    finally:
        client.close()
    # source-kind labels the training data. It is not a separate simulator type.
    result = dict(task=args.task, split=args.split, source_kind=args.source_kind,
                  seed=args.seed, episodes=episodes, purpose='integration_smoke_only')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
