"""RoboCasa365 closed-loop evaluation through the shared XPolicyLab interface."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .contract import make_xpl_observation, xpl_action_to_gym


def run_episode(env, client, *, seed, max_steps, action_steps=16, video_path=None):
    obs, _ = env.reset(seed=seed)
    client.call(func_name='reset')
    instruction = obs['annotation.human.task_description']
    writer = None
    if video_path is not None:
        import imageio.v2 as imageio
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=20, codec='libx264')

    def record_frame():
        if writer is not None:
            writer.append_data(np.concatenate([
                obs['video.robot0_agentview_left'],
                obs['video.robot0_agentview_right'],
                obs['video.robot0_eye_in_hand'],
            ], axis=1))

    started = time.monotonic()
    steps = calls = 0
    success = terminated = truncated = False
    try:
        record_frame()
        while steps < max_steps:
            client.call(func_name='update_obs', obs=make_xpl_observation(obs, instruction))
            chunk = client.call(func_name='get_action')
            calls += 1
            if not chunk:
                raise ValueError('Policy returned an empty action chunk')
            for action in chunk[:action_steps]:
                obs, _, terminated, truncated, info = env.step(xpl_action_to_gym(action))
                steps += 1
                success = bool(info.get('success', False))
                record_frame()
                if success or terminated or truncated or steps >= max_steps:
                    break
            if success or terminated or truncated:
                break
    finally:
        if writer is not None:
            writer.close()
    return {
        'seed': seed, 'instruction': instruction, 'success': success,
        'steps': steps, 'policy_calls': calls, 'elapsed_seconds': time.monotonic() - started,
        'terminated': bool(terminated), 'truncated': bool(truncated),
        'video': str(video_path) if video_path is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--split', choices=('pretrain', 'target'), required=True)
    parser.add_argument('--source-kind', choices=('human', 'mimicgen'), required=True)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Defaults to the official task registry horizon')
    parser.add_argument('--action-steps', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path)
    args = parser.parse_args()
    if (args.episodes < 1 or args.action_steps < 1
            or (args.max_steps is not None and args.max_steps < 1)
            or (args.source_kind == 'mimicgen' and args.split != 'pretrain')):
        parser.error('Positive step budget required; MimicGen uses split=pretrain')
    import gymnasium as gym
    import robocasa  # noqa: F401 -- registers the official gym environments
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
    from client_server.ws import WsModelClient

    client = WsModelClient(url=f'ws://{args.host}:{args.port}',
                           evaluation_id=f'robocasa365-{args.task}', trial_id='smoke')
    registry = ATOMIC_TASK_DATASETS | COMPOSITE_TASK_DATASETS
    max_steps = args.max_steps or registry[args.task]['horizon']
    episodes = []
    result = dict(task=args.task, split=args.split, source_kind=args.source_kind,
                  seed=args.seed, max_steps=max_steps, action_steps=args.action_steps,
                  robocasa_version=getattr(robocasa, '__version__', 'unknown'),
                  episodes=episodes, purpose='single_task_sanity_check')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_result():
        result['successes'] = sum(ep['success'] for ep in episodes)
        result['completed_episodes'] = len(episodes)
        result['success_rate'] = result['successes'] / len(episodes) if episodes else None
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    try:
        for i in range(args.episodes):
            env = gym.make(f'robocasa/{args.task}', split=args.split, seed=args.seed + i)
            try:
                video_path = (args.video_dir / f'{args.task}_seed{args.seed + i}.mp4'
                              if args.video_dir is not None else None)
                episode = run_episode(env, client, seed=args.seed + i, max_steps=max_steps,
                                      action_steps=args.action_steps, video_path=video_path)
                episodes.append(episode)
                save_result()
                print(json.dumps(episode), flush=True)
            finally:
                env.close()
    finally:
        client.close()
    # source-kind labels the training data. It is not a separate simulator type.
    save_result()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
