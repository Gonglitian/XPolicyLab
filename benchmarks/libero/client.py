"""LIBERO client for any policy implementing the shared benchmark contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from XPolicyLab.benchmarks.geometry import quat_wxyz_to_axis_angle, xyzw_to_wxyz


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def make_xpl_observation(obs: dict, instruction: str) -> dict:
    # Preserve native RGB orientation. Checkpoint preprocessing is policy-owned.
    agent = np.ascontiguousarray(np.asarray(obs["agentview_image"]))
    wrist = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"]))
    return {
        "vision": {
            "agentview": {"color": agent},
            "wrist": {"color": wrist},
        },
        "state": {
            "ee_pose": np.concatenate((np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                                        xyzw_to_wxyz(obs["robot0_eef_quat"]))),
            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        },
        "instruction": instruction,
        "benchmark": "libero",
        "data_format_version": "evomoe-benchmark-v1",
    }


def xpl_action_to_libero(action: dict) -> np.ndarray:
    pose = np.asarray(action["ee_pose"], dtype=np.float32).reshape(7)
    gripper = np.asarray(action["ee_joint_state"], dtype=np.float32).reshape(-1)
    if gripper.size != 1:
        raise ValueError(f"Expected one gripper action, got {gripper.shape}")
    if not np.isfinite(pose).all() or not np.isfinite(gripper).all() or not 0 <= gripper[0] <= 1:
        raise ValueError('Expected finite pose and gripper closure in [0,1]')
    return np.concatenate(
        (pose[:3], quat_wxyz_to_axis_angle(pose[3:]), [2.0 * float(gripper[0]) - 1.0])
    ).astype(np.float32)


def make_env(task, seed: int, resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def run_episode(env, client, obs, instruction, *, max_steps, wait_steps, chunk_steps, record_frame=None):
    """Count simulator control steps, not policy queries, against the budget."""
    if hasattr(env, 'env'):
        for robot in env.env.robots:
            robot.controller.use_delta = True
    for _ in range(wait_steps):
        obs, _, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
        if done:
            return True, 0
    steps = 0
    if record_frame is not None:
        record_frame(obs)
    while steps < max_steps:
        observation = make_xpl_observation(obs, instruction)
        if hasattr(env, 'env'):
            from robosuite.utils.transform_utils import mat2quat
            controller = env.env.robots[0].controller
            observation['state']['controller_ee_pose'] = np.concatenate(
                (controller.ee_pos, xyzw_to_wxyz(mat2quat(controller.ee_ori_mat))))
        client.call(func_name="update_obs", obs=observation)
        chunk = client.call(func_name="get_action")
        if not chunk:
            raise ValueError("Policy returned an empty action chunk")
        for action in chunk[:min(chunk_steps, max_steps - steps)]:
            mode = action.get('ee_pose_mode', 'delta')
            if mode not in ('delta', 'absolute'):
                raise ValueError(f'Unknown ee_pose_mode: {mode}')
            if mode == 'absolute' or hasattr(env, 'env'):
                for robot in env.env.robots:
                    robot.controller.use_delta = mode == 'delta'
            obs, _, done, _ = env.step(xpl_action_to_libero(action))
            steps += 1
            if record_frame is not None:
                record_frame(obs)
            if done:
                return True, steps
    return False, steps


def run(args: argparse.Namespace) -> dict:
    if args.episodes < 1 or args.action_chunk_steps < 1 or args.wait_steps < 0 or args.max_steps < 0:
        raise ValueError("Episodes/chunk steps must be positive; step limits must be nonnegative")
    if args.episodes > 5:
        raise ValueError("This entry point is a sanity check: use 1–5 episodes")
    from client_server.ws import WsModelClient
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_id < 0 or args.task_id >= suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks}), got {args.task_id}")
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    max_steps = args.max_steps or MAX_STEPS[args.suite]

    client = WsModelClient(
        url=f"ws://{args.host}:{args.port}",
        evaluation_id=f"libero-{args.suite}-task{args.task_id}",
        trial_id="smoke",
        request_timeout_s=args.request_timeout,
    )
    successes = []
    control_steps = []
    videos = []
    video_dir = Path(args.video_dir) if getattr(args, 'video_dir', None) else None
    if video_dir is not None:
        import imageio.v2 as imageio
        video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(args.episodes):
            env = make_env(task, args.seed + episode, args.resolution)
            writer = None
            try:
                env.reset()
                obs = env.set_init_state(initial_states[episode % len(initial_states)])
                client.call(func_name="reset")
                if video_dir is not None:
                    video_path = video_dir / f'{args.suite}_task{args.task_id}_seed{args.seed + episode}_running.mp4'
                    writer = imageio.get_writer(str(video_path), fps=args.video_fps, codec='libx264')
                def record_frame(frame_obs):
                    if writer is not None:
                        # Upright visualization only; model observations stay unchanged.
                        writer.append_data(np.ascontiguousarray(frame_obs['agentview_image'][::-1, ::-1]))
                done, steps = run_episode(
                    env, client, obs, task.language, max_steps=max_steps,
                    wait_steps=args.wait_steps, chunk_steps=args.action_chunk_steps,
                    record_frame=record_frame if writer is not None else None,
                )
                if writer is not None:
                    writer.close()
                    writer = None
                    final_video = video_path.with_name(video_path.name.replace('_running', '_success' if done else '_failure'))
                    video_path.rename(final_video)
                    videos.append(str(final_video))
                control_steps.append(steps)
                successes.append(bool(done))
                print(
                    f"[LIBERO] suite={args.suite} task={args.task_id} "
                    f"episode={episode} success={bool(done)}"
                )
            finally:
                if writer is not None:
                    writer.close()
                env.close()
    finally:
        client.close()

    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": task.language,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "seed": args.seed,
        "control_steps": control_steps,
        "max_steps": max_steps,
        "wait_steps": args.wait_steps,
        "action_chunk_steps": args.action_chunk_steps,
        "resolution": args.resolution,
        "videos": videos,
        "purpose": "integration_smoke_only",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[LIBERO] wrote {output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument(
        "--suite",
        required=True,
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--action-chunk-steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--output", default="outputs/libero_smoke.json")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
