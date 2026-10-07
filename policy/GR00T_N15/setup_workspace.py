"""Register a supported runtime robot without replacing unrelated workspace config."""
import argparse
import json
from pathlib import Path


def register(workspace: Path, robot_name: str = 'libero_franka') -> None:
    if robot_name not in {'libero_franka', 'robocasa_panda_omron'}:
        raise ValueError(f'Unsupported runtime robot: {robot_name}')
    config_root = workspace / "env_cfg"
    registry_path = config_root / "robot" / "_robot_info.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    robot = {"arm_dim": [7], "ee_dim": [1]}
    if robot_name in registry and registry[robot_name] != robot:
        raise ValueError(f"Conflicting {robot_name} entry in {registry_path}")
    config = config_root / f"{robot_name}.yml"
    expected = {"config": {"robot": robot_name}, "observation": {"collect_freq": 20}}
    if config.exists():
        import yaml
        if yaml.safe_load(config.read_text()) != expected:
            raise ValueError(f"Existing config differs; review it manually: {config}")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if robot_name not in registry:
        registry[robot_name] = robot
        temporary = registry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n")
        temporary.replace(registry_path)
    if not config.exists():
        config.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Registered {robot_name} runtime configuration in {config_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument('--robot', choices=('libero_franka', 'robocasa_panda_omron'), default='libero_franka')
    args = parser.parse_args()
    register(args.workspace.resolve(), args.robot)
