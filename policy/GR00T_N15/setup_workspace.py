"""Register the LIBERO runtime robot without replacing unrelated workspace config."""
import argparse
import json
from pathlib import Path


def register(workspace: Path) -> None:
    config_root = workspace / "env_cfg"
    registry_path = config_root / "robot" / "_robot_info.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    robot = {"arm_dim": [7], "ee_dim": [1]}
    if "libero_franka" in registry and registry["libero_franka"] != robot:
        raise ValueError(f"Conflicting libero_franka entry in {registry_path}")
    config = config_root / "libero_franka.yml"
    expected = {"config": {"robot": "libero_franka"}, "observation": {"collect_freq": 20}}
    if config.exists():
        import yaml
        if yaml.safe_load(config.read_text()) != expected:
            raise ValueError(f"Existing config differs; review it manually: {config}")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if "libero_franka" not in registry:
        registry["libero_franka"] = robot
        temporary = registry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n")
        temporary.replace(registry_path)
    if not config.exists():
        config.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"Registered LIBERO runtime configuration in {config_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    register(parser.parse_args().workspace.resolve())
