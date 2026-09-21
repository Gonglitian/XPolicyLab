"""Compatibility entry point. Implementation lives in the shared benchmark package."""
from XPolicyLab.benchmarks.libero.client import *  # noqa: F401,F403

if __name__ == '__main__':
    run(parse_args())
