#!/usr/bin/env bash
# Linux entry point: Conda environments, pinned sources, checkpoints and datasets.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${XPL_BOOTSTRAP_PYTHON:-python3}"
exec "${PYTHON}" "${HERE}/deploy.py" build "$@"
