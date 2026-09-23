#!/usr/bin/env bash
# Linux entry point; all paths/GPU IDs are supplied through the deployment config.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${XPL_BOOTSTRAP_PYTHON:-python3}"
exec "${PYTHON}" "${HERE}/deploy.py" run "$@"
