#!/bin/bash
# Usage: submit.sh METHOD SUITE [--chain-next] [--preflight|--entry-check|--chain-check] [--tasks N]
set -euo pipefail
export V1_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export V1_MACHINE_CONFIG="${V1_MACHINE_CONFIG:-$V1_CODE/labserver_env.sh}"
source "$V1_MACHINE_CONFIG"
v1_prepare_dirs
export PYTHONDONTWRITEBYTECODE=1
exec "$V1_PI_PY" "$V1_CODE/submit_jobs.py" "$@"
