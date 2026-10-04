#!/bin/bash
# Compatibility wrapper: preparation, training and evaluation now run inside Slurm.
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$CODE/submit.sh" "$@"
