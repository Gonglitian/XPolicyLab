#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GR00T_ROOT="${GR00T_N15_ROOT:-${SCRIPT_DIR}/Isaac-GR00T}"
GR00T_COMMIT="4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it before running this script." >&2
    exit 1
fi
if [[ ! -d "${GR00T_ROOT}/.git" ]]; then
    git clone --branch n1.5-release https://github.com/NVIDIA/Isaac-GR00T.git "${GR00T_ROOT}"
    git -C "${GR00T_ROOT}" checkout --detach "${GR00T_COMMIT}"
fi

actual_commit=$(git -C "${GR00T_ROOT}" rev-parse HEAD)
if [[ "${actual_commit}" != "${GR00T_COMMIT}" ]]; then
    echo "[GR00T_N15] Expected commit=${GR00T_COMMIT}, current=${actual_commit}; use a separate pinned checkout." >&2
    exit 2
fi

cd "${GR00T_ROOT}"
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.10
fi
uv pip install --upgrade setuptools
uv pip install -e ".[base]"
if [[ "${SKIP_FLASH_ATTN:-0}" != "1" ]]; then
    uv pip install --no-build-isolation flash-attn==2.7.1.post4
fi
uv pip install -e "${XPL_ROOT}"

echo "[GR00T_N15] installed in ${GR00T_ROOT}/.venv"
echo "[GR00T_N15] install LIBERO separately in the evaluation conda environment."
