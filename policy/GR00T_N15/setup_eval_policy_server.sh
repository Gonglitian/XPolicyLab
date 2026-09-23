#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
YAML_FILE="${SCRIPT_DIR}/deploy.yml"

if [[ "${action_type}" != "ee" ]] || ! {
    [[ "${bench_name,,}" == "libero" && "${env_cfg_type}" == "libero_franka" ]] ||
    [[ "${bench_name,,}" == "robocasa365" && "${env_cfg_type}" == "robocasa_panda_omron" ]];
}; then
    echo "Use LIBERO/libero_franka/ee or RoboCasa365/robocasa_panda_omron/ee." >&2
    exit 2
fi
if [[ ! -f "${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" ]]; then
    echo "Run: python ${SCRIPT_DIR}/setup_workspace.py --workspace ${BENCH_ROOT} --robot ${env_cfg_type}" >&2
    exit 2
fi

resolve_env_root() {
    if [[ "$1" == "uv" ]]; then
        printf '%s\n' "${GR00T_N15_ROOT:-${SCRIPT_DIR}/Isaac-GR00T}"
    else
        python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
    fi
}

if [[ "${policy_env}" == "uv" || "${policy_env}" == */* ]]; then
    env_root=$(resolve_env_root "${policy_env}")
    export GR00T_N15_ROOT="${GR00T_N15_ROOT:-${env_root}}"
    if [[ -x "${env_root}/bin/python" ]]; then
        python_bin="${env_root}/bin/python"
    else
        python_bin="${env_root}/.venv/bin/python"
    fi
    echo "[SERVER] using uv environment: ${env_root}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
    python_bin="$(command -v python)"
fi

if [[ ! -x "${python_bin}" ]]; then
    echo "Python not found: ${python_bin}. Run install.sh first." >&2
    exit 1
fi

export GR00T_N15_ROOT="${GR00T_N15_ROOT:-${SCRIPT_DIR}/Isaac-GR00T}"
export PYTHONPATH="${GR00T_N15_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

denoising_steps=8
if [[ "${bench_name,,}" == "robocasa365" ]]; then
    denoising_steps=4
fi

exec env \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONWARNINGS=ignore::UserWarning \
    "${python_bin}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${YAML_FILE}" \
        --overrides \
            host="${policy_server_host}" \
            port="${policy_server_port}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            libero_suite="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            action_type="${action_type}" \
            denoising_steps="${denoising_steps}" \
            seed="${seed}"
