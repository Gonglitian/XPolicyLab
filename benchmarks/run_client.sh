#!/usr/bin/env bash
# Same 11 arguments as policy/*/setup_eval_env_client.sh.
set -euo pipefail
bench_name=$1
task_name=$2
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
policy_server_port=${10}
policy_server_ip=${11:-localhost}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ "${action_type}" != ee ]]; then
    echo 'Shared LIBERO/RoboCasa365 clients require action_type=ee' >&2
    exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${eval_env_conda_env}"
export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONPATH="${XPL_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
output_dir="${BENCHMARK_OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
mkdir -p "${output_dir}"
video_args=()
if [[ -n "${BENCHMARK_VIDEO_DIR:-}" ]]; then
    video_args=(--video-dir "${BENCHMARK_VIDEO_DIR}")
fi
case "${bench_name,,}" in
    libero)
        if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
            exec python -m XPolicyLab.benchmarks.libero.debug --host "${policy_server_ip}" --port "${policy_server_port}"
        fi
        [[ "${EVAL_ENV_TYPE:-sim}" == sim ]] || { echo 'EVAL_ENV_TYPE must be sim or debug' >&2; exit 2; }
        output_dir="${LIBERO_OUTPUT_DIR:-${output_dir}}"
        exec python -m XPolicyLab.benchmarks.libero.client \
            --host "${policy_server_ip}" --port "${policy_server_port}" \
            --suite "${task_name}" --task-id "${LIBERO_TASK_ID:-0}" \
            --episodes "${LIBERO_NUM_EPISODES:-1}" --seed "${seed}" \
            --max-steps "${LIBERO_MAX_STEPS:-0}" \
            --action-chunk-steps "${LIBERO_ACTION_CHUNK_STEPS:-1}" \
            "${video_args[@]}" \
            --output "${output_dir}/${task_name}_task${LIBERO_TASK_ID:-0}_seed${seed}.json"
        ;;
    robocasa365|robocasa)
        [[ "${EVAL_ENV_TYPE:-sim}" == sim ]] || { echo 'RoboCasa365 supports EVAL_ENV_TYPE=sim' >&2; exit 2; }
        horizon_args=()
        if [[ -n "${ROBOCASA_MAX_STEPS:-}" ]]; then
            horizon_args=(--max-steps "${ROBOCASA_MAX_STEPS}")
        fi
        exec python -m XPolicyLab.benchmarks.robocasa365.client \
            --host "${policy_server_ip}" --port "${policy_server_port}" \
            --task "${task_name}" --split "${ROBOCASA_SPLIT:?Set ROBOCASA_SPLIT=pretrain or target}" \
            --source-kind "${ROBOCASA_SOURCE_KIND:?Set ROBOCASA_SOURCE_KIND=human or mimicgen}" \
            --episodes "${ROBOCASA_NUM_EPISODES:-1}" --seed "${seed}" \
            --action-steps "${ROBOCASA_ACTION_CHUNK_STEPS:-16}" \
            "${horizon_args[@]}" "${video_args[@]}" \
            --output "${output_dir}/${task_name}_seed${seed}.json"
        ;;
    *) echo "Unsupported benchmark: ${bench_name}" >&2; exit 2 ;;
esac
