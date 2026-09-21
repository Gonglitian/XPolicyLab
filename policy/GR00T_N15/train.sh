#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
    exit 1
fi
if [[ "${ALLOW_FULL_GR00T_TRAINING:-0}" != "1" ]]; then
    echo "Full GR00T N1.5 training is gated until senior review." >&2
    echo "After review, rerun with ALLOW_FULL_GR00T_TRAINING=1." >&2
    exit 2
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
if [[ "${bench_name,,}" != "libero" || "${env_cfg_type}" != "libero_franka" || "${action_type}" != "ee" ]]; then
    echo "This training recipe supports LIBERO/libero_franka/ee only." >&2
    exit 2
fi
: "${GR00T_BASE_MODEL_PATH:?Set a local approved LIBERO checkpoint path explicitly}"
if [[ "${seed}" != 42 ]]; then
    echo "Pinned NVIDIA trainer hard-codes seed=42. Use seed 42 to keep run naming truthful." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${GR00T_N15_ROOT:-${SCRIPT_DIR}/Isaac-GR00T}"
dataset="${GR00T_LEROBOT_HOME:?}/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-libero"
output="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
data_config="examples.Libero.custom_data_config:LiberoDataConfig"
if [[ "${ckpt_name,,}" == *goal* ]]; then
    data_config="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
cd "${GR00T_ROOT}"
source .venv/bin/activate
python scripts/gr00t_finetune.py \
    --base-model-path "${GR00T_BASE_MODEL_PATH}" \
    --embodiment-tag new_embodiment \
    --dataset-path "${dataset}" \
    --data_config "${data_config}" \
    --num-gpus "${NUM_GPUS:-1}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --output-dir "${output}" \
    --max-steps "${MAX_STEPS:-20000}" \
    --save-steps "${SAVE_STEPS:-1000}"
