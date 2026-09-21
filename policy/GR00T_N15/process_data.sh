#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <LIBERO|RoboCasa365> <dataset_name> <env_cfg_type> <action_type> [human|mimicgen]" >&2
    exit 1
fi
bench_name=$1
dataset_name=$2
env_cfg_type=$3
action_type=$4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GR00T_LEROBOT_HOME:?Set output dataset root}"
case "${bench_name,,}" in
    libero)
        [[ "${env_cfg_type}" == libero_franka && "${action_type}" == ee ]] || exit 2
        source_dir="${LIBERO_LEROBOT_SOURCE:?Set prepared LeRobot source}"
        modality="${GR00T_N15_ROOT:-${SCRIPT_DIR}/Isaac-GR00T}/examples/Libero/modality.json"
        source_kind=libero
        ;;
    robocasa365)
        [[ "${env_cfg_type}" == panda_omron && "${action_type}" == ee ]] || exit 2
        source_kind="${5:?Specify human or mimicgen explicitly}"
        [[ "${source_kind}" == human || "${source_kind}" == mimicgen ]] || exit 2
        source_dir="${ROBOCASA365_LEROBOT_SOURCE:?Set prepared LeRobot v2 source}"
        modality="${ROBOCASA365_MODALITY_PATH:-${source_dir}/meta/modality.json}"
        ;;
    *) echo "Unsupported benchmark ${bench_name}" >&2; exit 2 ;;
esac
target="${GR00T_LEROBOT_HOME}/${bench_name}-${dataset_name}-${env_cfg_type}-${action_type}-${source_kind}"
exec python "${SCRIPT_DIR}/prepare_dataset.py" --benchmark "${bench_name,,}" \
    --source "${source_dir}" --target "${target}" --modality "${modality}" --source-kind "${source_kind}"
