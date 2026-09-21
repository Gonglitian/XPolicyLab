#!/usr/bin/env bash
set -euo pipefail

suite=${1:-all}
mode=${2:---dry-run}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${SCRIPT_DIR}/configs/libero_checkpoints.json"
checkpoint_root=${GR00T_N15_CHECKPOINT_ROOT:-}

if [[ "${mode}" != "--dry-run" && "${mode}" != "--download" ]]; then
    echo "Usage: $0 <libero_spatial|libero_object|libero_goal|libero_10|all> [--dry-run|--download]" >&2
    exit 1
fi
if [[ "${mode}" == "--download" && -z "${checkpoint_root}" ]]; then
    echo "Set GR00T_N15_CHECKPOINT_ROOT before downloading." >&2
    exit 1
fi

mapfile -t suites < <(
    python -c 'import json,sys; data=json.load(open(sys.argv[1])); key=sys.argv[2]; print("\n".join(data if key == "all" else [key]))' "${manifest}" "${suite}"
)

for current_suite in "${suites[@]}"; do
    repo_id=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["repo_id"])' "${manifest}" "${current_suite}")
    revision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["revision"])' "${manifest}" "${current_suite}")
    if [[ "${mode}" == "--dry-run" ]]; then
        echo "[GR00T_N15] dry-run ${current_suite}: ${repo_id}"
        hf download "${repo_id}" --revision "${revision}" --dry-run
    else
        destination="${checkpoint_root}/${current_suite}"
        echo "[GR00T_N15] downloading ${repo_id} -> ${destination}"
        hf download "${repo_id}" --revision "${revision}" --local-dir "${destination}"
    fi
done
