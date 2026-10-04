#!/bin/bash
# Usage: submit.sh sf|er libero_spatial|libero_object|libero_goal|libero_10 [lane options]
set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 sf|er SUITE [--preflight|--entry-check] [--tasks N]" >&2; exit 2
fi
method=$1; suite=$2; shift 2
case "$method" in sf|er) ;; *) echo "Unknown method: $method" >&2; exit 2;; esac
case "$suite" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo "Unknown suite: $suite" >&2; exit 2;; esac
# Reject invalid options before consuming a Slurm allocation.
mode=normal
args=("$@")
while [ "$#" -gt 0 ]; do
    case "$1" in
        --preflight|--entry-check)
            [ "$mode" = normal ] || { echo "Choose one run mode" >&2; exit 2; }
            mode=$1; shift;;
        --tasks)
            [ "$#" -ge 2 ] || { echo "--tasks needs an integer" >&2; exit 2; }
            case "$2" in 1|2|3|4|5|6|7|8|9|10) ;; *) echo "--tasks must be 1..10" >&2; exit 2;; esac
            shift 2;;
        *) echo "Unknown option: $1" >&2; exit 2;;
    esac
done
if [ "$mode" = --preflight ] && [ "$method" != er ]; then
    echo "--preflight requires method er" >&2; exit 2
fi
set -- "${args[@]}"
export V1_CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export V1_MACHINE_CONFIG="${V1_MACHINE_CONFIG:-$V1_CODE/labserver_env.sh}"
source "$V1_MACHINE_CONFIG"
v1_prepare_dirs
mkdir -p "$V1_RUN/slurm"
exec sbatch --parsable --partition="${V1_SLURM_PARTITION:-gpu}" \
    --time="${V1_SLURM_TIME:-3-00:00:00}" --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task="${V1_SLURM_CPUS:-12}" --mem="${V1_SLURM_MEM:-60G}" \
    --job-name="v1_${method}_${suite}" --output="$V1_RUN/slurm/%x-%j.log" \
    --chdir="$V1_CODE" --export=ALL "$V1_CODE/labserver.sbatch" \
    --method "$method" --suite "$suite" "$@"
