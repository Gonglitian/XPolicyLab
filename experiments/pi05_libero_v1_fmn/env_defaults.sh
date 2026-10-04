# Shared configuration only; sourcing this file launches nothing and creates no directories.
export V1_REPO="${V1_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export V1_DATA="${V1_DATA:-$V1_ASSETS/datasets/libero_pi_lerobot}"
export V1_BASE="${V1_BASE:-$V1_ASSETS/checkpoints/cl_base/pi05_base}"
export V1_WS_PATHS="${V1_WS_PATHS-$V1_ASSETS/envs/xvla-ws-deps}"
export V1_CACHE="${V1_CACHE:-$V1_ASSETS/cache/pi05_libero_v1_fmn}"
export V1_JAX_CACHE="${V1_JAX_CACHE:-$V1_CACHE/jax}"
export TMPDIR="${V1_TMPDIR:-$V1_CACHE/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_HOME="${V1_HF_HOME:-$V1_CACHE/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE" TRANSFORMERS_CACHE="$HF_HOME/transformers"
export OPENPI_DATA_HOME="${V1_OPENPI_DATA_HOME:-$V1_CACHE/openpi}"
export XDG_CACHE_HOME="$V1_CACHE/xdg" TORCH_HOME="$V1_CACHE/torch"
export TRITON_CACHE_DIR="$V1_CACHE/triton" CUDA_CACHE_PATH="$V1_CACHE/cuda"
export JAX_COMPILATION_CACHE_DIR="$V1_JAX_CACHE"
export V1_DOWNLOAD_LOG="${V1_DOWNLOAD_LOG:-$V1_ASSETS/datasets/dl_libero256.log}"
v1_prepare_dirs() {
    mkdir -p "$V1_RUN" "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
        "$OPENPI_DATA_HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
        "$CUDA_CACHE_PATH" "$V1_JAX_CACHE" "$TRANSFORMERS_CACHE"
}
