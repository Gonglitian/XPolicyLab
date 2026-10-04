# Source in a fresh shell. All machine-specific paths are configurable via V1_*.
export V1_ROOT="${V1_ROOT:-/data2/vla-reasoning/proj}"
export V1_ASSETS="${V1_ASSETS:-$V1_ROOT/XPolicyLab-assets}"
export V1_STORAGE_ROOT=/data2
# Separate output from the retained baseline archive.
export V1_RUN="${V1_RUN:-$V1_ASSETS/baselines/pi05_libero_v1_fmn_portable}"
export V1_OPENPI="${V1_OPENPI:-$V1_ROOT/XPolicyLab-upstreams/openpi-robocasa}"
export V1_PI_PY="${V1_PI_PY:-$V1_ROOT/XPolicyLab-envs/pi05-robocasa/bin/python}"
export V1_SIM_PY="${V1_SIM_PY:-$V1_ASSETS/envs/xvla-sanity/bin/python}"
export V1_PI_PATHS="${V1_PI_PATHS:-$V1_OPENPI/src:/data1/vla-reasoning/proj/EvoMoE/eval/robocasa/deps/robocasa:/data1/vla-reasoning/proj/EvoMoE/eval/robocasa/deps/robosuite}"
export V1_SIM_PATHS="${V1_SIM_PATHS:-/home/vla-reasoning/proj/autofocus_3d/baselines/libero}"
export V1_LIBERO_CONFIG="${V1_LIBERO_CONFIG:-$V1_ASSETS/sanity_checks/20260922/xvla_libero/libero_config}"
export V1_CONDA_EXE="${V1_CONDA_EXE:-/home/vla-reasoning/miniconda3/bin/conda}"
export V1_CONDA_ENV="${V1_CONDA_ENV-base}"
source "$(dirname "${BASH_SOURCE[0]}")/env_defaults.sh"
