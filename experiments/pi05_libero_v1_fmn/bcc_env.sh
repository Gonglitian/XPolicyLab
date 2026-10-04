# Paths supplied by Litian in code_from_bcc_20261003/bcc_env.sh.
# BCC access/runtime validation is owned by Litian. Source in a fresh shell.
export V1_BCC_HOME="${V1_BCC_HOME:-/data/JiachenLi/litian}"
export V1_ROOT="${V1_ROOT:-$V1_BCC_HOME/xpl}"
export V1_ASSETS="${V1_ASSETS:-$V1_BCC_HOME/xpl-assets}"
# Use a new output directory; do not alter the original BCC baseline run.
export V1_RUN="${V1_RUN:-$V1_BCC_HOME/xpl-runs/pi05_libero_v1_fmn_portable}"
export V1_OPENPI="${V1_OPENPI:-$V1_ASSETS/upstreams/openpi}"
export V1_PI_PY="${V1_PI_PY:-$V1_ASSETS/envs/pi05/bin/python}"
export V1_SIM_PY="${V1_SIM_PY:-$V1_ASSETS/envs/libero/bin/python}"
export V1_PI_PATHS="${V1_PI_PATHS:-$V1_OPENPI/src:$V1_ASSETS/upstreams/robocasa:$V1_ASSETS/upstreams/robosuite}"
export V1_SIM_PATHS="${V1_SIM_PATHS:-$V1_ASSETS/upstreams/libero}"
export V1_LIBERO_CONFIG="${V1_LIBERO_CONFIG:-$V1_ASSETS/config/libero}"
# Caches default to the assets filesystem rather than the root filesystem.
source "$(dirname "${BASH_SOURCE[0]}")/env_defaults.sh"
