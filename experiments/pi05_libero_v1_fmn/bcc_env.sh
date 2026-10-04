# Path overrides that point the v1 code at the BCC deployment (built by the deploy branch's build_env.sh).
B=/data/JiachenLi/litian
A=$B/xpl-assets
export V1_ROOT=$B/xpl V1_ASSETS=$A V1_RUN=$B/xpl-runs/pi05_libero_v1_fmn_bcc
export V1_DATA=$A/datasets/libero_pi_lerobot V1_BASE=$A/checkpoints/cl_base/pi05_base V1_OPENPI=$A/upstreams/openpi
export V1_PI_PY=$A/envs/pi05/bin/python V1_SIM_PY=$A/envs/libero/bin/python
export V1_PI_PATHS=$A/upstreams/openpi/src:$A/upstreams/robocasa:$A/upstreams/robosuite
export V1_SIM_PATHS=$A/upstreams/libero V1_LIBERO_CONFIG=$A/config/libero
export V1_JAX_CACHE=/scratch/$USER/jax_cache V1_PRUNE_CHECKPOINTS=1
export HF_HOME=$A/cache/huggingface OPENPI_DATA_HOME=$A/cache/openpi
mkdir -p $V1_RUN /scratch/$USER/jax_cache
# Streams for the follow-up Slurm job (its frozen script reads V1_STREAMS after sourcing this file).
# 4 entries, one per GPU; each may be a comma chain. Finished stages are skipped on restart.
export V1_STREAMS="er:libero_goal,sf:libero_object er:libero_object er:libero_spatial sf:libero_spatial"
