# pi0.5 LIBERO v1 baseline (FMN-aligned recipe)

## Import status

This directory contains the unchanged v1 baseline source copied from labserver:

`/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/pi05_libero_v1_fmn/code_from_bcc_20261003/`

The source was supplied as the BCC copy dated 2026-10-03. See
[README_FROM_BCC.md](README_FROM_BCC.md) for its original handoff notes.
`SOURCE_SHA256SUMS` records every imported file for comparison with that source.
The destination branch is `evomoe/hanyun-robocasaLiberoN15`.

This initial import preserves the source training and evaluation behavior.
Machine configuration, a one-GPU/one-stream labserver Slurm entry point,
portable resume paths, checkpoint cleanup and optional trainable-parameter
snapshots are pending follow-up integration. The existing BCC submission script
requests four GPUs and is retained as source material, not the new labserver entry.
All GPU execution on labserver, including policy serving and simulator rendering,
must run inside a Slurm allocation. No GPU validation has been run for this import.

## Source map

- `pipeline.sh`: original preparation and execution order.
- `lane.py`: stream orchestration (training followed by evaluation).
- `train_stage.py`: training one task and carrying optimizer/global-step state.
- `evaluate.py`, `serve.py`: evaluation client and checkpoint policy server.
- `common.py`, `prepare_tasks.py`, `prepare_data.py`: recipe and data preparation.
- `bcc_env.sh`, `bcc_lanes.sbatch`: original BCC configuration and submission.

The loader consumes upstream `physical-intelligence/libero` LeRobot Parquet data
with embedded PNG bytes, rather than XPolicyLab-encoded trajectory image buffers.
Preserve the v1 image orientation and normalization when integrating it.

## Verification of this import

All imported files were checked byte-for-byte against the source. Python files
passed syntax compilation without executing imports; shell and sbatch files
passed `bash -n`. These checks do not establish runtime correctness.
