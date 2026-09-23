# GR00T_N15

**Contributor:** EvoMoE Team | **Paper:** GR00T N1.5 | **arXiv:** Not specified | **Original code:** https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release

`GR00T_N15` adapts NVIDIA GR00T N1.5 LIBERO policies to XPolicyLab. This is a review-stage integration, not a measured baseline. It also supplies a separate RoboCasa365 benchmark client and prepared-data validation. LIBERO checkpoints do not support RoboCasa365 mobile actions. The adapter uses external NVIDIA source code rather than vendoring it.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The inspected NVIDIA source revision is `4af2b622892f7dcb5aae5a3fb70bcb02dc217b96` from `n1.5-release`; real checkpoint loading has not yet been tested. An existing checkout may be selected with `GR00T_N15_ROOT`; otherwise `install.sh` clones into `Isaac-GR00T/` and pins that commit. Existing mismatching checkouts are rejected without alteration.

```bash
cd XPolicyLab/policy/GR00T_N15
export GR00T_N15_ROOT=/path/to/Isaac-GR00T-n15  # optional
bash install.sh
python setup_workspace.py --workspace /path/to/parent-of-XPolicyLab
```

Install LIBERO in a separate evaluation conda environment following its official instructions. Policy dependencies and simulator dependencies intentionally remain separate.

## Model Assets

NVIDIA's `n1.5-release/examples/Libero/README.md` documents four suite-specific checkpoints:

| Suite | Hugging Face repository | N1.5 data config |
| --- | --- | --- |
| `libero_spatial` | `youliangtan/gr00t-n1.5-libero-spatial-posttrain` | `LiberoDataConfig` |
| `libero_object` | `youliangtan/gr00t-n1.5-libero-object-posttrain` | `LiberoDataConfig` |
| `libero_goal` | `youliangtan/gr00t-n1.5-libero-goal-posttrain` | `LiberoDataConfigMeanStd` |
| `libero_10` | `youliangtan/gr00t-n1.5-libero-long-posttrain` | `LiberoDataConfig` |

The recorded Hugging Face `usedStorage` is 7,586,072,344 bytes per repository
(about 30.34 GB for four). This is repository storage metadata, not a guarantee
of download transfer size; use the pinned dry-run to inspect planned files.
The manifest pins the inspected Hugging Face revision of every suite. The model repositories do not
declare a weight license in their Hugging Face metadata; NVIDIA's source code
is Apache-2.0, but that does not by itself license the weights. Dry-run is the
default and downloads no weights:

```bash
bash download_checkpoints.sh all --dry-run
```

After explicit approval, keep large assets outside Git:

```bash
export GR00T_N15_CHECKPOINT_ROOT=/data2/<user>/proj/XPolicyLab-assets/checkpoints/gr00t_n15/libero
bash download_checkpoints.sh all --download
```

Pass a downloaded suite directory as `ckpt_name`, or set `model_path` in `deploy.yml`. A valid directory contains `config.json`, `model.safetensors.index.json`, model shards, and `experiment_cfg/metadata.json`.

## Data Processing

N1.5 consumes LeRobot v2.0/v2.1 plus `meta/modality.json`. `configs/data_sources.json` records sources. Prefer the prepared IPEC LIBERO exports named in NVIDIA's official example and RoboCasa's own LeRobot exports. These do **not** use XPolicyLab's default `cam_high/cam_left_wrist/cam_right_wrist` converter keys: LIBERO uses `image/wrist_image`, and RoboCasa365 uses `robot0_agentview_left/right/eye_in_hand` with a 12-D mobile action. This is why the adapter consumes benchmark-native prepared exports instead of the generic RoboDojo converter.

`process_data.sh` validates version, vector dimensions, modality coverage, camera keys, and required metadata before creating a view. Large `data/`, `videos/`, and `images/` directories are symlinked; `meta/` is copied. Source data remain unchanged. Existing output directories are never overwritten. This checks metadata compatibility, not every video frame or source label; a one-episode data-loader check remains necessary before training.

```bash
export GR00T_LEROBOT_HOME=/data2/<user>/proj/XPolicyLab-assets/lerobot
export LIBERO_LEROBOT_SOURCE=/path/to/libero_spatial_lerobot
bash process_data.sh LIBERO libero_spatial libero_franka ee
```

Output: `$GR00T_LEROBOT_HOME/LIBERO-libero_spatial-libero_franka-ee-libero`.

Raw LIBERO HDF5 is not converted in this stage. Official prepared sources have been found, so a bespoke streaming HDF5 converter is not implemented. If a chosen source later fails validation, an episode-wise converter can be added without persisting a second canonical-HDF5 dataset. LeRobot v3 is explicitly rejected rather than relabeled as v2.

RoboCasa365 has separate `human` and `mimicgen` source labels. Human here means human demonstrations in simulation, not physical-robot recordings. Both are available through the [official registry/downloader](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md). Preserve the source's `meta/modality.json`; dimensions alone cannot establish action order.

```bash
export ROBOCASA365_LEROBOT_SOURCE=/path/to/official/task/lerobot
bash process_data.sh RoboCasa365 task_name panda_omron ee human
# Select a separate MimicGen source before running:
bash process_data.sh RoboCasa365 task_name panda_omron ee mimicgen
```

Each output suffix records its source kind. The 16-D state includes base position/quaternion, relative EEF position/quaternion, and both gripper joints. Missing base state raises an error. The official dataset action order is `base(4), control_mode(1), EEF translation(3), EEF rotation(3), gripper(1)`. Raw robosuite order differs; the benchmark client uses the official gym action dictionary so the official wrapper performs controller mapping and gripper/mode binarization.

## Training

An unexecuted LIBERO training recipe is included, blocked by default. It requires senior review, a prepared dataset, and an explicit local base checkpoint to prevent an implicit base-model download. The pinned upstream trainer hard-codes seed 42; this entry rejects other seeds rather than mislabeling output folders:

```bash
ALLOW_FULL_GR00T_TRAINING=1 \
GR00T_BASE_MODEL_PATH=/path/to/approved/libero_checkpoint \
GR00T_LEROBOT_HOME=/path/to/lerobot \
bash train.sh LIBERO libero_spatial libero_franka ee 42 0
```

Do not set `ALLOW_FULL_GR00T_TRAINING=1` during stage 2. Goal training automatically selects the required mean/std action normalization; the other suites use min/max.

## Evaluation

`task_name` is the LIBERO suite. Stage-2 smoke tests use task id 0, 1–3 episodes, and a fixed seed; they validate the integration and are not benchmark results.

```bash
cd XPolicyLab/policy/GR00T_N15
export GR00T_N15_ROOT=/path/to/Isaac-GR00T-n15
export LIBERO_TASK_ID=0
export LIBERO_NUM_EPISODES=1

bash eval.sh LIBERO libero_spatial /path/to/checkpoints/libero_spatial \
  libero_franka ee 0 0 1 uv libero
```

Repeat with `libero_object`, `libero_goal`, and `libero_10`, always pairing the suite with its own checkpoint. Images remain RGB throughout. The policy adapter applies NVIDIA's documented 180-degree spatial rotation and gripper binarization. The shared client preserves native RGB camera orientation and does no checkpoint-specific preprocessing.

For synthetic-observation debugging, run the same command with `EVAL_ENV_TYPE=debug`, then with `EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1`. These use synthetic observations through the real model server and do not start LIBERO; they still execute the real checkpoint. The environment client counts simulator steps correctly even with multiple actions per prediction.

The adapter also supports the official RoboCasa365 Human300 PandaOmron checkpoint.
Use the `robocasa-benchmark/Isaac-GR00T` fork (validated source revision
`9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10`) and the checkpoint directory
`robocasa/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000`.
Keep this source separate from NVIDIA's LIBERO fork. Register its runtime robot:

```bash
python setup_workspace.py --workspace <parent-workspace> --robot robocasa_panda_omron
```

Set `bench_name=RoboCasa365`, `env_cfg_type=robocasa_panda_omron`, `action_type=ee`,
`gr00t_root=<robocasa-fork>`, `model_path=<checkpoint-directory>`, and
`denoising_steps=4`. The model uses the fork's `PandaOmronDataConfig` and checkpoint
normalization metadata. With that policy server running:

```bash
python -m XPolicyLab.benchmarks.robocasa365.client \
  --task OpenDrawer --split pretrain --source-kind human \
  --port 9000 --episodes 5 --action-steps 16 --seed 0 \
  --output /tmp/robocasa_sanity.json --video-dir /tmp/robocasa_videos
```

It expects XPL `ee_pose` (delta pose, quaternion wxyz), `ee_joint_state` (one value in [0,1]), plus `base_motion` (4) and `control_mode` (1). These two mobile fields are a documented benchmark extension. Its observation includes three RGB cameras and named `base_pose`, `ee_pose`, and `gripper_qpos` fields (pose quaternions wxyz). Model-specific packing belongs to each policy adapter. `source-kind` labels the training data; it does not create a different simulator. Truncation is not counted as success.

The default step budget comes from the installed RoboCasa task registry (750 for
OpenDrawer in RoboCasa 1.0.1). The client executes up to 16 predicted actions per
policy call, stops on benchmark-reported success, and writes per-episode seeds,
instructions, step counts, success flags, timing, and three-camera RGB MP4 paths.
JSON results are saved after each completed episode. This single-task check does
not reproduce aggregate benchmark scores.

## Notes

- Implemented: LIBERO `spatial/object/goal/10`, `env_cfg_type=libero_franka`, `action_type=ee`; sequential batch policy calls preserve active environment IDs. The simulator client is single-environment.
- Goal uses `LiberoDataConfigMeanStd`; using the default config silently changes action de-normalization and is invalid.
- `LIBERO_ACTION_CHUNK_STEPS=1` is the smoke-test default and matches NVIDIA's reference evaluator, which replans after the first predicted action.
- RoboCasa365 uses its separate official fork and Human300 checkpoint. NVIDIA's `RoboCasa` GR1 tabletop example is not a PandaOmron substitute.
- RoboCasa runtime validation on 2026-09-22: official Human300 checkpoint loaded and evaluated through the shared XPolicyLab client on OpenDrawer/pretrain, seeds 0–4, horizon 750, chunk 16, denoising 4. Success was 3/5 (seeds 0, 2, 4); all five MP4s and per-episode JSON were saved. This is a single-task sanity check, not an aggregate benchmark score. Full conversion and training remain untested.

## Validation

From the repository root:

```bash
git diff --check
for f in policy/GR00T_N15/*.sh; do bash -n "$f"; done
python -m compileall -q policy/GR00T_N15
python -m pytest policy/GR00T_N15/tests benchmarks/tests -q
```

The adapter and shared benchmark CPU tests cover action/observation mappings, Goal normalization selection, checkpoint shard validation, environment-root precedence, batch alignment, rollout budget, RoboCasa truncation, source-preserving dataset views, version rejection, and conflict-safe robot registration. These are contract/regression tests, not a real model or simulator test. See [REVIEW_zh.md](REVIEW_zh.md) for the handoff and pending run commands.

The canonical clients now live in `benchmarks/libero` and `benchmarks/robocasa365`; `clients/` contains compatibility shims only. All three policy launchers delegate to the same `benchmarks/run_client.sh`. See [the shared contract](../../benchmarks/README.md).
