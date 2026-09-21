# Shared benchmark clients / 公共 benchmark client

Each benchmark owns one environment client. Policy-specific preprocessing lives
in `policy/<name>/`; the common client never imports a policy or branches on its name.
All three `setup_eval_env_client.sh` entry points delegate to `benchmarks/run_client.sh`.
The old N1.5 `clients/` modules are compatibility shims.

每个 benchmark 只维护一个 client，负责环境 reset/step、观测字段、控制器映射、
episode 步数和 success 统计。图像旋转/resize、模型 state 打包、归一化、动作解码
由各 policy adapter 负责。离线 LeRobot/HDF5 格式不参与这条在线链路。

```text
LIBERO / RoboCasa365
        ↕ benchmark client (native simulator ↔ common fields)
        ↕ XPolicyLab WebSocket: reset → update_obs → get_action
        ↕ policy adapter (common fields ↔ model-specific tensors)
        ↕ X-VLA / OpenPI JAX π0.5 / GR00T N1.5
```

## Observation and action contract

Version: `data_format_version=evomoe-benchmark-v1`. Camera arrays are native-orientation
RGB HWC; the server owns byte decoding. No policy-specific rotation occurs in the client.

| Field | LIBERO | RoboCasa365 PandaOmron |
|---|---|---|
| `benchmark` | `libero` | `robocasa365` |
| `vision.*.color` | `agentview`, `wrist` | `agentview_left`, `eye_in_hand`, `agentview_right` |
| `state.ee_pose` | observed world xyz + quaternion wxyz | observed relative xyz + quaternion wxyz |
| `state.gripper_qpos` | two observed finger joints | two observed finger joints |
| extra observed state | `controller_ee_pose` when using the simulator | `base_pose` xyz + quaternion wxyz |
| `instruction` | task language | task language |
| action `ee_pose` | xyz + quaternion wxyz | delta xyz + quaternion wxyz |
| action `ee_pose_mode` | `delta` (default) or `absolute` | `delta` only |
| action `ee_joint_state` | closure fraction: 0 open, 1 closed | closure fraction: 0 open, 1 closed |
| extra action fields | none | `base_motion` (4), `control_mode` (1) |

Observed poses and commanded poses have different meanings. LIBERO delta values retain
the simulator controller's normalized command units; absolute positions are world targets.
The client converts quaternion representation back to axis-angle and closure to signed
gripper commands. It does not binarize model predictions. The adapter implements the
checkpoint's gripper threshold if its reference recipe requires one. Absolute control is
enabled only after the initial settling steps; this prevents zero settling actions from
being interpreted as absolute targets at the origin.

RoboCasa365 uses its official gym wrapper for controller mapping/binarization. Its mobile
fields are mandatory; the client never fills missing base actions with invented zeros.
`human`/`mimicgen` describe training data provenance, not separate simulator types.

## Policy-specific LIBERO recipes

| Adapter | Images | State/action mapping |
|---|---|---|
| X-VLA | rotate agentview 180°, wrist unchanged | domain 3; column-stacked rot6d; absolute EEF; previous chunk proprio |
| π0.5 JAX | rotate both 180°, official padded resize to 224 | official 8-D observation / 7-D delta action; continuous gripper |
| N1.5 | rotate both 180°, upstream model transforms | modality keys; per-suite normalization; openness converted/binarized to closure |

Sources: [X-VLA evaluator](https://github.com/2toINF/X-VLA/blob/main/evaluation/libero/libero_client.py),
[OpenPI evaluator](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py),
[OpenPI config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py),
[NVIDIA N1.5 LIBERO example](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/examples/Libero/README.md).

π0.5 uses `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)` and
`extra_delta_transform=False`. The adapter defines this inference config because the
vendored OpenPI registry currently contains ALOHA/Wuji configs only. It rejects PyTorch
weights for this path and reads normalization from the downloaded JAX checkpoint.

## Smoke entry points

Run from the checkout with its root on `PYTHONPATH`, in the relevant simulator environment:

```bash
python -m XPolicyLab.benchmarks.libero.client \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 0 \
  --port 9000 --action-chunk-steps 1 --output /tmp/libero-smoke.json

python -m XPolicyLab.benchmarks.robocasa365.client \
  --task PickPlaceCounterToCabinet --split pretrain --source-kind human \
  --port 9000 --episodes 1 --max-steps 50 --seed 0 --output /tmp/robocasa-smoke.json
```

The LIBERO command is the same for all policy servers. Choose chunk execution explicitly:
N1.5 reference uses 1, OpenPI uses 5, X-VLA consumes the full predicted chunk (30 in this
checkpoint). One-step replanning is useful for interface smoke tests but does not reproduce
each paper's evaluation schedule. The shared default step budgets are 220/280/300/520 for
spatial/object/goal/10; `--max-steps` or `LIBERO_MAX_STEPS` can override them. Outputs record
the budget, wait steps, resolution and chunk length. These bounded 1–3 episode runs do not
constitute benchmark scores.

The existing 10-argument policy `eval.sh` interface remains unchanged. For LIBERO pass
`libero_franka ee`, an absolute checkpoint directory, and your policy/simulator environments.
The shared launcher supports `LIBERO_TASK_ID`, `LIBERO_NUM_EPISODES`,
`LIBERO_ACTION_CHUNK_STEPS`, `LIBERO_MAX_STEPS`, `BENCHMARK_OUTPUT_DIR` and
`LIBERO_OUTPUT_DIR`. RoboCasa requires `ROBOCASA_SPLIT` and `ROBOCASA_SOURCE_KIND`.
Use separate output directories when comparing policies, to avoid overwriting results.

## Assets and validation status

User-approved remote asset root:

```text
/data2/vla-reasoning/proj/XPolicyLab-assets/checkpoints/
├── xvla-libero/
├── pi05-libero/                 # official OpenPI JAX Orbax/OCDBT
└── gr00t_n15/libero/
    ├── libero_spatial/
    ├── libero_object/
    ├── libero_goal/
    └── libero_10/
```

X-VLA's old EvoMoE-new copy was compared file-by-file with SHA256 before removal.
All six checkpoints are present. N1.5 LFS files passed SHA256 verification;
all 16 OpenPI JAX files passed GCS CRC32C verification. Download provenance and
verification manifests live alongside the remote assets; the final record is
`/data2/vla-reasoning/proj/XPolicyLab-assets/checkpoints/libero-assets-status.json`.

已完成代码层面的 client 解耦，并提供三个 policy 的 LIBERO 映射。34 项 CPU 测试已通过，覆盖
不同图像方向、四元数/rot6d、夹爪方向、步数预算和协议边界；真实模型加载、GPU
推理和模拟器闭环尚未执行。RoboCasa365 client 已独立，但三个 policy 的 PandaOmron
模型端适配/checkpoint 尚未就绪，因此目前不能宣称 3×2 组合均已跑通。
现有 LIBERO/双臂 adapter 对 RoboCasa365 请求明确报错。

```bash
python -m pytest benchmarks/tests policy/GR00T_N15/tests -q
```
