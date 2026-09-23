# EvoMoE / XPolicyLab 适配交接

目标分支：`evomoe/hanyun-robocasaLiberoN15`；团队仓库：`Gonglitian/XPolicyLab`。

本阶段交付代码和 CPU 检查。真实 checkpoint 加载、单帧推理、每个 LIBERO suite 一个任务的 1–3 episode smoke test，仍需用户确认后执行。全量评测、全量转换和正式训练留待师兄 review。

## 先读哪些文件

1. 根目录 README 的 Standard Data Formats：理解 RGB、state、action 约定。
2. `policy/demo_policy/model.py`：看 `update_obs → get_action → reset` 接口。
3. 本目录 `model.py`：看 XPL 观测怎样映射成 N1.5 的 video/state/language。
4. `../../benchmarks/libero/client.py`：看 benchmark 的观测和动作如何接入同一个接口。
5. `prepare_dataset.py`：数据集适配与在线 benchmark 适配是两条不同的链路。

## 当前交付与限制

- LIBERO 四套 suite 的模型 adapter、checkpoint manifest、下载 dry-run、启动脚本、单任务 smoke client。
- Goal 自动使用官方 MeanStd action normalization，其余 suite 使用 MinMax。
- RoboCasa365 独立 gym client，保留 16 维状态、12 维动作中的底座和模式字段。human/MimicGen 是数据来源标签；human 在这里是仿真人工示范。
- LIBERO 与 RoboCasa365 的 LeRobot v2 metadata 校验与轻量视图；只复制 meta，data/videos 保持软链接。v3 不会被冒充成 v2。
- RoboCasa365 尚无已接入的 PandaOmron N1.5 checkpoint。LIBERO 的 7 维动作模型不能直接控制其 12 维移动机械臂；不能据此宣称 RoboCasa365 模型闭环已跑通。
- 找到了官方预构建数据，因此这次未实现新的 raw-HDF5 streaming converter，也没有全量转换。若后续选定数据无法使用，再实现按 episode 写入最终 LeRobot 的 fallback。

## 服务器位置与现场核对

```text
/data2/vla-reasoning/proj/XPolicyLab
/data2/vla-reasoning/proj/XPolicyLab-upstreams/Isaac-GR00T-n15
/data2/vla-reasoning/proj/XPolicyLab-assets/datasets/libero
/data2/vla-reasoning/proj/env_cfg/libero_franka.yml
```

接续时，远程 6 个旧测试通过；新增回归后 17 个测试通过。服务器已补齐运行时 robot registry，仓库内部维度表也保留对应注册。测试运行在已有 base Python；没有安装模型依赖。

只读抽查的真实 LIBERO HDF5 含两路 128×128 RGB、6 维 EEF pose 与两维 gripper state、7 维 action。原任务已检查 130 个文件下载完成；本轮抽查不等于 130 个文件逐帧验收。旧 EvoMoE 的 RoboCasa 数据软链接目标不可读，不能算作数据就绪。

## Checkpoint 下载完成；模型运行仍待确认

用户确认后的六份 checkpoint 已就位：X-VLA 复用并迁移后删除旧副本；π0.5 使用
OpenPI 官方 JAX Orbax/OCDBT（16 个文件通过 CRC32C）；N1.5 四套 LIBERO 权重
通过 SHA256。统一目录为 `/data2/vla-reasoning/proj/XPolicyLab-assets/checkpoints/`，
最终状态清单为该目录下的 `libero-assets-status.json`。公共接口与 adapter 合计
34 项 CPU 回归通过，尚未加载真实模型或运行 GPU/模拟器。

四套 NVIDIA 官方 N1.5 示例引用的 `youliangtan` checkpoint 已记录到 `configs/libero_checkpoints.json`，固定各自 revision。记录的仓库存储量合计约 30.34 GB；权重 HF metadata 未声明许可证，不能用代码的 Apache-2.0 替代权重许可说明。

```bash
cd /data2/vla-reasoning/proj/XPolicyLab/policy/GR00T_N15
export GR00T_N15_ROOT=/data2/vla-reasoning/proj/XPolicyLab-upstreams/Isaac-GR00T-n15
export GR00T_N15_CHECKPOINT_ROOT=/data2/vla-reasoning/proj/XPolicyLab-assets/checkpoints/gr00t_n15/libero

# 先预览下载清单；此命令不下载权重。
bash download_checkpoints.sh all --dry-run

# 已获用户确认的下载命令：
bash download_checkpoints.sh all --download

# 独立模型环境，不改变 base 环境；模拟器另用 libero 环境。
bash install.sh

# 先无模拟器检验，然后检验服务端图像解码。
EVAL_ENV_TYPE=debug bash eval.sh LIBERO libero_spatial "$GR00T_N15_CHECKPOINT_ROOT/libero_spatial" libero_franka ee 0 0 1 uv libero
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh LIBERO libero_spatial "$GR00T_N15_CHECKPOINT_ROOT/libero_spatial" libero_franka ee 0 0 1 uv libero

# 最后一个 suite 一个任务、一个 episode。四套 suite 各用自己的 checkpoint。
LIBERO_TASK_ID=0 LIBERO_NUM_EPISODES=1 bash eval.sh LIBERO libero_spatial "$GR00T_N15_CHECKPOINT_ROOT/libero_spatial" libero_franka ee 0 0 1 uv libero
```

GPU 编号需要运行前核对空闲情况；`libero` conda 环境需要按官方 LIBERO/robosuite 依赖准备。现有服务器未验证这两个运行环境，以上命令是待执行流程，不是运行成功记录。先在一个 suite 验证加载与接口，再逐个做其余 suite 的小规模闭环。

可学习的三个命令：`git diff --check` 检查补丁空白错误；`bash -n script.sh` 只检查脚本语法；`pytest` 运行无需模型权重的回归测试。检查结果与真实 GPU 推理是不同层次的证据。

## 公共 benchmark 接口

LIBERO 与 RoboCasa365 client 已移至根目录 `benchmarks/`，三个 policy 的
环境启动入口调用同一个脚本。N1.5 旧 `clients/` 路径只做兼容转发。
公共 client 保留原始 RGB 方向，姿态四元数统一为 wxyz，夹爪动作统一为
0=open、1=closed。图像旋转、模型 state 打包、归一化和动作解码归各 policy。
LIBERO 的绝对/增量控制由 action 的 `ee_pose_mode` 明确表达，不按 policy 名称分支。

X-VLA 和 π0.5 已补 LIBERO adapter；π0.5 仅接官方 JAX checkpoint。
此处代码/CPU 合同测试不代表真实权重加载或仿真闭环通过。RoboCasa365 的公共
client 可以复用，但三个 policy 的 PandaOmron 模型端适配与 checkpoint 仍待补齐。
详细字段、命令和来源见 `benchmarks/README.md`。
