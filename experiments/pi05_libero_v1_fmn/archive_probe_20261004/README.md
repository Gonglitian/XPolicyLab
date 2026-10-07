# 归档：SigLIP 冻结实验代码与旧版运行辅助脚本（2026-10-04）

这里的文件不属于上一级目录的 Slurm 流程，只为留档和复现而保存。

`probe/` 和 `extras/keep_sf_finals.sh` 是针对导入时的旧版代码写的：commit `4ce3685` 里的本目录，
即 UCR BCC 集群运行目录 `code/` 的原样拷贝，8 条 v1 基线流最后都是在 BCC 上用它跑完的。它们依赖旧版的
`gpu_guard.py`、`lane.py` 和 `common.py` 接口，这些在上一级目录里已被删除或改写，所以**在当前版本下
不能直接运行**。要复现，把旧版代码导出到一个单独的目录，再把 `probe/` 里的文件放进去（BCC 上的
`code_probe/` 就是这样得到的）：

```bash
mkdir code_probe
git archive 4ce3685 experiments/pi05_libero_v1_fmn | tar -x --strip-components=2 -C code_probe
cp experiments/pi05_libero_v1_fmn/archive_probe_20261004/probe/* code_probe/
```

评测时 import 的公共部分（`benchmarks/libero/client.py`、`policy/Pi_05/model.py`）当时用的是部署分支
`codex/portable-benchmark-deploy`（`a9331e2`）的版本，与本分支加入本归档目录时的版本逐字节相同；
不要用 `4ce3685` 那个时间点的这两个文件。

脚本里写死的路径都是 BCC 上的，换机器要改 `launch_probe*.sh` 里的 `PROBE_CODE` 和 `bcc_env.sh`。

## probe/：SigLIP 冻结实验

要回答的问题：冻结 π0.5 的图像编码器 SigLIP 之后，模型还能不能从底座 `pi05_base` 学会一个新的
LIBERO 任务。每次运行只训一个任务（从底座冷启动，1 万步，batch 8，其余与 v1 配方相同），训完在该
任务固定的 50 个初始状态上评测。

- 两组：`frozen_siglip`（冻结图像编码器；Gemma 和动作专家的 LoRA、四个小投影层照常训练）和
  `trainable_siglip`（v1 配方原样，作对照）。
- `probe_train.py` 训练，`probe_eval.py` 评测，`probe_lane.py` 在一张卡上按顺序跑若干个
  （组、套件、任务）。
- `launch_probe.sh` / `launch_probe2.sh`：在一个已有的 Slurm 作业里用 `srun --overlap` 启动。
- 2026-10-04 在 BCC 上跑过 8 次：冻结组是 Spatial 的任务 0、2、4 和 Goal 的任务 0、3、5，对照组是
  Spatial 任务 4 和 Goal 任务 3（任务编号从 0 开始）。
- 每次运行的输出在 `<PROBE_ROOT>/<组>/<套件>/taskNN/` 下，成功率在 `evaluated.json`。结果不放在本仓库。

## extras/

- `keep_sf_finals.sh`：旧版 `lane.py` 在开启 `V1_PRUNE_CHECKPOINTS` 时会删掉已用过的检查点。这个脚本在删除前把顺序微调每个任务
  结束时的检查点硬链接到 `kept_sf_finals/`。当前版本的 `retention.py` 已覆盖这个用途。
- `compute_metrics.py`：从成功率矩阵 `matrix.json` 算四个数，只用标准库。`matrix.json` 里的
  `M[i][j]` 是学完任务 i 后任务 j 的成功率。四个数是：learned（每个任务刚学完时成功率的平均）、
  final（学完全部 10 个任务后的平均）、NBT（刚学完时的成功率减去之后各次评测的成功率，取平均）、
  learned−final（只看最后一次评测的下降）。脚本读取与它同目录的 `<sf 或 er>/<套件>/matrix.json`，
  所以要把它复制到结果目录里再运行。
- `bcc_rhel8_deploy.patch`：针对部署分支 `codex/portable-benchmark-deploy`（`a9331e2`）里
  `scripts/deploy/deploy.py` 的补丁，在 RHEL 8（glibc 2.28）上搭 π0.5 环境时需要：跳过没有对应
  安装包的 `rerun-sdk`（只用于可视化），并补装 `pytest==8.3.5`（openpi 在模块顶层 import 它，
  而 `--no-dev` 会把它去掉）。BCC 上的环境就是带着这个补丁搭的。它没有提交到部署分支，因为照现在
  的写法会让所有机器都跳过 `rerun-sdk`。
