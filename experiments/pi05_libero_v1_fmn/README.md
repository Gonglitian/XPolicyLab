# pi0.5 LIBERO v1 baseline

本目录从 labserver 的
`XPolicyLab-assets/baselines/pi05_libero_v1_fmn/code/` 原样导入（`eacea21`），
随后进行机器路径配置与迁移支持改造。机器配置参考 Litian 于 2026-10-03
提供的 `code_from_bcc_20261003/bcc_env.sh`；没有引入该副本的四卡调度或清理逻辑。
训练配方、图像方向、优化器、学习率和评测次数保持 v1 设置。

## 当前范围

已完成代码入库、labserver/BCC 配置和可迁移路径。单卡单流 Slurm 入口、
GPU guard 移除、动态端口、跨作业 GPU 验证与快照/清理仍是后续工作。
所有训练、模型服务及仿真渲染必须在 Slurm 作业内运行。
`pipeline.sh` 和 `gate_then_launch.sh` 是原多 lane 流程，现已要求 Slurm allocation，
但仍保留旧 GPU 编号调度，不能作为新的单卡提交入口；本步不提供 GPU 启动命令。
`stop_at_boundary.py` 是历史迁移辅助工具，现要求显式 `--lanes-json`，不再携带旧 PID。

## 两份机器配置

在仓库根目录、全新的 shell 中选择一份配置（不要连续 source 两种机器配置）：

```bash
# labserver：配置加载和检查只使用 CPU，不提交实验。
source experiments/pi05_libero_v1_fmn/labserver_env.sh
"$V1_PI_PY" experiments/pi05_libero_v1_fmn/check_paths.py
```

```bash
# BCC：由 Litian 在 BCC 执行。本地 shell 语法已检查，BCC 实机尚未验证。
source experiments/pi05_libero_v1_fmn/bcc_env.sh
"$V1_PI_PY" experiments/pi05_libero_v1_fmn/check_paths.py
```

这些是配置/检查命令，不是 Slurm 提交命令。Slurm 命令会在下一步加入。
机器配置可在 source 前使用同名 `V1_*` 变量覆盖，例如：

```bash
export V1_RUN=/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/my_validation
source experiments/pi05_libero_v1_fmn/labserver_env.sh
```

| 变量 | 含义 |
| --- | --- |
| `V1_ROOT`, `V1_ASSETS` | 机器工作区、外部资源根目录 |
| `V1_REPO` | 实际 XPolicyLab checkout；默认从当前脚本位置推导，防止导入旧 checkout |
| `V1_RUN` | 实验输出；默认使用新 `pi05_libero_v1_fmn_portable` 目录，避开历史归档 |
| `V1_DATA`, `V1_BASE`, `V1_OPENPI` | 数据、初始权重、openpi 源码 |
| `V1_PI_PY`, `V1_SIM_PY` | 训练/服务与仿真 Python 可执行文件；保留环境内的符号链接路径 |
| `V1_PI_PATHS`, `V1_SIM_PATHS`, `V1_WS_PATHS` | policy、仿真及 websocket 的额外模块搜索目录 |
| `V1_LIBERO_CONFIG` | LIBERO 配置目录 |
| `V1_CACHE`, `V1_JAX_CACHE` | 缓存根目录、JAX 编译缓存 |
| `V1_TMPDIR`, `V1_HF_HOME`, `V1_OPENPI_DATA_HOME` | 可选的临时目录、HF 和 openpi 缓存覆盖 |
| `V1_FFMPEG` | 可选 ffmpeg 路径；默认从 PATH 查找 |
| `V1_DOWNLOAD_LOG` | 旧准备脚本使用的下载完成日志 |
| `V1_LEGACY_DATA_ROOT` | 非标准旧数据目录迁移时显式指定旧根目录 |

所有机器地址集中在 `labserver_env.sh` / `bcc_env.sh`。配置变量是绝对路径，
共享 Python 代码不再写死机器地址；Linux 的 `/proc`、`/dev/null` 不属于机器资源路径。
配置加载本身不创建目录，实际工作时才创建所需输出和缓存目录。
labserver 的 TMPDIR、HF、openpi、JAX、XDG、Torch、Triton 和 CUDA 缓存默认全部位于 `/data2`。
BCC 数据/环境路径来自 Litian；输出改为新目录，缓存默认位于 assets 文件系统，
不自动采用原脚本的 `/scratch`。Litian 可用 `V1_JAX_CACHE` 覆盖。
HPCC 的真实路径需要 Litian 提供；同一 commit 可通过配置覆盖使用，不预设其目录。

## checkpoint 和数据记录的迁移

新 `trained.json` 的 checkpoint 相对于流目录（例如 `sf/libero_spatial`）：

```json
{
  "checkpoint": "task00/checkpoints/pi05_libero_cl/stage/10000",
  "checkpoint_path_base": "stream"
}
```

训练衔接和评测加载均在当前流目录解析它。旧绝对 checkpoint 记录按
`方法/套件/taskXX/checkpoints/...` 识别并映射到当前流；即使原地址仍存在，也不会回退读取原地址。
缺失、目录越界、方法或任务不匹配时直接报错。完整性检查只确认 `params` 和 `train_state`
目录存在，真正的 Orbax 恢复及学习率/Adam 连续性仍需 Slurm 测试验证。

新 manifest 的 `episode_files` 相对于 `V1_DATA`。旧 manifest 的标准
`data/chunk-NNN/*.parquet` 路径自动映射到新数据根目录；其他旧布局需设置
`V1_LEGACY_DATA_ROOT`，未知路径不会静默猜测。图像方向检查也使用同一个解析逻辑。

每条新流首次训练时保存 `metadata/manifest.json`（该 suite、相对数据路径）以及
`metadata/norm/<suite>/`（原归一化文件的精确副本，不重新计算）。搬迁时复制完整流目录，
包括 `metadata`、buffer、checkpoint、trained/evaluated 记录，并保持方法/套件目录结构；
目标配置还须指向同一数据集、兼容的底座权重与依赖。旧流没有 `metadata` 时仍支持
从 `V1_RUN/manifest.json` 和 `V1_RUN/norm/` 读取；迁移这种旧流时须一并携带这两份原始资料。
已有历史实验与 JSON 文件不会被批量改写。

运行日志中的绝对路径仅用于诊断，不作为续跑索引；它们无需为历史可读性而重写。

## CPU 检查

```bash
source experiments/pi05_libero_v1_fmn/labserver_env.sh
mkdir -p "$TMPDIR"
PYTHONDONTWRITEBYTECODE=1 "$V1_PI_PY" -m unittest discover \
  -s experiments/pi05_libero_v1_fmn/tests -v
```

测试覆盖新旧 checkpoint 迁移、原地址仍存在时优先新流、旧 manifest 兼容、
目录越界、符号链接、环境覆盖、流自带 metadata、评测加载路径和已评测任务跳过。
这些 CPU 检查不代表 BCC/HPCC 已验证，也不代表 Slurm 训练/评测验收已完成。
