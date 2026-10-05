# pi0.5 LIBERO v1 baseline

本目录从 labserver 的
`XPolicyLab-assets/baselines/pi05_libero_v1_fmn/code/` 原样导入（`eacea21`），
随后进行机器路径配置与迁移支持改造。机器配置参考 Litian 于 2026-10-03
提供的 `code_from_bcc_20261003/bcc_env.sh`；没有引入该副本的四卡调度或清理逻辑。
训练配方、图像方向、优化器、学习率和评测次数保持 v1 设置。

## 当前范围

已完成代码入库、机器配置/路径迁移，以及 labserver 单卡单流 Slurm 入口。
每个作业申请一张 GPU，依次执行准备、训练、模型服务和仿真评测，所有子进程留在
同一个 Slurm step/cgroup 中。程序继承 Slurm 的 CUDA_VISIBLE_DEVICES，不接受物理 GPU 编号。
GPU 空闲与排队由 Slurm 负责；已移除 wait_idle、GPU guard 和退出码 75 的等待重试。

ER 两任务训练冒烟已通过（作业 1001）；1 万步/50 次单任务评测、scancel 训练恢复及双流并行仍待验收。checkpoint 清理与可训练参数快照已实现并通过 CPU 检查，完整模型 Slurm 验证仍待完成。当前入口检查不等同于这些验收。

## Slurm 提交（labserver）

在仓库根目录执行；默认 gpu 分区、1 GPU、12 CPU、60G 主机内存、3 天时限：

```bash
# 一条正式 SeqFT Spatial 流；输出作业号，由 Slurm 分配空闲 GPU。
bash experiments/pi05_libero_v1_fmn/submit.sh sf libero_spatial

# 仅训练并评测第一个任务（1 万步、50 次评测），用于后续验收。
bash experiments/pi05_libero_v1_fmn/submit.sh sf libero_spatial --tasks 1

# ER 两任务预检：每任务 5 步，每个已见任务 2 次评测。
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial --preflight

# 轻量入口检查：小型 JAX GPU 运算、4 个真实 WebSocket 服务、LIBERO 256px 渲染。
# 不训练模型、不加载模型权重，不产生成功率结论。
V1_SLURM_PARTITION=debug V1_SLURM_TIME=00:10:00 \
V1_SLURM_CPUS=4 V1_SLURM_MEM=16G \
bash experiments/pi05_libero_v1_fmn/submit.sh sf libero_spatial --entry-check

squeue -u "$USER"
scontrol show job JOB_ID
scancel JOB_ID
```

不要因为 Slurm 仍显示 PENDING 就在 SSH 终端直接启动训练。内存或 CPU 也可能造成排队，
即使部分 GPU 没有分配。debug 最长 30 分钟，完整预检含编译和模型加载，可能超时，
因此完整预检默认仍提交 gpu 分区。`--tasks` 指从任务 0 开始共执行多少个任务；已完成的会跳过。

`submit.sh` 加载机器配置，`submit_jobs.py` 固定代码版本并调用 sbatch；`labserver.sbatch` 在作业内重新加载
配置，用 `eval "$(conda shell.bash hook)"` 初始化 Conda 后再 activate，然后用 srun 启动 lane。
labserver 默认激活 base 作为脚本环境，训练和仿真仍使用各自的 `V1_PI_PY` / `V1_SIM_PY`。
若不需要 Conda，显式设置 `V1_CONDA_ENV=''`。不要直接提交缺少 V1_CODE 的 sbatch 文件。

可用 `V1_SLURM_PARTITION`、`V1_SLURM_TIME`、`V1_SLURM_CPUS`、`V1_SLURM_MEM` 覆盖资源配置；
GPU 数固定为 1。通过 `V1_RUN` 分开不同重复实验。`pipeline.sh` 现在只转发到 `submit.sh`；
旧 `gate_then_launch.sh` 会报错提示使用新入口，`stop_at_boundary.py` 和 `gpu_guard.py` 已退役删除。
BCC/HPCC 的分区、账号等资源规则尚未验证，当前提交脚本的默认值仅针对 labserver。

## 锁、端口、状态与日志

- 每条流目录里的 `stream.lock` 是内核 flock：重复提交同一输出流会快速失败，进程退出后自动释放。
  文件可以留在磁盘，不代表锁仍被占用，也不要通过删除文件解锁。
- 每次运行写 `status_job_<job_id>.json`，各作业互不覆盖；preflight 和入口检查使用独立子目录。
- 每个 policy server 用端口 0 让操作系统直接绑定空闲端口，成功监听后才写出带 PID/job ID 的就绪记录。
  不存在“找空闲端口后先释放再绑定”的竞争窗口，每个评测客户端独占一个服务。
- EGL 在 Slurm cgroup 内识别唯一可初始化的 NVIDIA 设备，不假定 EGL 编号等于 CUDA 编号。
  若无法唯一确定，作业报错停止，不使用未分配的 GPU。
- 调度日志在 `$V1_RUN/slurm/<job-name>-<job-id>.log`；流内保留训练和评测子进程日志。
  评测失败会保留已完成 episode 记录，重新提交时读取这些记录。
- 信号会中断调度并清理子进程；完整 checkpoint 保存在原有的每 1000 步间隔和任务结束点。
  没有新增“取消时立即存盘”；自动接续作业见下文。
- 本机 Slurm accounting 当前关闭，sacct 不可用；以 squeue/scontrol、Slurm 日志和状态 JSON 为准。

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

这些是配置/检查命令；labserver 的 Slurm 提交命令见上文。
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
目录存在，真正的 Orbax 恢复及学习率/Adam 连续性仍需后续完整 Slurm 训练测试验证。

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
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "$V1_PI_PY" -m unittest discover \
  -s experiments/pi05_libero_v1_fmn/tests -v
```

测试覆盖新旧 checkpoint 迁移、原地址仍存在时优先新流、旧 manifest 兼容、
目录越界、符号链接、环境覆盖、流自带 metadata、评测加载路径和已评测任务跳过。
这些 CPU 检查不代表 BCC/HPCC 已验证，也不代表完整 Slurm 训练/评测验收已完成。

## 已执行的 Slurm 入口验证

作业 **980**：debug 分区，1 GPU / 4 CPU / 16G，COMPLETED，ExitCode=0:0，耗时 66 秒。
JAX 只看到 `cuda:0` 并完成小型运算；4 个系统动态端口上的真实 WebSocket 服务均收到
`probe-ok` 响应；LIBERO 生成了 256x256 RGB 图像。policy 和 simulator 的进程均位于
`job_980/step_0` cgroup。证据见 [validation/slurm_entry_980.json](validation/slurm_entry_980.json)。
这次未加载 pi0.5 checkpoint、未训练模型；不作为完整冒烟、训练成功率、训练中断恢复
或两条训练流并行的验收证据。BCC/HPCC 尚未实机验证。

## 跨作业续跑和一个接续作业

```bash
# 一次提交两个作业，第二个依赖第一个结束（afterany）。
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial --chain-next

# 只重新提交一次；保持 V1_RUN、方法、套件与预检模式一致。
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial
```

提交输出列出两个 job ID，记录保存在 `$V1_RUN/slurm/submission_<first_job>.json`。
`--chain-next` 只排一个接续作业，不无限重提。依赖是 afterany，成功、超时或取消之后都
能解除依赖，但仍须等待资源；不要使用只接受成功退出的 afterok。若要彻底停止这对作业，
执行 `scancel FIRST_JOB NEXT_JOB`，只取消第一个会让接续作业获得运行资格。

接续作业会检查前一个作业的流内状态：明确的 failed 状态或缺失状态会报错退出，
要求先排查；被中断或被系统终止后留下的训练/评测状态可以继续。提交第二个作业失败时，
错误会明确报告第一个已提交的 job ID，不会悄悄取消或重复提交它。

每次提交使用当前 **已提交的 HEAD** 在 `$V1_RUN/code_snapshots/<commit>/XPolicyLab`
生成源码快照。同一对作业使用完全相同的快照、机器配置和输出目录，之后修改工作区不会
影响它们。实验目录/机器配置有未提交修改时拒绝提交。仓库其他文件的未提交改动不会进入
快照；若实验需要它们，先有选择地提交。外部 openpi、Python 环境和数据不复制，也必须保持
兼容；不能在运行过程中替换外部依赖。快照只保存代码，不保存数据和权重。

恢复规则：

- `trained.json` 和完整的 `evaluated.json` 都存在、协议一致：跳过任务，即使旧 checkpoint 已清理。
- 训练完成、评测未完成：只补评测，保留已完成 episode 记录。
- 任务训练未完成：仅从 Orbax 返回的 finalized checkpoint 中取最新一步，恢复完整参数、
  optimizer 状态及全局 step。未完成的临时保存目录不参与选择。最终保存目录缺失组件时会报错，
  不静默重置训练；完全没有 checkpoint 时才从底座/上一任务重新开始该任务。
- 不重新 warmup；学习率按恢复的 global step 计算。起始步数大于 0 时检查 Adam 动量范数非零。
- 每次启动在 `resume_history.jsonl` 和训练日志 `RESUME_STATE` 中记录来源、checkpoint、
  job ID、commit、起始全局/任务步数、实际起始学习率和 Adam 动量范数。
- 已完成整条请求流时，接续作业只检查并跳过任务，不重新准备数据或训练。

每 1000 步及任务末保存一次，取消/超时后未保存的尾部需要重算。真正恢复时仍由 Orbax
检查 checkpoint 内容；本步的 CPU 测试不替代后续完整模型的 scancel/恢复验收。

### 合成接续调度检查（非训练验收）

```bash
# 使用全新 V1_RUN。第一个作业写入小型 JSON 后故意退出 99；第二个读取并推进它。
V1_RUN=/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/my_chain_check \
V1_SLURM_PARTITION=debug V1_SLURM_TIME=00:02:00 \
V1_SLURM_CPUS=1 V1_SLURM_MEM=2G \
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial --chain-next --chain-check
```

该模式仅验证真实 Slurm 的 afterany 依赖、顺序启动、同一代码版本和共享进度交接。
小型 JSON 不是模型 checkpoint，没有训练、Adam 或学习率恢复证据，也不是三天超时实测。

合成作业 **989 → 990** 的应用层进度交接已核实：989 写入 value=1 后按设计退出 99，
990 在其后启动，以同一 commit `74bbe0fbb76f051eff883e04d582ab5ee4d34d9d`
读取进度并写入 value=2、`chain_probe_passed`，提交记录确认依赖 `afterany:989`。
989 的退出码有 srun 日志佐证；990 的最终 Slurm 退出码未能补查：scontrol 记录已过期，
accounting/JobComp 未启用，当前账号无权读取控制器日志。因此仅确认合成进度交接成功，
不宣称完整调度退出码验收通过，也不代表模型 checkpoint、Adam 或学习率恢复通过。
原始状态、日志、时间顺序及核查限制见
[validation/slurm_chain_989_990.json](validation/slurm_chain_989_990.json)。

## Checkpoint 清理与可训练参数快照

新建流默认启用 checkpoint 清理，额外快照默认关闭：

```bash
# 关闭额外快照：完成十个任务后仅留下任务 9 的完整 checkpoint。
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial --chain-next

# 开启额外快照：另用一个新输出目录，从任务 0 开始留存全部历史快照。
V1_RUN=/data2/vla-reasoning/proj/XPolicyLab-assets/baselines/pi05_v1_with_snapshots \
bash experiments/pi05_libero_v1_fmn/submit.sh er libero_spatial \
  --chain-next --save-trainable-snapshots
```

重新提交时保持输出目录、模式和快照开关一致。开关写入流内
`retention_policy.json`；中途改变会报错，因为已清理的早期模型无法补出历史快照。
接续作业继承相同开关。已有 task 目录、但没有该策略文件的旧流仍按旧方式续跑，
不自动清理、不补历史快照；开启本规则请从新的 V1_RUN 开始。不会批量修改历史归档。

### 清理时机

1. 训练每 1000 步及任务末保存完整 checkpoint。新流使用原生 Orbax 的
   `keep_period=1` 保留训练期间的保存点，不让原生 `max_to_keep=1` 提前淘汰中间点。
2. 当前任务最终保存已完成、`trained.json` 已发布后，清理前一任务的完整 checkpoint。
   前一任务必须已经评测完成；开启快照时，两任务的快照也必须完整。
3. 当前任务评测成功，评测记录和 matrix 已保存后，删除当前任务的中间 checkpoint，
   只保留最终完整 checkpoint。评测失败时保留当前任务的中间保存点。
4. 十任务流完成后，只剩 task09 的完整 checkpoint。开启快照时，另保留 task00–09
   各一份 `trainable_snapshot/`。日志、buffer、metadata、训练及评测记录均保留。

删除只发生在持有 stream.lock 的调度进程中，限定本流已知 checkpoint 目录，
拒绝符号链接。删除前写退休记录和审计日志；中途被终止后可以重试，
并沿退休记录确认还有较新的完整 checkpoint，不能把意外缺文件当作已清理。
`retention.jsonl` 记录删除开始、完成和释放的已分配字节。没有评测完成记录不会执行中间清理。

### 快照的内容与读取

`taskXX/trainable_snapshot/params/` 是 Orbax PyTree，仅保存
`state.params.filter(config.trainable_filter)`：与实际训练相同的可训练参数集合，
包括视觉编码器、LoRA 及其余可训练投影等；没有冻结权重、Adam 状态或学习率调度状态。
快照不是完整续训 checkpoint。训练恢复仍从完整 checkpoint 加载。

`manifest.json` 记录任务、global step、参数名称/形状/dtype、参数数量/字节、
代码 commit、模型配置和底座来源。base_path_provenance 是来源说明，不是迁移后的加载路径。
保存先写流内临时目录，成功后原子发布；不完整快照不会允许完整 checkpoint 被清理。
中断遗留临时快照在成功发布且完整 checkpoint 确认后清理。

以后编写探针时，可用兼容版本的 Orbax 读取 `params` 树，再按 manifest 中的参数名称、
形状和 dtype 校验目标模型后替换选定部件。跨 CPU/GPU 拓扑恢复需要为目标树显式提供
restore/sharding 参数，不能假设保存时的 GPU 拓扑在其他机器仍然存在。
本次只增加快照保存，没有实现具体 MoE 探针实验。

### 磁盘记录与尚未完成的实测

以下按用户提供的约 8.9 GB/完整 checkpoint 和约 2 GB/快照估算，
**不是新实现的十任务流实测峰值**：

| 设置 | 完成后的权重占用估算 | 运行中同时保留的权重占用估算上界 |
| --- | --- | --- |
| 快照关闭 | 约 8.9 GB | 约 97.9 GB（前一任务 1 份 + 当前任务 10 个保存点） |
| 快照开启 | 约 28.9 GB | 约 117.9 GB（上述 11 份 + 最多 10 份快照，保守相加） |

该估算不包含共享缓存、数据、底座、源码、文件系统开销和中断残留临时文件。
`disk_usage.json` 在保存/清理边界记录本流已分配字节及采样峰值；这是离散采样，
不包含共享缓存，也不能代替连续磁盘峰值实测。十任务流开/关两种配置的真实峰值仍待
后续 Slurm 完整实验测量，不能将小型测试数字当作模型占用。

labserver 在创建目录之前校验 V1_RUN 与所有配置缓存目录的真实路径都在 /data2 下，
包括 TMPDIR、HF、openpi、JAX、XDG、Torch、Triton、CUDA、pip、uv、matplotlib、numba。
路径覆盖或符号链接指向根分区时会报错。BCC 使用自己的机器配置，不套用 /data2 路径限制。

### 本步验证范围

通过 36 项 CPU 回归测试：包含清理时序、评测失败、删除中断重试、目录迁移、
旧流保护、十任务结束的保留数量、快照开关传递给接续作业，以及真实 Orbax
快照往返读取（包括 bf16）。另外使用当前 openpi 原生保存器在 CPU 上保存三个
小型 checkpoint，确认 keep_period=1 保留全部保存点且合成 step/state 可恢复。
根分区缓存覆盖被实际拒绝，shell 语法与本实验目录补丁检查通过。

这些检查没有运行完整 pi0.5 训练，不替代 Slurm 冒烟、真实 Adam/LR 中断续跑、
成功率、并行实验、BCC 实机或磁盘峰值验收。


## 实际 ER 两任务冒烟：作业 1001

固定代码 commit 为 05721d3bc66055572de264dec7f8a76c4fc46d25。
gpu 分区，1 GPU / 12 CPU / 60G；Slurm 已确认 COMPLETED、ExitCode=0:0，耗时 10 分 6 秒。
任务 0 全局步数 0→5，任务 1 为 5→10；任务 1 起始 Adam 动量范数
0.13622810071570227，衔接学习率 1.498501660535112e-7。
任务 0 batch 为当前 8，任务 1 为当前 8 + 回放 8。
两阶段分别完成 2 和 4 次评测，三格成功率均为 0；5 步冒烟仅证明流程，
不作为完整训练成功率结论。

开启快照后保留两份各含 466,957,072 个参数的快照，参数载荷每份 1,867,828,288 字节
（不含文件格式开销）。task00 完整 checkpoint 已按规则删除，task01 最终 checkpoint 保留。
这不是十任务峰值实测。原始配置、逐 episode 结果、清理记录和 Slurm 最终状态见
[validation/slurm_preflight_1001.json](validation/slurm_preflight_1001.json)。

从后续提交起，batch 脚本另写入 $V1_RUN/slurm/exit_JOB_ID.json，保留 srun 返回后的
batch 退出码。SIGKILL/节点故障可能阻止落盘，缺文件不能当作退出 0；
此记录与调度器最终状态分开报告，及时留存 scontrol 输出仍有必要。

## 实际任务内中断恢复：1021 → 1131

SeqFT Spatial task00 的作业 1021 在 checkpoint 5000 完整提交后执行 scancel，
取消前日志记录到 5100 步；Slurm 留存为 CANCELLED / ExitCode=143:0，batch 退出记录也为 143。
同一 commit 036a60a9d71c5f8da6697304185d4a90e3f4179a、配置及流目录重提为 1131。
1131 已从 checkpoint 5000 恢复，起始 Adam 动量范数 0.10483583735883913，
起始学习率 2.3960221369634382e-5，与中断前第 5000 步记录一致；未重新 warmup。
1131 随后完成 10000 步，50 个不重复 episode（0–49，seed 42–91）全部成功，
成功率 1.00，达到 >=0.90 的验收标准；lane 完成且 batch EXIT trap 记录退出码 0。
复查时 Slurm 最终状态记录已过期，因此未确认调度器最终状态，不将 batch 记录冒充
Slurm COMPLETED。已完成任务跳过已由作业 1865 核实（见下文）；双流并行尚待最终验收。
证据见 [validation/slurm_resume_1021_1131.json](validation/slurm_resume_1021_1131.json)。

## 已完成任务跳过：作业 1865

以同一 V1_RUN 和原冒烟命令（含 --save-trainable-snapshots）重提，
1865 跳过 task00、task01，Slurm COMPLETED / ExitCode=0:0，batch 退出码 0。
两任务训练、恢复、评测日志及 JSON 记录的文件集合、SHA256、大小和修改时间均未改变。
仅 checkpoint 清理标记 checkpoints_retired.json 被原样重写，内容及大小未改变。
证据见 [validation/slurm_skip_1865.json](validation/slurm_skip_1865.json)。

并行冒烟 1863（Spatial）与 1864（Object）已同时运行，Slurm 分别分配物理 GPU 2、5；
固定代码 commit 59251995242d7cb14350bddec81d5b0fa2346cfe，快照关闭。
这仅确认实际启动重叠与资源分配，两条流最终通过及端口互不冲突仍待核实。
