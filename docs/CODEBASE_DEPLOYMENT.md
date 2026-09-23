# XPolicyLab：跨服务器部署与实验运行

这份说明对应分支 `codex/portable-benchmark-deploy`。目标是让新服务器使用相同的 adapter、benchmark client 和实验协议，所有环境、权重、数据和结果均放在 Git 仓库外。

两个入口：

- [`scripts/deploy/build_env.sh`](../scripts/deploy/build_env.sh)：创建 Conda 环境、获取固定版本源码、下载并校验 checkpoint、准备数据和模拟器资产。
- [`scripts/deploy/run_experiments.sh`](../scripts/deploy/run_experiments.sh)：启动 checkpoint 评测，或 π0.5 + LIBERO 的顺序微调 / ER 正式队列。

`.sh` 只负责定位入口；参数校验、安装、下载和进程管理由同目录 Python 文件实现。命令使用参数数组执行，支持含空格的路径。

## 1. Codebase 结构与当前支持范围

```text
XPolicyLab/
├── policy/{Pi_05,X_VLA,GR00T_N15}/   模型适配：图像、state、归一化和 action
├── benchmarks/libero/               LIBERO 的 reset / step / success
├── benchmarks/robocasa365/           RoboCasa365 的统一环境 client
├── client_server/ws/                统一 WebSocket 调用
├── experiments/pi05_libero/         本次正式 SF/ER 队列、回放、评测和恢复
├── scripts/deploy/                  新服务器部署和运行入口
└── docs/CODEBASE_DEPLOYMENT.md
```

在线链路是 `simulator → benchmark client → WebSocket → policy adapter → model`，动作沿反向路径返回。公共 client 不依赖具体模型；图像保持 RGB，模型特定旋转在 adapter 中处理。

| 模型 / 环境 | 一键 checkpoint 评测 | 本入口的正式 SF/ER |
|---|---|---|
| π0.5 / LIBERO | 支持，官方 `pi05_libero` | 支持，从 `pi05_base` 开始 |
| X-VLA / LIBERO | 支持，`X-VLA-Libero`，绝对 EEF 动作 | 尚未提供 |
| GR00T N1.5 / LIBERO | 支持，4-suite combined checkpoint | 尚未提供 |
| π0.5 / RoboCasa365 | 支持，Human300 checkpoint | 尚未提供 |
| GR00T N1.5 / RoboCasa365 | 支持，Human300 checkpoint | 尚未提供 |
| X-VLA / RoboCasa365 | 没有本次验证的 checkpoint / 完整配方 | 尚未提供 |

前期 sanity check 验证过上述五种推理组合的部分任务，不能据此宣称完整 benchmark 分数。现有 `training_smoke.py` 属于小规模训练检查，不等同于正式 SF/ER；尤其 X-VLA 原始 delta 示范与绝对动作 checkpoint 的标签转换还需要独立验证。本部署不把未完成的转换作为正式训练入口。

## 2. 服务器前置条件

- Linux x86_64、Bash、Git、Python 3.10+、可运行的 Conda（支持 `--conda /path/to/conda`）。脚本创建独立 Conda prefix，不修改 base 环境。
- NVIDIA GPU、工作正常的驱动和 `nvidia-smi`。现有验证硬件是 RTX 6000 Ada 48 GB。正式 π0.5 队列要求四张卡，参数不随显卡数变化。
- GR00T 的 FlashAttention / PyTorch3D 需要 CUDA 12 系列开发工具链（`nvcc`）、C/C++ 编译器、足够的编译内存。仅有驱动不能编译这些扩展。
- 系统安装 EGL/OpenGL 库和 FFmpeg；Ubuntu 可由管理员安装 `libegl1 libgl1 libgles2 libglvnd0 ffmpeg build-essential`。脚本不会自动使用 sudo。
- 能访问 GitHub、Hugging Face 和 Google Cloud Storage；RoboCasa 数据/资产还需要 Box。需要 HF 认证时在环境中设置 `HF_TOKEN`，不要写入配置或提交 Git。
- 权重通常为每套数 GB 到十余 GB，另有依赖、数据及编译缓存。建议预留至少 200 GB；选择多个模型、RoboCasa 数据或长实验时需更多。以实际下载 manifest 为准。

## 3. 一键部署

```bash
git clone --branch codex/portable-benchmark-deploy \
  https://github.com/Gonglitian/XPolicyLab.git
cd XPolicyLab

# 先预览；不创建环境、不下载、不写结果。
bash scripts/deploy/build_env.sh --root /data/$USER/xpl-assets --dry-run

# 三个模型 + 两个模拟器 + 基础/评测权重 + LIBERO 40 个任务的原始示范。
# RoboCasa 的厨房资产会一起安装，任务示范可按需下载。
bash scripts/deploy/build_env.sh --root /data/$USER/xpl-assets
```

只协助跑当前 π0.5 / LIBERO baseline 时，使用更小的安装范围：

```bash
bash scripts/deploy/build_env.sh \
  --root /data/$USER/xpl-assets \
  --models pi05 --benchmarks libero --checkpoints base --datasets libero
```

只做 checkpoint 评测可以用 `--checkpoints eval --datasets none`。LIBERO 初始状态和 BDDL 来自固定源码，无需训练示范也能评测。`--models`、`--benchmarks` 支持逗号分隔；重复执行可补装其他组合，已有环境及已验证权重会复用。

RoboCasa 示范按任务显式选择，默认示例为 `OpenDrawer / pretrain / human`：

```bash
bash scripts/deploy/build_env.sh \
  --root /data/$USER/xpl-assets \
  --models pi05,gr00t --benchmarks robocasa \
  --checkpoints eval --datasets robocasa \
  --robocasa-tasks OpenDrawer --split pretrain --source human
```

这不会隐式下载整个 RoboCasa365 数据集。`human` 与 `mimicgen` 保留来源区别；不修改 action 标签。LIBERO 的 GR00T LeRobot 训练数据不是原始 HDF5 的同一格式，后续训练可参阅 [`policy/GR00T_N15/README.md`](../policy/GR00T_N15/README.md) 的数据入口；本安装命令下载的 LIBERO HDF5 直接供 π0.5 队列使用。

目录示例：

```text
/data/<user>/xpl-assets/
├── deployment.json            后续运行使用的配置；不含 token
├── envs/                      bootstrap、pi05、xvla、gr00t、libero、robocasa
├── upstreams/                 固定 commit 的上游源码
├── checkpoints/               cl_base、各模型 LIBERO、RoboCasa checkpoint
├── datasets/                  原始数据和规范目录链接
├── config/                    LIBERO 路径配置、已安装 package 版本
├── downloads/                 可复用的 RoboCasa 下载归档
└── cache/                     模型辅助文件缓存
```

运行所需的 `env_cfg/` 由现有注册工具写到 **XPolicyLab 仓库的父目录**，符合项目路径约定。若已有同名配置内容不一致，会明确报错，不覆盖。

源码版本在 [`sources.json`](../scripts/deploy/sources.json) 固定；π0.5 使用该 fork 的 `uv.lock`。X-VLA、GR00T 和模拟器分别固定关键依赖，并保存实际安装清单。HF 未预先指定 revision 的资产首次解析成具体 commit 后写入 `download_manifest.json`；以后重试沿用该 revision。GCS 权重记录 generation 并校验 CRC32C；HF 大文件校验 LFS SHA256。仅复制文件大小相同不能代替校验。

## 4. 一键运行 checkpoint 评测

无需手动分别启动 policy server 和 simulator。入口会等待 server 就绪，client 退出后清理本次 server，并保留日志。

```bash
bash scripts/deploy/run_experiments.sh \
  --config /data/$USER/xpl-assets/deployment.json \
  --mode eval --model pi05 --benchmark libero \
  --suite libero_spatial --task-id 0 --episodes 5 --seed 42 \
  --policy-gpu 0 --env-gpu 0 --port 6321 \
  --output /data/$USER/xpl-results/pi05-libero-t0 --video

# X-VLA 或 GR00T：替换 --model xvla / gr00t，并使用不同输出目录。
# RoboCasa：
bash scripts/deploy/run_experiments.sh \
  --config /data/$USER/xpl-assets/deployment.json \
  --mode eval --model gr00t --benchmark robocasa \
  --task OpenDrawer --split pretrain --source human \
  --episodes 5 --seed 42 --policy-gpu 0 --env-gpu 0 \
  --output /data/$USER/xpl-results/gr00t-opendrawer --video
```

默认按各模型已验证配方执行 action chunk：LIBERO π0.5=5、X-VLA=30、GR00T=1；RoboCasa π0.5=5、GR00T=16。这里的评测是单个已有 checkpoint 的评测，**不会生成持续学习下三角矩阵**。结果在 `result.json`，服务日志在 `server.log`，`--video` 写 MP4。

`--checkpoint` 可提供兼容的本地评测 checkpoint；LIBERO GR00T 默认配置针对 combined 权重。不要把 π0.5 LoRA 持续学习 checkpoint 交给官方完整参数架构的普通评测入口；正式队列已配套正确的 LoRA 重载和归一化。

## 5. π0.5 / LIBERO 正式 SF、ER

```bash
bash scripts/deploy/run_experiments.sh \
  --config /data/$USER/xpl-assets/deployment.json \
  --mode baseline --model pi05 --benchmark libero \
  --gpus 0,1,2,3 --suites all --methods all \
  --output /data/$USER/xpl-results/pi05-libero-baselines
```

在 `tmux` 中运行，或使用 `nohup ... > queue.log 2>&1 &`。入口先检查数据，再做两阶段技术预检，然后自动训练与评估。恢复时再次执行相同命令、指定同一输出目录即可；锁阻止重复队列。选择不同 suite/method 必须使用不同输出目录。

| 协议 | 固定设置 |
|---|---|
| 顺序 | Spatial → Object → Goal → Long，每个 suite 先 SF 后 ER；各自从 base 开始 |
| 数据 | 每任务 50 条示范，任务默认顺序 0 |
| 训练 | 每任务 10k 更新；任务内恢复优化器，切换任务重置优化器和 schedule |
| batch | SF 全局 8；ER 当前 8 + 回放 8，第一任务为 8 |
| ER | 每旧任务固定 1000 个起点/action chunk，固定子集，合池 shuffle 重复遍历 |
| 模型 | π0.5 base，Gemma 与 action expert 仅 LoRA，bf16，FSDP=4，无 EMA |
| 优化器 | AdamW β=(0.9,0.95)，eps=1e-8，wd=1e-10，clip=1 |
| LR | warmup 1k，峰值 2.5e-5，10k cosine 到 2.5e-6 |
| 归一化 | 新任务到来时累计已见唯一数据统计，阶段内冻结；不包含未来任务 |
| 评测 | 最后一步 checkpoint，所有已见任务，固定 50 初始状态逐一配对 |
| 控制 | chunk 10 / 每 5 步重规划；horizon 220/280/300/520，另 10 步静置 |
| 种子 | 训练 42，rollout 使用确定性的 task/episode 派生种子 |

每 suite/method 一个 `matrix.json`，行是训练阶段，列是评测任务，仅填写已见任务。完整队列包含 80 阶段、800k 更新、440 格和 22k 次 rollout。`preflight/` 不计入正式结果。

四张所选卡必须连续 60 秒无其他计算进程、显存占用低于 1000 MiB、利用率不超过 5% 才开始。训练中每 5 秒检查争用，发现后保存恢复点并等待，不停止其他人的进程。此机制是检测和等待，不是集群调度器的独占预约；在共享集群上优先配合管理员/调度系统分配资源。普通单 checkpoint `eval` 不使用这个四卡条件，需自己选择获准使用的 GPU。

## 6. 师兄分担一部分实验

按独立 suite/method 分工即可，各流都从 base 初始化，例如让另一台服务器只负责 Goal、Long 的 ER：

```bash
bash scripts/deploy/run_experiments.sh \
  --config /data/$USER/xpl-assets/deployment.json \
  --mode baseline --gpus 0,1,2,3 \
  --suites libero_goal,libero_10 --methods er \
  --output /data/$USER/xpl-results/pi05-er-goal-long
```

同一流中任务存在顺序依赖，不应把 task 0–9 分给多台机器同时训练。交接时复制对应的 `matrix.json`、`evaluated.json`、每回合 `evaluation/worker*.json`、`run_config.json`、`moments.json`、`launch.json` 和源码 commit。恢复 checkpoint 含路径关联；直接把旧 `deployment.json` 或旧 manifest 搬到新服务器不等于完成迁移。推荐两台机器分别跑独立流。

只读状态/计时摘要，参数要与原启动一致：

```bash
bash scripts/deploy/run_experiments.sh \
  --config /data/$USER/xpl-assets/deployment.json \
  --mode baseline --status --gpus 0,1,2,3 \
  --suites libero_goal,libero_10 --methods er \
  --output /data/$USER/xpl-results/pi05-er-goal-long
```

计时只使用 `idle_gpu_timing_valid=true` 的稳定窗口，排除技术预检和共享卡旧数据。ER 未实测时会明确给出假设场景。训练、评测、编译加载保存开销和 GPU 等待时间应分别统计；摘要中的纯训练区间不是完整交付时间。

## 7. 检查与常见问题

```bash
python3 -m unittest discover -s scripts/deploy/tests -v
bash -n scripts/deploy/build_env.sh scripts/deploy/run_experiments.sh
```

- `waiting_for_four_idle_gpus`：队列正常等资源；查看 `status.json` 的 `busy`。
- `failed`：查看 `status.json`、对应阶段 `train.log` 和评测 worker 日志；修复后原命令恢复。
- HF 401/403：检查账号访问权和 `HF_TOKEN`；不会退回另一个 checkpoint。
- checksum mismatch：保留 manifest，针对报错文件修复后重试，不要把整个共享数据目录删除。
- 已有源码 commit 不同：换新的部署根目录；安装器不会 `git reset --hard` 覆盖源码。
- RoboCasa dataset 已存在但无验证标记：安装器会停下，防止把半成品当作完成；手动验证或使用新目录。
- 端口冲突：换 `--port`；baseline 会占连续四个端口。

本次验证：13 项部署控制测试、37 项 adapter/benchmark 回归测试通过；Linux 上使用真实 Spatial HDF5 数据通过目录清单、动作 chunk 边界、归一化逆变换、累计统计、模型输入形状和学习率端点检查；GR00T/LIBERO 的依赖清单通过 Linux 解析检查。所有验证使用独立目录，数据检查禁用 GPU，没有替换正在等待的正式队列。

验证边界：上述检查复用了现有服务器的模型/模拟器环境，没有在全新机器重新下载并安装全部 Conda 环境、权重和数据，也没有完成 80 阶段全量训练。目标机器的 CUDA 扩展编译、网络和首次冷启动仍需实际验证；dry-run 本身不代表完整复现。
