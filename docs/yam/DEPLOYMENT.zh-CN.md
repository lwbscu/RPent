# YAM 三机部署与现场验收顺序

这些命令是本地实现的部署说明，**本轮未在真机执行**。完整依据见 [INTEGRATION_PLAN.zh-CN.md](INTEGRATION_PLAN.zh-CN.md)。只有本地验收后才同步控制机。现场参数须从实际环境读取，不能使用示例空标定直接运动。

## 1. 本地安装与验证

在 fork `yam` 分支的 checkout 内，使用 Python 3.11：

```bash
uv venv --python 3.11 .venv311
uv pip install --python .venv311/bin/python -e . pytest ruff==0.15.22 omegaconf imageio-ffmpeg pyrealsense2==2.58.4.10922
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv311/bin/python -m pytest -q tests/yam tests/robotwin tests/robocasa
.venv311/bin/ruff check --preview robots/yam tests/yam rpent/cli/main.py
.venv311/bin/python -m rpent.cli.main --robot yam --help
```

关闭第三方 pytest 自动插件仅避免本机 ROS 插件污染；这里不需要 ROS 或真实 CAN/相机。控制机的实际 RLinf、i2rt、mink/mujoco、RealSense 依赖复用已经工作的真机虚拟环境，不盲目升级或重装。

Agent 机处理非零畸变的像素反投影时也需要 `pyrealsense2`（只调用纯几何函数，不打开相机）；安装到 Agent 自身虚拟环境即可。未安装或遇到 SDK 不支持直接反投影的模型时保留图像并报告世界坐标不可用，不能静默忽略畸变。

## 2. 控制机配置

1. 保留 `/home/yambox/cynws/RPent` 现有内容，先查看是否存在、分支和未提交文件。不要对未知目标使用 `rsync --delete`、`git reset --hard` 或覆盖原配置。
2. 新安装从 fork 拉取验收提交；已有工作区先比较差异后只同步本次 RPent 代码。Git worktree 的 `.git` 是本地主仓库路径，不能直接把整个本地工作树连 `.git` 拷到另一台机器。
3. 复制 `robots/yam/config.example.json` 为未跟踪的现场 JSON。填 `task_name`（机器可读名称）、`task_language`（采集语言）、`extrinsics_path`、`operator_receipt_path`、现场 table_z/限位、相机 serial 与实际设备参数。
4. `extrinsics_path` 指向已验证的 RLinf `calib_data/run1/extrinsics.json`。核对其中 wrist 和 top 矩阵；改变相机安装后须重标定。标定文件、weights、相机序列号与现场配置不进入公开提交。
5. RLinf 路径用 `RPENT_RLINF_ROOT=/home/yambox/cynws/RLinf` 或 `RLINF_REPO_PATH`。适配只 import 实际 RLinf runtime/IK，不复制/改写 RLinf 文件。

先让现场人员确认采集、Pico、主臂遥操、hover、其他 env_server 均未占用 follower CAN。相机和 CAN 的实际使用权必须唯一。服务默认绑定 loopback，跨机用已配置的专网或 SSH 端口转发。

```bash
export RPENT_RLINF_ROOT=/home/yambox/cynws/RLinf
python -m robots.yam.env_server \
  --config /path/to/yam-site.json \
  --transport http --host 127.0.0.1 --port 8110
```

服务构造不运动；首次 observe 会初始化相机和控制运行时并保持实测姿态。**初始化电机/保持扭矩也是现场操作**，不能把启动硬件等同纯网络查询。当前实现不会自动 fold/home、清错、重新使能或解锁硬件故障。

本地操作员通道：

```bash
python -m robots.yam.operator_control --config /path/to/yam-site.json --event status
python -m robots.yam.operator_control --config /path/to/yam-site.json \
  --event start --episode-id CURRENT_ID --note '已人工布置首回合，确认工作区与停驻臂'
```

`start` 用于 Agent 开始记录前的首回合：写 ready 回执并调用 env.reset，输出新的 episode ID。Agent 客户端连接不 reset，evaluation 模式不提供 reset 工具。

Explore 重试时，操作者恢复场景后改用 `--event ready`；它只写一次性回执，随后由 **Agent 的 reset 工具**消费回执、开始新 episode 并记录尝试。不要在 Agent 的尝试之间使用 start。ready 不是动作，也不自动开始新 episode：

```bash
python -m robots.yam.operator_control --config /path/to/yam-site.json \
  --event ready --episode-id ACTIVE_ID --note '已恢复场景，供 Agent 重试'
```

人工 success/failure/abort 绑定实际执行的 episode：

```bash
python -m robots.yam.operator_control --config /path/to/yam-site.json \
  --event success --episode-id ACTIVE_ID --note '物体已稳定放入目标容器，夹爪松开'
```

现场中止使用硬件急停；软件 `abort` 是补充。软件返回 `stop_requested` 只证明已发出请求，不证明物理已停。记录测量状态与最终 hold 结果。

## 3. 推理机

**当前 VLA 未训练完成，本阶段后置；原语与 Explore 不依赖该服务。**

先准备 **YAM SFT 权重**与其 `yam` norm_stats；LIBERO/RobotWin 权重不能只改 action_dim 就替用。加载路径与 RLinf 的 `evaluations/realworld/realworld_dual_yam_openpi_rlinf_eval.yaml` 一致，使用 `openpi_rlinf.get_model` 的 eval wrapper；`pi05_yam_joint`、三图、预测 horizon 30，RPent 每次执行前 5 帧绝对 qpos14。若训练更改 horizon/频率/变换，更新明确 contract 并重新核对，不静默兼容。

```bash
export RPENT_RLINF_ROOT=/path/to/the/same-yam-rlinf-source
python -m robots.yam.vla_server \
  --model-path /path/to/yam-checkpoint \
  --norm-stats-path /path/to/yam-norm-stats \
  --device cuda --transport http --host 127.0.0.1 --port 8220
```

`norm-stats-path` 可省略，此时 RLinf 从 checkpoint 对应的 YAM assets 加载。使用公共 `vla.predict` RPC；YAM 输入与动作形状在实际预测调用中检查。输出仅通过 CPU numpy 跨 RPC；实际输出值和 checkpoint 行为仍需真实观测推理验证。

## 4. Agent 机

将服务通过可信连接暴露到 Agent 机后，用实际 endpoint 替换下例。当前先原语，再无 VLA Explore；不传 `--vla-endpoint` 时默认只连接 ENV，工具列表不含 `pi05_act`。`--without-vla` 仍可显式覆盖已配置的 endpoint：

```bash
python -m rpent.cli.main --robot yam \
  --task-name pick_place --seed 0 \
  --env-endpoint http://127.0.0.1:8110 --without-vla \
  --memory-profile local --memory-dir /path/to/yam-memory

python -m rpent.cli.main --robot yam \
  --task-name pick_place --seed 0 \
  --env-endpoint http://127.0.0.1:8110 \
  --explore --explore-sessions 1 --explore-attempts-per-session 5 \
  --memory-dir /path/to/yam-memory
```

示例未含 planner 凭据，需要使用已有 planner 配置；无 VLA 仍需要负责观察和决策的 LLM/VLM planner。Explore 默认在人工确认成功后自动将有效草稿、当前 episode 的 recipe/audit 归入本地 memory，并重建索引。失败笔记留在 inbox，冲突草稿进入 `_conflicts`，不会自动覆盖已有正文。`--no-auto-merge-memory` 可显式关闭归档。评测保持 memory read_only。更换 task_name/step_limit 时，ENV 配置与 Agent 启动参数须相同。真实 seed 是布局/试次标签，不承诺仿真式可重复生成场景；独立试验使用不同标签，同一 cell 重跑不会被当作新的独立记忆证据。

训练完成后，增加 `--vla-endpoint http://127.0.0.1:8220` 才会连接 VLA 并注册 `pi05_act`。YAM 不注册 Dashboard，也不从 Agent 进程启动本地 VLA。

## 5. 每阶段留存最小证据

| 阶段 | 记录 |
|---|---|
| 本地 | baseline/commit、Python/依赖、测试结果、RPC清理结果 |
| 观测 | 实际配置/标定指纹、三图/深度、K/畸变、cam2world、时间戳、静止姿态误差 |
| 原语 | 操作员在场、requested/accepted/measured、执行步数、位置/姿态误差、停止原因、录像 |
| VLA | checkpoint/norm指纹、camera_order、输入输出shape/数值、实际推理和端到端时延 |
| 任务 | task/episode ID、人工回执、失败/中止/成功计数、episode视频、recipe/audit |
| Explore | 每次人工复位与预算消耗、失败分析、成功当前attempt的recipe、本地memory merge结果 |

观测成功不等于动作安全，原语成功不等于任务成功，mock通过不等于真机验收。现场缺任何必要资产时报告具体缺失，不写虚假的完成标记。
