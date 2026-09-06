# YAM 本地接入验收记录

日期：2026-09-06。结论范围是本地代码与无硬件契约验证，**不代表真机操作、SFT 或任务成功已验收**。

## 代码与资产边界

- 本地工作区：`/home/lwb/Projects/thusigs/yam/RPent`；fork：`lwbscu/RPent`；新分支：`yam`。
- 基线：`codex/pr133-final-plan` / `0cf9d002b317db13a250ccceacc1eac519e9a1e2`。
- 写入范围：新增 `robots/yam/`、`tests/yam/`、`docs/yam/`；公共代码仅修改 `rpent/cli/main.py` 的 YAM 路由和权威 finish 记录。
- 原 RPent 工作树、RLinf 副本、现场标定和模型均未修改。
- RLinf 本地 Git pack 损坏，无法可靠记录其 commit；关键读取文件指纹保存在 [rlinf-source-sha256.txt](rlinf-source-sha256.txt)。未尝试修复该仓库。

## 当前交付

| 内容 | 状态与证据边界 |
|---|---|
| RobotSpec / CLI / 工具 / 三机 RPC | 本地实现；HTTP 与 socket 均通过 fake 环境回环测试 |
| RGBD 相机 / RLinf writer / FK / IK 适配 | 代码实现、静态签名核对与 fake 测试；真实依赖构造及硬件未验收 |
| move_to / act / gripper / finish | 本地完整调用链；实际机械可达性、时序和到位误差待现场验证 |
| 手眼变换与反投影 | legacy 矩阵方向回归；RealSense SDK 畸变函数对标；现场外参精度未复测 |
| reset / 人工裁决 / episode ID | fake 验证 ready 一次消费、新回合 ID、旧 success/旧动作拒绝 |
| 取消 | fake 验证 stop 请求与动作边界；软件 ACK 不证明物理停止 |
| Explore / recipe / MemoryManager | 本地 CLI 路由、成功导出、失败不发布测试；未运行真实 LLM Explore |
| SFT / norm_stats / YAM checkpoint | 未完成；本次仅实现加载现有 YAM transforms 的推理适配 |
| 现场同步 | 待本地检查完成及提供可用 YAM Box SSH 连接；未启动真机服务 |

## 本地验证

环境：Python 3.11.14，独立 `.venv311`；RPent editable install；ruff 0.15.22；pyrealsense2 2.58.4.10922。RealSense SDK 测试仅调用几何函数，不访问设备。

| 命令 | 最终结果 |
|---|---|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv311/bin/python -m pytest tests/yam tests/robotwin tests/robocasa -q` | **73 passed, 1 skipped** |
| `.venv311/bin/ruff check --preview robots/yam tests/yam rpent/cli/main.py` | 通过 |
| `.venv311/bin/python -m compileall -q robots/yam rpent/cli/main.py` | 通过 |
| `python -m rpent.cli.main --robot yam --help` 与 env_server/vla_server/operator_control 的 `--help` | 通过，未初始化硬件 |
| `git diff --check` / `git diff --cached --check` | 通过 |

唯一跳过项是 RoboCasa GPU 集成测试，需要显式 `RPENT_RUN_ROBOCASA_INTEGRATION=1` 与实际运行环境。其余 73 项不是实机测试。测试所有者已确认 fake 服务/thread 关闭并 join，任务进程残留为空。

关闭 pytest 第三方插件自动加载，是为了避开本机 ROS 插件引入的依赖，不改变项目代码。跳过项应按实际测试原因记录，不按通过计数。

## 本轮发现并修复的实际缺陷

1. `T_grasp_to_cam` 历史名字误导了矩阵方向：读取已改为直接使用，与 `solve_handeye.py` 的 `FK @ t_tcp_cam @ board_in_camera` 一致。
2. VLA 推理期间换回合导致旧动作可能执行：观测回合 ID 在推理前冻结，动作 RPC 到 writer 全链路比对；不匹配 hold/reject。
3. finish 的 `_finish` 信号被状态记录覆盖：保留终止信号；CLI 最终结果优先读取真实工具结果，并用当前 env 成功状态修正模型声明。
4. 用当前 FK 配旧腕相机帧可能给出错误坐标：显式观测等待 host 时间范围内的新帧并比较两次关节状态；运动/陈旧/无手眼时禁用该视角世界点，保留 RGB。该方法是静止近似，不是曝光同步。
5. stop 在 pacing 等待期间到达仍可能多下发一帧：command 前再次检查停止请求。
6. 块被提前中止时原语仍报告 success：完成标记按实际执行计数产生，move_to 另检查测量位姿。

## 尚不能据此宣称的能力

- 不能从用户报告的标定 std 推断绝对抓取误差；本机没有真实 extrinsics 可复测。
- TCP 桌面净空不是全连杆、自碰、双臂互撞或接触力规划；首轮单臂、另一臂停驻。
- 未复用 hover 脚本中可能在递归上限绕过桌面检查的执行路径。现阶段精确 quaternion IK 可返回失败；多 seed 姿态候选、抬移降策略仍需受控移植与现场验收。
- 30 Hz 是服务端命令目标节拍；相机、RPC、推理与观测开销未在 YAM Box 实测，不保证端到端 30 Hz。
- 未自动折叠、故障清错、重新使能或松扭矩；reset 是人工场景确认后开始新回合。
- 当前本地环境未安装完整 torch/mink/mujoco/i2rt 栈，不能把 fake 注入构造当真实 RLinf runtime/model 已加载。

后续按 [DEPLOYMENT.zh-CN.md](DEPLOYMENT.zh-CN.md) 逐步取得控制机观测、原语、checkpoint、Agent 和 Explore 证据。完整研究与实验设计见 [INTEGRATION_PLAN.zh-CN.md](INTEGRATION_PLAN.zh-CN.md)。
