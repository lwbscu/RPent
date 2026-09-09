# YAM 本地接入验收

## 2026-09-09 yambox 接手状态

已部署 fork/yam 并合并 upstream/main `24541d6`，现场配置在未跟踪日志目录。
本轮截至现场确认前：176 项离线回归通过，覆盖真实模型、无 VLA/Explore、公共 Memory 和 CLI。
重算原始标定得到与保存 JSON 完全一致的矩阵；未证明当前相机安装和绝对精度。
已修正 TCP 轴说明、保持目标漂移、相机观测配对、硬件租约失败重入，并增加实测轨迹检查与模型碰撞保护。
现场确认后，环境服务 loopback:8110 已初始化硬件并读取三视角、关节和 TCP；任务动作计数仍为 0。
真实无 VLA Agent 已调用观察和空间采样工具；首次启动因 Codex 版本失败，换用本机 0.153.4 后完成观察预检。
相同 Codex/MCP 链路对小图和约 1.8 MB 三图均正确识别随机字符及颜色；大图日志的文本封装是展示截断，未阻断实际视觉输入。
现场桌面拟合倾角约 4.5°、RMS 6.7 mm；标定推得右底座高 43 mm，尚需实际安装信息核对。
统一 10 mm 余量拒绝模型约 9.4 mm 的 base/link2 近接；已增加限定同臂几何对的正距离规则及有限倾斜桌体保护，通过离线测试和独立审查，未应用现场参数。
结果文件的 `not_successful` 表示未确立成功，不将尚未尝试的观察预检误称为人工裁决失败。
A/B 均未尝试搬运，成功 recipe 及新 episode 实际检索使用均未验收。
本轮后续相关测试合计 216 项通过，Ruff 与差异检查通过。真实 runtime 加假 backend 验证了非均匀跟踪滞后时预检目标与实际下发目标一致；这仍不是硬件动作验收。
当前以 [部署说明](DEPLOYMENT.zh-CN.md) 为准；以下保留历史验收，不作为当前接线描述。

## 历史记录

日期：2026-09-06。**只覆盖本地代码和无硬件契约验证，不代表真机、SFT 或任务成功已验收。**

## 代码边界

- fork：`lwbscu/RPent`；分支：`yam`；基线：`0cf9d002b317db13a250ccceacc1eac519e9a1e2`。
- YAM 实现在 `robots/yam/`。共享改动仅有 CLI 的 YAM 路由/finish，以及 `BaseEnvClient` 的 `reset_on_connect=True` 参数；YAM 显式传 False，其他机器人默认行为不变。
- 原 RPent 工作树、RLinf 副本及真机未修改。RLinf 本地 Git pack 损坏，本轮按实际源文件核对接口。

## 精简后的实现

| 保留内容 | 实际用途 |
|---|---|
| RobotSpec / Toolkit / EnvState / MemoryManager | 复用公共生命周期、工具、记录和记忆，支持 CLI 原语与 Explore |
| BaseEnvClient / BaseVLAClient / Base Facade / RPC | ENV 连接仅 observe；VLA 使用公共 predict；支持 HTTP/socket |
| RLinf YAM runtime / FK / IK | 单一控制路径；保留已有 PD 问题对应的 previous-command slew 和硬限位，`enforce_runtime_joint_limits=False` |
| RealSense RGBD + 标定 | 仅加载真实 solve_handeye JSON；内参来自 SDK；静止近似支持腕相机投影，有效性与限制随帧返回 |
| 人工 start / ready / verdict / episode ID / stop | start 启动首回合；重试 ready 仅写回执，由 Agent reset 消费；recipe 只取当前 episode 动作；ACK 不等于物理急停 |
| step 图片 / episode.mp4 / 成功 recipe | 现场复核和记忆导出；不保留 Dashboard 专用视频支线 |
| openpi_rlinf eval factory | 对齐现有 YAM eval YAML；预测 horizon 30，仅执行前 5 帧 qpos14 |

当前 VLA 未训好，默认不连接：无 endpoint 时只注册几何原语，提示词明确不可调用 pi05_act。双臂原语沿用 RoboTwin 的 `move_to/set_gripper/release + arm=left/right`；单次几何动作保持另一臂，不提供同时双臂 Cartesian 规划。夹爪返回实测开度。

无 VLA Explore 可使用公共文件工具写 technique，并在当前 episode 成功后自动调用 MemoryManager 合并本地 corpus、保留冲突和重建索引。跨 session 读取实际 session 记录，不再假设仿真式自动复位。物理场景恢复与成功裁决仍由操作者完成，尚非无人值守探索。

已删除专用 VLA client/元数据握手、无调用 helper、act 别名、无效参数、状态重复缓存、Dashboard 配置、本地 VLA 自动启动、额外 root 环境变量、另一套标定 schema、未接线 smoke JSON 和开发过程文件。

`table_z` 未配置时不启用桌面检查；配置后保留采样 TCP 净空检查。它不覆盖连杆、自碰、双臂互撞或接触力，现场配置与首轮单臂验证仍不可省略。

## 最终验证

环境：Python 3.11.14，`.venv311`，RPent editable install；ruff 0.15.22；pyrealsense2 2.58.4.10922。测试 SDK 只调用几何函数，不访问硬件。

| 检查 | 结果 |
|---|---|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv311/bin/python -m pytest tests/yam tests/robotwin tests/robocasa -q` | **86 passed, 1 skipped** |
| ruff check / format、compileall、YAM CLI help、git diff --check | 通过 |
| 子 Agent / fake RPC 服务与测试线程 | 已结束并确认无任务进程残留 |

跳过项是需显式启用的 RoboCasa GPU 集成测试。回归覆盖公共 BaseEnvClient 默认 reset、YAM 连接不 reset、单步 RPC 拒绝多步输入、三图 batch、人工成功与预算中止的区分、手眼方向、静止投影、旧回合拒绝、stop、finish 和记忆导出。

本轮新增回归覆盖无 VLA 默认/显式关闭、工具与 prompt 可用性、左右夹爪互不修改、实测夹爪返回、ready/start 分工、外部 reset 后 recipe 隔离，以及无 VLA 原语 → 公共写文件工具 → 真实 MemoryManager → suite/task/index 的完整本地流程。成功标记与硬件仍为 fake，不代表真实任务成功。

## 未完成

真实控制机依赖构造、相机/CAN 时序、标定绝对精度、原语到位误差、SFT recipe/norm_stats/checkpoint、真实 Agent/Explore 和 recipe 回放仍待验收。此版本不自动 fold/home、清错、重新使能或松扭矩。

尚未同步 `/home/yambox/cynws/RPent`；缺可用 SSH 连接信息。部署顺序见 [DEPLOYMENT.zh-CN.md](DEPLOYMENT.zh-CN.md)，研究依据与实验设计见 [INTEGRATION_PLAN.zh-CN.md](INTEGRATION_PLAN.zh-CN.md)。
