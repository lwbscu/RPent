# YAM 本地接入验收

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
| 人工 ready / verdict / episode ID / stop | 回合确认与动作归属；拒绝旧回合动作，取消后 hold；ACK 不等于物理急停 |
| step 图片 / episode.mp4 / 成功 recipe | 现场复核和记忆导出；不保留 Dashboard 专用视频支线 |
| openpi_rlinf eval factory | 对齐现有 YAM eval YAML；预测 horizon 30，仅执行前 5 帧 qpos14 |

已删除专用 VLA client/元数据握手、无调用 helper、act 别名、无效参数、状态重复缓存、Dashboard 配置、本地 VLA 自动启动、额外 root 环境变量、另一套标定 schema、未接线 smoke JSON 和开发过程文件。

`table_z` 未配置时不启用桌面检查；配置后保留采样 TCP 净空检查。它不覆盖连杆、自碰、双臂互撞或接触力，现场配置与首轮单臂验证仍不可省略。

## 最终验证

环境：Python 3.11.14，`.venv311`，RPent editable install；ruff 0.15.22；pyrealsense2 2.58.4.10922。测试 SDK 只调用几何函数，不访问硬件。

| 检查 | 结果 |
|---|---|
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv311/bin/python -m pytest tests/yam tests/robotwin tests/robocasa -q` | **74 passed, 1 skipped** |
| ruff check / format、compileall、YAM CLI help、git diff --check | 通过 |
| 子 Agent / fake RPC 服务与测试线程 | 已结束并确认无任务进程残留 |

跳过项是需显式启用的 RoboCasa GPU 集成测试。回归覆盖公共 BaseEnvClient 默认 reset、YAM 连接不 reset、单步 RPC 拒绝多步输入、三图 batch、人工成功与预算中止的区分、手眼方向、静止投影、旧回合拒绝、stop、finish 和记忆导出。

## 未完成

真实控制机依赖构造、相机/CAN 时序、标定绝对精度、原语到位误差、SFT recipe/norm_stats/checkpoint、真实 Agent/Explore 和 recipe 回放仍待验收。此版本不自动 fold/home、清错、重新使能或松扭矩。

尚未同步 `/home/yambox/cynws/RPent`；缺可用 SSH 连接信息。部署顺序见 [DEPLOYMENT.zh-CN.md](DEPLOYMENT.zh-CN.md)，研究依据与实验设计见 [INTEGRATION_PLAN.zh-CN.md](INTEGRATION_PLAN.zh-CN.md)。
