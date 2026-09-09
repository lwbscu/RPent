# YAM 无 VLA 部署（yambox，2026-09-09）

本次在控制机 `/home/yambox/cynws/RPent` 的 fork `yam` 工作线上合并 upstream/main。
实际 fetch 基线：fork `ba08a259840f0db50f4bfddf06d8d9950e8af8c7`，
upstream `24541d6ff5f6d4398cf0dbb63ed827e7198237cc`。
软件验证不代表真机验收。当前现场状态和基线保存在未跟踪的 `logs/yambox_deployment/`。

## 环境与现场配置

RPent 使用独立 `.venv` 安装本项目；其中 `yam_rlinf_runtime.pth` 引入既有
`/home/yambox/cynws/RLinf/.venv/lib/python3.11/site-packages`，复用 i2rt、MuJoCo、
SciPy、RealSense 等真机库。NumPy 保持 2.2.6，没有改动 RLinf 的虚拟环境或控制参数。
规划器默认使用本机已登录的 Codex；没有 VLA，也不加载历史 smoke 权重。

本地配置（不提交、不上传）：

- `logs/yambox_deployment/site/task_a.json`
- `logs/yambox_deployment/site/task_b.json`

两个配置使用相同任务名 `tabletop_cleanup`、不同 seed 标签和语言映射，语言取自当前
RLinf `collect_tabletop_cleanup_1.sh` 与 `_2.sh`。三瓶全部入袋、同色勺碗、碗归位，
左右都按未镜像 top 画面。几何原语中没有品牌/袋子的固定坐标规则。

设备字段继承当前数采配置，包括 gravity 和摩擦开关；RPent 不自动清故障，
`enable_auto_recovery=false`。当前原语每步指令限制 0.02 rad，规划间隔 0.01 rad，
指令领先实测不得超过 0.1 rad；它们不修改 VR/主臂遥操参数。RPent 在下发前拒绝越过硬关节限位的目标。
RPent 构造 runtime 时关闭其重复逐关节限幅，避免实际发送目标偏离已检查路径；
RPent 的硬限位、步幅、最新实测路径与跟踪误差检查保留，SDK 硬限位仍生效。
`reset.tolerance=0.04` 是数采回位容差，不用于此处笛卡尔成功判断。

`collision_guard.enabled=true`，模型间距设为 0.01 m；缺模型资产/基座外参时拒绝。
`require_table_guard=true`：桌面几何必须在新鲜观察后核实并填写；当前未填写，所有原语运动均被拒绝，只允许观察。
水平桌面可用 `table_z`。有限倾斜桌面使用 `table_surface`，两者不能同时配置：
`plane_z_equals_ax_by_c` 为左基座世界坐标下的 `[a,b,c]`，
`footprint_xy` 为沿边界顺序排列的严格凸多边形，`depth_m` 为沿世界 Z 向下的禁入深度，
`uncertainty_m` 为定位不确定性余量。参数需现场验证，不能直接将单帧拟合当作验收。
可选 `collision_guard.base_link2_clearance_m` 仅调整同臂基座与 link2 的正距离余量；
默认仍等于全局余量，不排除该几何对或放宽跨臂检查。实际模型在常用姿态约有 9.4 mm
间距，接近关节极限时会缩小甚至穿透，因此不能简单忽略。
采样模型检查不包含相机、线缆、主臂、
夹持物、纸袋、碗和其他场景障碍，也不是连续碰撞或实际跟踪保证。

## 首次接管与服务

必须取得现场人员确认：四臂支撑稳定、可急停、工作区和允许范围明确、其他数采/遥操
正常退出且全程不重启。RPent 按通道加进程锁并检查现存 SocketCAN 订阅，但其他程序
不共享该锁，仍需现场保证排他。不能擅自终止用户进程。

首次 observe（包括 Agent 连接）会打开三相机、连接电机并输出保持扭矩。
`gripper_limits=null` 会触发夹爪寻两端标定，必须保证夹爪无物且运动空间清楚。
服务退出可能释放扭矩，退出前必须支撑四臂；软件 stop 不是硬件急停。

完成现场确认后启动 A 环境服务：

```bash
cd /home/yambox/cynws/RPent
export RPENT_RLINF_ROOT=/home/yambox/cynws/RLinf
.venv/bin/python -m robots.yam.env_server \
  --config logs/yambox_deployment/site/task_a.json \
  --transport http --host 127.0.0.1 --port 8110
```

服务 `healthz` 与静态 metadata 不打开硬件；observe/reset/render 则可能初始化硬件。
先核实三图、相机身份、静止腕图投影、实际关节与 TCP、标定多点投影和活动范围，
补齐桌面配置。原始标定已用当前模型重算且数值完全一致，但这不证明相机安装未移动。

## 人工回执

首回合用 operator_control 的 status 取得 episode ID，再由现场确认写 start：

```bash
.venv/bin/python -m robots.yam.operator_control \
  --config logs/yambox_deployment/site/task_a.json --event status
.venv/bin/python -m robots.yam.operator_control \
  --config logs/yambox_deployment/site/task_a.json \
  --event start --episode-id ACTUAL_EPISODE_ID --note '现场已确认摆场和允许范围'
```

status 可能调用 observe，须在接管确认后使用。start 消费 ready 并开始首回合，
不自动复位机器人。Explore 要求人工恢复场景时，实际恢复后写 `--event ready`，
由 Agent reset 消费回执；不能仅根据等待时长、Agent 声明或脚本成功写 ready/success。
最终 success/failure/abort 也必须来自真实现场反馈，绑定当前 episode ID。

## 无 VLA Agent 与 Explore

用配置中的实际语言启动，确保 ENV 和 Agent task/seed/step limit 相同：

```bash
cd /home/yambox/cynws/RPent
export RPENT_RLINF_ROOT=/home/yambox/cynws/RLinf
SITE_CONFIG=logs/yambox_deployment/site/task_a.json
export CODEX_BIN=/home/yambox/.vscode-server/extensions/openai.chatgpt-26.903.61454-linux-x64/bin/linux-x86_64/codex
TASK_LANGUAGE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_language"])' "$SITE_CONFIG")"
.venv/bin/python -m rpent.cli.main --robot yam --planner codex \
  --task-name tabletop_cleanup --task-language "$TASK_LANGUAGE" --seed 0 \
  --max-episode-steps 12000 --env-endpoint http://127.0.0.1:8110 --without-vla \
  --memory-profile local --memory-dir /home/yambox/cynws/RPent/memory/yam \
  --reasoning-effort high --planner-timeout-s 1800
```

Explore 使用同一命令追加：

```bash
--explore --explore-sessions 1 --explore-attempts-per-session 3
```

本机 SDK 随附的 Codex 0.147.0 启动默认模型时返回版本过旧；上述 `CODEX_BIN`
使用已安装的 0.153.4，已实际完成 Agent 调用。扩展升级后需核对该路径。

先验收 A，再由现场恢复布局并换 B 服务配置与 `--seed 1`、B 语言；保持其他控制代码
相同。每次抓、提、放后必须重新观察，执行成功不能替代物体状态验证。

## 公共 Memory 与验收

复用 RobotSpec 能力、公共 Toolkit/MemoryManager 和 finalize_run，不保留 YAM CLI 特判。
默认记忆只在本地。Explore 草稿在 `memory/yam/_internal/inbox/<tag>/`；成功 recipe 为
`<tag>_recipe.jsonl`，按公共机制发布。失败 audit 保留，不能标成成功 recipe。
新 episode 需要真实检索并应用经验，文件存在本身不算复用证据。

每次运行检查 `result.json` 的 `environment_success`；`planner_finish_request` 仅是
Agent 声明，不能单独据其 success 字段判定真机成功。保留三视角、请求/接受/实测、
每次抓放后的证据和最终人工裁决。A/B 真机结果、经验的真实应用目前均未验收。
