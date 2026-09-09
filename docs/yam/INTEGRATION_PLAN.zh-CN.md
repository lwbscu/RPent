> 历史方案：公共 API 的旧结论以当前 upstream/main 和 [部署说明](DEPLOYMENT.zh-CN.md) 为准。2026-09-09 已迁移到 RobotSpec 公共 Explore/finalize 扩展点。

# YAM 接入 RPent：核实后的实现与验收方案

调研日期：2026-09-06。本文区分**本轮源码/联网证据**、**用户报告的现场结果**和**待完成验收**。机器人成功不由离线测试、录像存在或 Agent 的文字声明代替。

## 1. 结论与实施边界

采用三机分工：Agent 机运行 RPent/工具/记忆；推理机运行 YAM Pi0.5；YAM Box 运行 RGBD、标定/FK/IK 和唯一 CAN writer。复用 RLinf fork 的控制运行时与 YAM OpenPI 变换；RPent 只添加 robot extension 与控制/感知适配。先本地写代码和 mock 验证，最后再同步 `/home/yambox/cynws/RPent`。现场已有组件不重写。

基线锁定 `codex/pr133-final-plan` 的 `0cf9d002b317db13a250ccceacc1eac519e9a1e2`；独立工作区 `/home/lwb/Projects/thusigs/yam/RPent`，分支 `yam`，fork `https://github.com/lwbscu/RPent`。原 RPent/其他工作树原状保留。2026-09-06 `git ls-remote upstream main` 为 `d4e9b3a6a7342ab08d3b80a05e558f34170a529b`，不能称指定基线已包含今日最新 main；本轮不混入这些后续变更。

RLinf 来源为 `/home/lwb/Projects/thusigs/yam/RLinf`，标示分支 `fix/yam-review-findings`。该本地副本的 Git pack 损坏，不能以 `git status/log` 可靠证明其版本；实施以实际文件内容为准。未修改该副本或远程源码。现场版本需之后核对。

## 2. 原方案必须修正之处

| 原说法 | 核实与决定 |
|---|---|
| env 固定 pickle TCP，`http://` 也走 pickle | 指定基线已有 `make_rpc_client()` 分流：`http://` 是 JSON+带类型 ndarray 编码，`socket://` 才是 pickle TCP。默认用 HTTP，三机用可信专网或 SSH 隧道，不对公网暴露控制服务。 |
| RoboTwin MODEL_SPEC 对应 qpos14 | 指定基线 LingBot MODEL_SPEC 是 **eef16**；环境支持 qpos14 不代表策略支持 qpos14。YAM 新建独立 Pi0.5 contract。 |
| YAM `use_length=50` | YAM registry 的 `pi05_yam_joint` 是 **action_horizon=30**；首次部署 **use_length=5** 是可执行的初始工程选择，需测推理时延后调节；不能照搬 LingBot 的 50。 |
| 三视角名完全一样 | 物理角色对应，但 RLinf env 名为 `top_rgb/left_rgb/right_rgb`，RPent 名为 `top/left/right`；OpenPI slots 为 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`。要显式映射。 |
| 14 维逐字节一致即可复用权重 | 仅向量布局一致。机械零位、夹爪标度、相机标定、控制频率与训练分布都不同。必须使用 YAM 数据适配及匹配 checkpoint/norm_stats。 |
| RLinf env 直接提供 RGBD | 当前 YAM env 明确拒绝 enable_depth，通用 RealSense 相机不暴露同步元数据。RPent 需增加三路 RGBD 捕获适配，仍复用控制和 IK。 |
| 构造 EnvClient 不会动 | `BaseEnvClient.__init__()` 会调用 reset。YAM 覆盖该行为，只连接/读取，不把重连当新 episode。 |
| reset=回折叠位 | 回关节位不等于还原场景；折叠运动也不能默认从任意持物状态执行。人工布置、episode 授权与 home/park 是不同动作。 |
| `_safe_move_to` 全路径保护已足够 | 现有脚本递归深度达到 2 后，即使净空不足也会警告后执行；且只检查采样 TCP，未覆盖全部连杆和双臂碰撞。不能直接作为常驻服务安全实现。 |
| 关闭 runtime limits 只取消摆率限制 | 同时关闭 hard position clipping 与 measured±delta clipping。适配层必须保留硬关节范围、有限数检查和目标轨迹节拍；避免重新引入已造成 PD 推不动的 measured±delta 限制。 |
| cam2world 在工具调用时 FK 现算即可 | 腕相机必须使用图像捕获时附近的关节状态；旧图配新 FK 会产生系统误差。快照必须绑定 RGB、对齐深度、K、外参、关节与时间戳。 |
| 两臂共享 base | 两个 FK 都以各自安装 base 为原点。定义 world=left_base，利用两份 top-camera 外参或显式安装变换关联 right_base。 |
| `finish(status='success')` 可以写人工成功 | 人工裁决必须由操作员通道写入，与 episode ID 绑定；Agent 只能读取。未判定应为 unknown/pending，不等于成功。 |
| Explore 照抄 LIBERO prompt | 不能保留“每个任务一定可解”“随时自动 reset”等仿真假设。重试须人工恢复现场，有尝试预算和中止；评测不重试、不写入记忆。 |

来源：本地 `rpent/utils/rpc/{client_utils,http_rpc}.py`、`robots/robotwin/{robot_spec,primitives,toolkit}.py`、`rpent/robots/components/env_client_base.py`；RLinf `yam/{config,control_runtime,kinematics}.py`、`openpi/dataconfig/{__init__,yam_dataconfig}.py` 和 `toolkits/calibration/hover_over_pixels.py`。

## 3. 数据与坐标契约

### 3.1 状态/动作与策略

外部状态和动作固定为 absolute qpos14：`[left_q0..q5, left_grip, right_q0..q5, right_grip]`，关节单位 rad，夹爪 `0=闭、1=开`。不能将夹爪、关节速度或 EEF 位姿当作 qpos。

| 层 | top | left wrist | right wrist |
|---|---|---|---|
| RLinf 采集 | `top_rgb` | `left_rgb` | `right_rgb` |
| LeRobot features | `image` | `extra_view_image-0` | `extra_view_image-1` |
| RPent | `top` | `left` | `right` |
| 推理输入 | `main_images[:,…]` | `extra_view_images[:,0,…]` | `extra_view_images[:,1,…]` |
| OpenPI | `base_0_rgb` | `left_wrist_0_rgb` | `right_wrist_0_rgb` |

推理输入为 RGB uint8 HWC：主图 `[1,H,W,3]`，左右腕 `[1,2,H,W,3]`，`wrist_images=None`，`states=[1,14]`，一条任务语言。复用 RLinf `openpi_rlinf` eval wrapper 及其 `YamInputs/YamOutputs` 变换；工厂选择与现有 YAM eval YAML 一致，不另建 backend 选择器。

YAM dataconfig 对 12 个关节应用 DeltaActions，对夹爪不做 delta；这是**训练内部变换**。推理先反归一化再 AbsoluteActions，控制机得到绝对关节目标；RPent 不再加第二次当前 qpos。norm_stats 必须来自相同处理管线及 YAM 数据。OpenPI 上游的预训练夹爪约定与本地 YAM 不完全相同，不能只看 14D 就替换 ALOHA stats。[OpenPI 官方归一化说明](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)

### 3.2 坐标与像素反投影

统一 `T_A_from_B` 表示把 B 中的列向量变换到 A。world 定义为 left_base。直接读取 RLinf `solve_handeye.py` 产出的 `extrinsics.json`，不另设内联标定结构。文件中的历史字段 `T_base_to_cam` 实际是 base-from-camera；`T_grasp_to_cam` 同样实际是 grasp-from-camera，须直接使用，不能取逆。依据是 `solve_handeye.py` 的生成方程 `FK @ t_tcp_cam @ board_in_camera`。进入适配层后按明确语义命名：

```
T_world_from_top = T_leftbase_from_top
T_world_from_rightbase = T_leftbase_from_top @ inv(T_rightbase_from_top)
T_world_from_leftcam = FK_left(q_left) @ T_leftgrasp_from_leftcam
T_world_from_rightcam = T_world_from_rightbase @ FK_right(q_right)
                       @ T_rightgrasp_from_rightcam
```

相机采用 CV 光学坐标：x 右、y 下、z 前；使用彩色对齐深度及对应内参/畸变模型，深度换成米。禁止沿用 RoboTwin 的 GL 坐标 `y`/`z` 取负。针孔近似只适用于已去畸变图像或可忽略的畸变，RealSense SDK 的反投影接口应带实际模型和参数。无效/零深度返回无效点，不能默默当桌面目标。[RealSense 官方投影文档](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)

动作 RPC 携带观测的 `expected_episode_id`；在推理/规划期间若操作员开始新回合，旧动作请求拒绝执行。

每次 `observe()` 返回一个快照，包含三视角 RGBD、K、CV cam2world、图像设备时间、host 接收时间、关节测量时间、snapshot ID。三相机未做硬件同步，不宣称同时曝光；host 到达时间也不是曝光时间。首版动作后静止读快照，腕相机时间差必须可检查并保留在记录里；未来若要边动边定位，需要关节历史插值与时钟域标定。

### 3.3 运动语义

所有 Cartesian xyz/quat 使用 world=left_base，quat 为 wxyz。`move_to(arm,xyz,quat,gripper)` 执行完整规划路径；删除初版中没有执行语义的 `substeps` 参数。一般 move_to 的显式姿态不能被 IK 扫描悄悄修改；视觉 hover 可另外请求候选姿态，并把实际采用姿态返回。垂直顶抓不可达是用户报告的工作空间限制，GUIDE 需注明前下逼近、Y_site 捏合、-Z_site 逼近的现场约定。

RPC 服务端拥有 30 Hz 执行节拍，网络延时不能决定 CAN 控制周期。LLM 不逐帧控制。默认短 VLA 块；不以“推理机能跑”推断稳定 30 Hz 闭环。5 帧=0.167 s 的下发窗口，周期还包含图像、网络和推理时延。无 RTC 时块间保持，延迟实测后再决定长度。

## 4. 安全与人工监督的职责

复用 `YamControlRuntime` 作为唯一 follower writer；几何模块只规划，相机模块只采集，RPC stop 只置事件，由执行循环在下一边界 hold。停止、失败、步数用尽、人工结束后不继续消费块。stop ACK 与已经 hold 的状态分开；不能声称一个置位 ACK 就是物理急停。操作员保留硬件急停。

保留现场已发生的问题对应措施：相机先启动保活，再连 CAN；640×480@30；位置控制不用 measured-error 摆率裁剪；零重力关闭摩擦补偿。清错/重新使能必须是现场启动操作，运行中不能自动反复 0xFB/0xFC 掩盖故障。不同 CAN 程序不能同时拥有 follower；采集/Pico/hover 与 env_server 互斥运行。

桌面 TCP 检查在配置 `table_z` 后生效；未配置时不执行该检查。它只声明为桌面净空检查，**不等价于碰撞规划**；全连杆、自碰、双臂互撞、夹持物碰撞没有现成证据。首阶段优先单臂，另一臂在已确认的停驻区；双臂交叉工作空间要另行标定和分区或引入完整碰撞模型。配置值应来自现场标定和实际 i2rt 限位，示例值不能代替现场认证。

人工流程采用 episode ID 绑定：操作员布置并确认 ready → 显式开始 episode → Agent 执行 → 操作员裁决 success/failure/abort。重连不清零、重试不复用旧成功。自动回 home 与 fold/断电后置，因持物、重力掉落、途中障碍均依赖现场条件。

## 5. 包结构与职责

| 模块 | 职责 |
|---|---|
| `contracts.py` | 纯常量、维度、相机顺序、policy/runtime 契约 |
| `robot_spec.py` / `__init__.py` | 自动发现、CLI、RunConfig、外部 env/VLA 初始化、本地 memory |
| `env_client.py` / `env_server.py` | Base 客户端/Facade 的兼容适配，显式 observe/reset，CPU numpy 边界 |
| `rlinf_env.py` | lazy import RLinf，唯一 runtime writer、执行预算/取消/episode、快照 |
| `cameras.py` / `geometry.py` | 常驻三路 RGBD、明确 CV/world 变换、FK/IK 适配与桌面检查 |
| `primitives.py` | pi05_act、move_to、rotate_wrist、set_gripper、release |
| `toolkit.py` / `tools.py` | 与 RoboTwin 对齐的工具及状态、图片、世界点、动作记录 |
| `vla_server.py` + 公共 `BaseVLAClient` | 复用 YAM OpenPI config/transforms，3 图 + qpos14 输入输出 |
| `prompts/` / `guides/` | 真机限制、工具使用与 explore/eval 区别 |
| `tests/yam/` | 无硬件 fake runtime、RPC、VLA、几何、取消/人工终态测试 |

不执行 `cp -r` 整包后仅字符串替换：robotwin 的 assets、LingBot eef16、GL 相机、sim reset、原生成功判定均不适用。复用公共 `Toolkit/MemoryManager/EnvState/RpcFacade`，通过 YAM 包承接不同实现。

## 6. Explore 与 Memory：软件流程和现场边界

Explore 沿用 LIBERO 的组织思路：独立 planner sessions、每 session 尝试预算、观测/动作轨迹、失败归因、成功 recipe、最终归纳。当前不传 VLA endpoint 即使用原语探索，不注册 pi05_act。真机 reset 消费操作员 ready 回执并更新 episode ID，本身不重建物理场景；操作者负责摆物和裁决。首回合由 operator start 开始，重试由 operator ready + Agent reset 开始。跨 session 读取实际 session 记录与 inbox 笔记，不能宣称新 planner 自动复原了场景。

本地 memory 使用一个官方 MemoryManager corpus，scope 为 global/suite；task 保存 audit/recipe。工具的局部 `success=True` 只表示原语完成，不能发布成功任务记忆。只有 fresh 当前 episode 的人工成功证据可生成成功 recipe/audit；当前 audit 记录 task、episode、动作与人工裁决来源；现场验收还须单独附布局标签、标定/权重/norm 指纹和失败尝试数，尚未自动采集这些外部资产指纹。失败笔记暂存 inbox，不冒充已验证技能。

评价时 MemoryManager 只读；冻结 VLA、相机/标定、任务语言、场景布局、预算。人工裁判尽量不知道是哪一组方法。HF memory 上传使用用户自建 repo，先本地验证 corpus，发布与真机运行分开；不自动推向 RLinf 公共 memory。

“1 个任务 ≤5 次尝试收敛”应作为实验目标，不能作保证或强制报告成功的验收条款。LIBERO parse 通过仅验证格式，不能证明 YAM 动作可回放；回放前需要 YAM 自身布局、姿态与 episode 语义相容。[RPent 官方 Explore 说明](https://rpent.readthedocs.io/en/latest/rst_source/usage/libero.html)

## 7. SFT 与评测计划

已具备流式采集和 YAM transforms；用户报告录过 6+ 条真实数据，本轮未打开现场数据或运行训练，不能标记 SFT/权重 ready。阶段 1 仍需训练 recipe、norm_stats、有效 checkpoint 和推理输出验证。

建议先选择单臂、可重复布置、易人工判定的 pick-place 任务，另一臂固定；50 条/任务可作起始数据预算，不是成功率依据。数据覆盖起始位置、物体外观、接近方向、小幅背景变化，保留真实失败/恢复片段但不要默认全用于成功 imitation。按 episode/采集 session 划分 train/validation，禁止帧级随机切分导致泄漏；norm_stats 只从训练集产生。核对 state 与 accepted action 时间关系、失效帧、夹爪极性、三相机顺序和成功标记。训练先小批 overfit/离线回放，再扩大。

阶段划分：

1. **本地契约**：fork yam 分支；mock 的 env/client/toolkit/recipe/VLA 通路、CPU serialization、取消/错误/预算、几何合成测试。无需 checkpoint，也不运行硬件。
2. **控制机观测**：本地验收后同步；核对进程归属、CAN/USB、源码/标定指纹；相机常驻、三视角图、深度单位/变换、静止反投影；此阶段不下动作。
3. **原语真机**：操作员现场确认后，先小幅单关节/夹爪，再抬高净空的 move_to；录像、requested/accepted/measured 轨迹、失败/stop。不从任意姿态自动折叠。
4. **无 VLA Agent 与 Explore**：先单次双臂原语工具闭环，再人工 reset 的 ≤5 次尝试流程；发布当前成功 episode 的 recipe，并验证下次读取 memory。双臂几何动作按指定 arm 交替执行，尚无同时双臂 Cartesian 规划。
5. **VLA（后置）**：独立完成 SFT/统计量/权重；先离线真实观测推理看 qpos14、关节范围和块内连续性，再短块现场 rollout。短块反复感知是闭环，“开环一次”需明确仅一块且不能等同任务完成。
6. **比较实验**：冻结 VLA；VLA only / Agent primitives+VLA / Agent+validated memory。任务和初始布局成对、不同 trial 顺序随机化，报告逐任务成功率、原始次数、置信区间、人工介入/安全中止/用时/模型调用数。5–10 任务×5 rollout 只够试点，不能当高精度总体结论；正式样本量据效果差与区间宽度确定。

阶段 0 的标定 std 1.7–2.8 mm、hover 往返和流式数据为**用户提供的现场证据**，当前窗口尚未独立复测。重复性 std 也不等于绝对精度：之后需要未参与求解的点/姿态验证，并覆盖左右工作空间。

工期按依赖估算，不承诺固定 1 周：本地适配取决于接口缺口；控制机验证取决于现场可用性；SFT 取决于数据/GPU/首轮学习曲线。数据训练与原语探索可并行；仅 VLA 驱动的 Agent 验收依赖训练权重。

## 8. 联网一手来源

- [RPent 官方仓库](https://github.com/RLinf/RPent)：当前 README 标注 LIBERO、RoboTwin 与 Explore；真机条目不能作为已有 YAM 实现证据。
- [RoboTwin 官方接入文档](https://rpent.readthedocs.io/en/latest/rst_source/usage/robotwin.html)：双臂 agent 参考，具体字段以锁定提交为准。
- [新增机器人官方文档](https://rpent.readthedocs.io/en/latest/rst_source/development/add_robot.html)：RobotSpec 与工具/运行时分层；在线文档可能随 main 更新。
- [i2rt 官方 SDK](https://github.com/i2rt-robotics/i2rt)：YAM 与 SDK 背景；实际命令/参数以现场 fork 使用的 SDK 签名核对。
- [RLinf 官方仓库](https://github.com/RLinf/RLinf)：训练基础设施背景；本用户 YAM fork 的新增功能不能归为已 upstream。
- [OpenPI norm_stats](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)：训练/推理统计量与动作定义必须一致。
- [RealSense 投影与反投影](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)：米制深度、光学坐标、内参/畸变/外参含义。

以上链接仅支持相应公共契约。CAN 故障、机型工作空间、现场标定精度和采集成功均不通过公共网页推断。
